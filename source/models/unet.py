import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class Up_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.in_channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x

    
class Down_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.in_channels
        x = self.conv(x)
        return x

def zero_(m):
    for p in m.parameters():
        nn.init.zeros_(p)
    return m

class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, E, dropout=0.0, groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.embedding = nn.Linear(E, 2 * out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero_(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, e):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        s, b = self.embedding(F.silu(e))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + s) + b
        h = self.dropout(F.silu(h))
        h = self.conv2(h)
        return self.skip(x) + h


class Attention_Block(nn.Module):
    def __init__(self, channels, num_heads=1, head_channels=None, groups=32):
        super().__init__()
        if head_channels is not None:
            assert channels % head_channels == 0
            num_heads = channels // head_channels
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.proj_out = zero_(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        assert x.shape[1] == self.channels
        B, C, *spatial = x.shape
        h = self.norm(x).reshape(B, C, -1)
        # (B, 3C, L) -> 3 x (B, heads, L, head_dim)
        q, k, v = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, -1).transpose(-1, -2).unbind(1)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(-1, -2).reshape(B, C, -1)
        h = self.proj_out(h)
        return x + h.reshape(B, C, *spatial)


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Embed_Sequential(nn.Sequential):
    # passes the timestep embedding to the layers that take it
    def forward(self, x, e):
        for layer in self:
            x = layer(x, e) if isinstance(layer, Residual_Block) else layer(x)
        return x


class UNet(nn.Module):
    # attn_levels index into channel_mult (0 = input resolution), so attention placement
    # is the same whether the input is a training crop or a full 4K latent
    def __init__(self, in_channels=16, out_channels=16, cond_channels=16, model_channels=128,
                 channel_mult=(1, 2, 2, 4), num_res_blocks=2, attn_levels=(2, 3),
                 head_channels=64, dropout=0.0, groups=32):
        super().__init__()
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.factor = 2 ** (len(channel_mult) - 1)
        E = 4 * model_channels

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, E),
            nn.SiLU(),
            nn.Linear(E, E),
        )

        def res(c_in, c_out):
            return Residual_Block(c_in, c_out, E, dropout, groups)

        def attn(c):
            return Attention_Block(c, head_channels=head_channels, groups=groups)

        ch = model_channels
        skip_chs = [ch]
        self.input_blocks = nn.ModuleList([Embed_Sequential(nn.Conv2d(in_channels + cond_channels, ch, 3, padding=1))])
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [res(ch, mult * model_channels)]
                ch = mult * model_channels
                if level in attn_levels:
                    layers.append(attn(ch))
                self.input_blocks.append(Embed_Sequential(*layers))
                skip_chs.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(Embed_Sequential(Down_Block(ch, ch)))
                skip_chs.append(ch)

        self.middle_block = Embed_Sequential(res(ch, ch), attn(ch), res(ch, ch))

        self.output_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                layers: list[nn.Module] = [res(ch + skip_chs.pop(), mult * model_channels)]
                ch = mult * model_channels
                if level in attn_levels:
                    layers.append(attn(ch))
                if level != 0 and i == num_res_blocks:
                    layers.append(Up_Block(ch, ch))
                self.output_blocks.append(Embed_Sequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(groups, ch),
            nn.SiLU(),
            zero_(nn.Conv2d(ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, t, cond=None):
        # x: noisy latent (B, in_channels, H, W), t: timesteps (B,),
        # cond: VAE latent of the bicubic-upscaled LR image (B, cond_channels, H, W), concatenated onto x
        # (cond_channels=0 builds an unconditional model that ignores cond)
        if self.cond_channels:
            assert cond is not None and cond.shape[1] == self.cond_channels
            x = torch.cat([x, cond], dim=1)

        # pad so every Down_Block halves evenly (a 4K latent is 270 x 480, and 270 isn't divisible by 8)
        H, W = x.shape[-2:]
        x = F.pad(x, (0, -W % self.factor, 0, -H % self.factor), mode="reflect")

        e = self.time_embed(timestep_embedding(t, self.model_channels))
        hs = []
        h = x
        for block in self.input_blocks:
            h = block(h, e)
            hs.append(h)
        h = self.middle_block(h, e)
        for block in self.output_blocks:
            h = block(torch.cat([h, hs.pop()], dim=1), e)
        return self.out(h)[..., :H, :W]



