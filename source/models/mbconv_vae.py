import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(channels):
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)



class SEBlock(nn.Module):
    def __init__(self, channels, sequeeze_channels):
        super().__init__()

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, sequeeze_channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.SiLU(),
            nn.Conv2d(max(1, sequeeze_channels), channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)




class MBConv(nn.Module):
    def __init__(self, c_in, c_out, expansion=4, stride=1):
        super().__init__()

        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        hidden = c_in * expansion
        self.body = nn.Sequential(
            nn.Conv2d(c_in, hidden, kernel_size=1, stride=1, padding=0, bias=False), norm(hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1, groups=hidden, bias=False), norm(hidden), nn.SiLU(),
            SEBlock(hidden, c_in // 4),
            nn.Conv2d(hidden, c_out, kernel_size=1, stride=1, padding=0, bias=False), norm(c_out),
        )
        self.skip = nn.Identity() if c_in == c_out and stride == 1 else nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        return self.skip(x) + self.body(x)



class MBConvVAE(nn.Module):
    def __init__(self, base_channels=32, blocks_per_stage=2, latent_channels=16, max_channels = 512):
        super().__init__()

        if min(base_channels, max_channels, blocks_per_stage, latent_channels) < 1:
            raise ValueError("All architecture parameters must be positive integers.")

        self.latent_channel = latent_channels
        encoder = [nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)]
        c_in = base_channels
        stages = []

        growth = (max_channels - base_channels) // (3 * blocks_per_stage)

        for _ in range(3):
            transtions = []
            for _ in range(blocks_per_stage):
                c_out = c_in + growth
                encoder.append(MBConv(c_in, c_out))
                transtions.append((c_in, c_out))
                c_in = c_out
            encoder.append(MBConv(c_in, c_in, stride=2))
            stages.append(transtions)
        encoder += [norm(c_in), nn.SiLU()]
        self.encoder = nn.Sequential(*encoder)
        self.moments = nn.Conv2d(c_in, 2 * latent_channels, kernel_size=1, stride=1, padding=0, bias=False)


        self.from_latent = nn.Conv2d(latent_channels, c_in, kernel_size=3, stride=1, padding=1, bias=False)
        decoder = []
        for transtions in reversed(stages):
            decoder.append(nn.ConvTranspose2d(c_in, c_in, kernel_size=4, stride=2, padding=1, bias=False))
            for prev, curr in reversed(transtions):
                decoder.append(MBConv(curr, prev))
                c_in = prev
        decoder += [norm(c_in), nn.SiLU(), nn.Conv2d(c_in, 3, kernel_size=3, stride=1, padding=1, bias=False), nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)


    def encode(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 8:
            raise ValueError('Expected [B, 3, H, W], with H and W at least 8.')

        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode="replicate")
        mu, logvar = self.moments(self.encoder(x)).chunk(2, dim=1)
        return mu, logvar.clamp(-20.0, 10.0)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


    def decode(self, z, output_size=None):
        x = self.decoder(self.from_latent(z))
        if output_size is not None:
            h, w = output_size
            if not (0 < h <= x.shape[-2] and 0 < w <= x.shape[-1]):
                raise ValueError('output_size must fit within the decoded image.')
            x = x[..., :h, :w]
        return x


    def forward(self, x, sample=None):
        mu, logvar = self.encode(x)
        sample = self.training if sample is None else sample
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(z, x.shape[-2:]), mu, logvar




def vae_loss(reconstruction, target, mu, logvar, beta=1e-4):
    reconstruction_loss = F.l1_loss(reconstruction.float(), target.float())
    mu, logvar = mu.float(), logvar.float()
    kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).mean()
    loss = reconstruction_loss + beta * kl
    return loss, {'loss': loss.detach(), 'l1': reconstruction_loss.detach(), 'kl': kl.detach()}

