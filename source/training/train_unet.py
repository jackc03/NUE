"""Train the latent-diffusion UNet for 4x super-resolution on HR crops, with validation and resume.

Nothing is precomputed: every step makes its own LR input and latents. Each HR (4K) frame is
bicubic-downsampled to 960x540 and bicubic-upsampled back to 4K, the frozen VAE encodes the
HR crop and the matching upscaled-LR crop, and the UNet learns to denoise the HR latent with
the upscaled-LR latent concatenated as its condition.

Every epoch also upscales a few full val frames with the tiled pipeline of unet_inference.py,
and their PSNR picks best.pt: the crop val loss and center-crop samples never showed the
per-tile color drift that cost whole frames 3.6dB against bicubic.

Example: python source/training/train_unet.py --train-dir data/train --val-dir data/val --out-dir runs/unet
Multi-GPU: torchrun --standalone --nproc_per_node=4 source/training/train_unet.py ...
"""

import argparse
import copy
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Repo root, wherever it is cloned
from source.inference.vae_inference import ramp, tile_starts, to_uint8
from source.models.diffusion import GaussianDiffusion
from source.models.mbconv_vae import MBConvVAE
from source.models.unet import UNet
from source.training.train_VAE import EXTENSIONS, EpochSampler, all_reduce_sums, learning_rate

DEFAULT_VAE = Path.home() / 'scratch/runs/mbconv_vae/best.pt'


def upscaled_lr_crop(image, lr_size, box, margin=3):
    # `box` of the image bicubic-downsampled to lr_size and bicubic-upsampled back to its own
    # size. Only an LR patch `margin` pixels wider than the box is resampled: Pillow clips its
    # kernels at the image edge, not at the resize box, and margin 3 covers the 2px reach of the
    # upsampling kernel, so this equals resizing the whole frame twice, pixel for pixel, in ~2ms
    # instead of ~180ms per 4K frame.
    width, height = image.size
    lr_width, lr_height = lr_size
    sx, sy = width / lr_width, height / lr_height
    left, top, right, bottom = box
    x0, y0 = max(math.floor(left / sx) - margin, 0), max(math.floor(top / sy) - margin, 0)
    x1 = min(math.ceil(right / sx) + margin, lr_width)
    y1 = min(math.ceil(bottom / sy) + margin, lr_height)
    patch = image.resize((x1 - x0, y1 - y0), Image.BICUBIC, box=(x0 * sx, y0 * sy, x1 * sx, y1 * sy))
    return patch.resize((right - left, bottom - top), Image.BICUBIC,
                        box=(left / sx - x0, top / sy - y0, right / sx - x0, bottom / sy - y0))


class SRCrops(Dataset):
    # Aligned (HR, upscaled LR) crops as uint8 [3, crop, crop] tensors.
    def __init__(self, root, crop, lr_size, training, limit=0):
        paths = sorted(p for p in Path(root).rglob('*') if p.suffix.lower() in EXTENSIONS)
        if not paths:
            raise ValueError(f'No supported images in {root}')
        if limit and len(paths) > limit:
            # Even spacing spans many clips instead of the first clip's first frames.
            paths = [paths[i * len(paths) // limit] for i in range(limit)]
        self.paths, self.crop, self.lr_size, self.training = paths, crop, lr_size, training

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        width, height = image.size
        if width < self.crop or height < self.crop:
            raise ValueError(f'{path} is smaller than the {self.crop}px crop')
        # Offsets are multiples of 8, so a crop's latent grid lines up with a full-frame encode
        if self.training:
            left = random.randrange(0, width - self.crop + 1, 8)
            top = random.randrange(0, height - self.crop + 1, 8)
        else:
            left = (width - self.crop) // 2 // 8 * 8
            top = (height - self.crop) // 2 // 8 * 8
        box = (left, top, left + self.crop, top + self.crop)
        hr, upscaled = image.crop(box), upscaled_lr_crop(image, self.lr_size, box)
        if self.training and random.random() < .5:
            hr, upscaled = ImageOps.mirror(hr), ImageOps.mirror(upscaled)
        return tuple(torch.from_numpy(np.asarray(crop).copy()).permute(2, 0, 1) for crop in (hr, upscaled))


def to_unit(x):
    # uint8 batch -> float [-1, 1], as the VAE was trained
    return x.float() / 127.5 - 1


@torch.no_grad()
def encode(vae, x, latent):
    # Images in [-1, 1] -> VAE posterior means, normalized per channel. The posterior std is
    # ~0.03 against a latent std of ~0.8, so sampling it would add next to nothing.
    mu = vae.encode(x)[0].float()
    return (mu - latent['mean'].view(-1, 1, 1)) / latent['std'].view(-1, 1, 1)


@torch.no_grad()
def decode(vae, z, latent):
    return vae.decode(z * latent['std'].view(-1, 1, 1) + latent['mean'].view(-1, 1, 1))


def to_tensor(image):
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0)  # uint8 [1, 3, H, W]


def gaussian_blur(x, sigma):
    # Separable Gaussian blur of [N, C, H, W], reflect-padded at the borders.
    radius = math.ceil(3 * sigma)
    kernel = torch.exp(-torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype).square() / (2 * sigma ** 2))
    kernel = (kernel / kernel.sum()).repeat(x.shape[1], 1, 1, 1)  # [C, 1, 1, K]
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode='reflect'), kernel, groups=x.shape[1])
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode='reflect'), kernel.transpose(2, 3), groups=x.shape[1])


@torch.no_grad()
def restore_low_frequencies(sample, upscaled, sigma=2.):
    # The sample's high frequencies over the upscaled LR's low ones. Sampled latents drift in
    # brightness and color, tile by tile, by up to ~60 levels on flat 4K frames, while bicubic
    # upsampling keeps all the LR frame holds. sigma=2 passes about what a 4x downscale keeps.
    # In float32 even under autocast: bfloat16 would add ~1 level of error to every pixel.
    with torch.autocast(device_type=sample.device.type, enabled=False):
        sample, upscaled = sample.float(), upscaled.float()
        return sample - gaussian_blur(sample, sigma) + gaussian_blur(upscaled, sigma)


@torch.no_grad()
def tiled_upscale(vae, model, diffusion, latent, upscaled, noise, tile, overlap, batch, steps):
    # upscaled: [1, 3, H, W] in [-1, 1] with H, W multiples of 8; noise: [1, C, H/8, W/8].
    # Each tile is encoded, sampled and decoded alone, then cross-faded into the frame, which
    # then takes its low frequencies from `upscaled`.
    height, width = upscaled.shape[-2:]
    tile_h, tile_w = min(tile, height), min(tile, width)
    boxes = [(top, left) for top in tile_starts(height, tile_h, overlap)
             for left in tile_starts(width, tile_w, overlap)]
    out = torch.zeros(1, 3, height, width, device=upscaled.device)
    total = torch.zeros(1, 1, height, width, device=upscaled.device)
    for first in range(0, len(boxes), batch):
        chunk = boxes[first:first + batch]
        cond = encode(vae, torch.cat([upscaled[..., top:top + tile_h, left:left + tile_w]
                                      for top, left in chunk]), latent)
        start = torch.cat([noise[..., top // 8:(top + tile_h) // 8, left // 8:(left + tile_w) // 8]
                           for top, left in chunk])
        z = diffusion.ddim_sample(model, cond, steps, noise=start)
        for (top, left), y in zip(chunk, decode(vae, z, latent).float()):
            weight = (ramp(tile_h, overlap, top > 0, top + tile_h < height)[:, None]
                      * ramp(tile_w, overlap, left > 0, left + tile_w < width)).to(out.device)
            out[..., top:top + tile_h, left:left + tile_w] += weight * y
            total[..., top:top + tile_h, left:left + tile_w] += weight
    return restore_low_frequencies(out / total, upscaled)


@torch.no_grad()
def latent_statistics(vae, dataset, crops, args, rank, world, device):
    # Per-channel mean and std of HR latents over random training crops from every rank. The
    # channels differ (std 0.7 to 3.4 for the MBConv VAE), so each is normalized on its own to
    # match the unit-variance noise. HR statistics normalize the LR condition too.
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:crops]
    loader = DataLoader(Subset(dataset, order[rank::world].tolist()), batch_size=args.batch_size,
                        num_workers=args.workers)
    totals = torch.zeros(3, vae.latent_channel, dtype=torch.float64, device=device)
    for hr, _ in loader:
        with torch.autocast(device_type='cuda', enabled=args.amp, dtype=torch.bfloat16):
            mu = vae.encode(to_unit(hr.to(device)))[0].double()
        totals[0] += mu[:, 0].numel()
        totals[1] += mu.sum((0, 2, 3))
        totals[2] += mu.square().sum((0, 2, 3))
    if dist.is_initialized():
        dist.all_reduce(totals)
    count, total, squares = totals
    mean = total / count
    std = (squares / count - mean.square()).sqrt()
    return dict(mean=mean.float(), std=std.float())


@torch.no_grad()
def update_ema(ema, model, decay):
    for averaged, current in zip(ema.parameters(), model.parameters()):
        averaged.lerp_(current, 1 - decay)


def psnr(x, y):
    # In [0, 1] RGB, over all pixels and images
    return -10 * math.log10(max(((x.float() - y.float()) / 2).square().mean().item(), 1e-12))


def save_preview(upscaled, sample, hr, path, rows=4):
    # One row per val crop: bicubic-upscaled LR | diffusion sample | HR.
    grid = []
    for i in range(min(rows, len(hr))):
        panels = [((x[i].detach().float().cpu().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy()
                  for x in (upscaled, sample, hr)]
        grid.append(np.concatenate(panels, axis=1).round().astype(np.uint8))
    Image.fromarray(np.concatenate(grid, axis=0)).save(path)


def save_checkpoint(path, **payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--val-dir', required=True)
    parser.add_argument('--vae', default=str(DEFAULT_VAE), help='Frozen VAE checkpoint from train_VAE.py')
    parser.add_argument('--out-dir', default='runs/latent_unet')
    parser.add_argument('--resume', default=None, help='A checkpoint from this script')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=16, help='Per GPU')
    parser.add_argument('--crop', type=int, default=256, help='HR crop side, a multiple of 8')
    parser.add_argument('--lr-size', type=int, nargs=2, default=(960, 540), metavar=('W', 'H'),
                        help='Frames are bicubic-downsampled to this size, then upsampled back')
    parser.add_argument('--val-limit', type=int, default=2000,
                        help='Val frames, spread evenly over the sorted list (0 = all)')
    parser.add_argument('--val-frames', type=int, default=8,
                        help='Full val frames upscaled each epoch, spread evenly; their PSNR picks best.pt')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=1000, help='Linear lr warmup')
    parser.add_argument('--lr-decay-epochs', type=int, default=5,
                        help='Cosine-decay lr to --min-lr over this many final epochs (0 = constant)')
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--ema-decay', type=float, default=.9999)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--schedule', choices=('linear', 'cosine'), default='linear')
    parser.add_argument('--prediction', choices=('v', 'eps'), default='v')
    parser.add_argument('--sample-steps', type=int, default=50,
                        help='DDIM steps for the per-epoch val samples and preview')
    parser.add_argument('--latent-crops', type=int, default=1024,
                        help='Training crops used to estimate latent statistics on a fresh run')
    parser.add_argument('--model-channels', type=int, default=128)
    parser.add_argument('--channel-mult', type=int, nargs='+', default=(1, 2, 2, 4))
    parser.add_argument('--num-res-blocks', type=int, default=2)
    parser.add_argument('--attn-levels', type=int, nargs='*', default=(2, 3))
    parser.add_argument('--head-channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--amp', action='store_true', help='CUDA mixed precision')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint-every', type=int, default=1000,
                        help='Also overwrite latest.pt every N steps (0 = only at epoch end)')
    args = parser.parse_args()
    distributed = 'LOCAL_RANK' in os.environ  # Launched by torchrun
    if distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
        rank, world = dist.get_rank(), dist.get_world_size()
        args.device = f'cuda:{local_rank}'
    else:
        rank, world = 0, 1
    log = print if rank == 0 else (lambda *_, **__: None)
    factor = 8 * 2 ** (len(args.channel_mult) - 1)  # VAE stride times the UNet's downsampling
    if args.crop % 8 or args.crop < factor:
        parser.error(f'crop must be a multiple of 8 and at least {factor} for this channel-mult')
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or min(args.lr_size) < 1:
        parser.error('epochs >= 1, batch-size >= 1, workers >= 0, positive lr-size required')
    if args.checkpoint_every < 0 or args.val_limit < 0 or args.val_frames < 1 or args.latent_crops < 1:
        parser.error('checkpoint-every >= 0, val-limit >= 0, val-frames >= 1, latent-crops >= 1 required')
    if args.lr <= 0 or args.warmup_steps < 0 or not 0 <= args.ema_decay < 1:
        parser.error('lr > 0, warmup-steps >= 0 and 0 <= ema-decay < 1 required')
    if not (0 <= args.lr_decay_epochs <= args.epochs and 0 <= args.min_lr <= args.lr):
        parser.error('0 <= lr-decay-epochs <= epochs and 0 <= min-lr <= lr required')
    if not 1 <= args.sample_steps <= args.timesteps:
        parser.error('1 <= sample-steps <= timesteps required')
    if args.amp and not args.device.startswith('cuda'):
        parser.error('--amp is supported only with CUDA')
    # Per-rank seeds vary the crops and noise; DDP copies rank 0's initial weights to every rank.
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed + rank)
        torch.backends.cudnn.benchmark = True  # Fixed crop size, so the tuned kernels are reused
    device = torch.device(args.device)
    autocast = lambda: torch.autocast(device_type='cuda', enabled=args.amp, dtype=torch.bfloat16)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lr_size = tuple(args.lr_size)
    train_set = SRCrops(args.train_dir, args.crop, lr_size, training=True)
    val_set = SRCrops(args.val_dir, args.crop, lr_size, training=False, limit=args.val_limit)
    if set(train_set.paths) & set(val_set.paths):
        raise ValueError('Training and validation paths overlap')
    val_frames = val_set.paths
    if len(val_frames) > args.val_frames:
        val_frames = [val_frames[i * len(val_frames) // args.val_frames] for i in range(args.val_frames)]
    train_sampler = EpochSampler(len(train_set), rank, world, shuffle=True, seed=args.seed)
    val_sampler = EpochSampler(len(val_set), rank, world, shuffle=False, seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                              drop_last=True, num_workers=args.workers,
                              pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=args.workers, pin_memory=device.type == 'cuda')

    vae_path = Path(args.vae).expanduser().resolve()
    vae_checkpoint = torch.load(vae_path, map_location='cpu', weights_only=True)
    vae = MBConvVAE(**vae_checkpoint['config'])
    vae.load_state_dict(vae_checkpoint['model'])
    vae.to(device).eval().requires_grad_(False)
    vae_id = dict(path=str(vae_path), step=vae_checkpoint['step'])
    del vae_checkpoint

    channels = vae.latent_channel
    config = dict(unet=dict(in_channels=channels, out_channels=channels, cond_channels=channels,
                            model_channels=args.model_channels, channel_mult=tuple(args.channel_mult),
                            num_res_blocks=args.num_res_blocks, attn_levels=tuple(args.attn_levels),
                            head_channels=args.head_channels, dropout=args.dropout),
                  diffusion=dict(timesteps=args.timesteps, schedule=args.schedule, prediction=args.prediction),
                  lr_size=lr_size)
    start_epoch, epoch_samples, step, best_psnr = 0, 0, 0, -math.inf
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if config != checkpoint['config']:
            parser.error('UNet, diffusion and lr-size arguments must match the resumed checkpoint config')
        if vae_id != checkpoint['vae']:
            parser.error(f'The resumed run used the VAE {checkpoint["vae"]}, not {vae_id}')
    diffusion = GaussianDiffusion(**config['diffusion'])
    model = UNet(**config['unet']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        latent = {key: value.to(device) for key, value in checkpoint['latent'].items()}
        start_epoch = checkpoint['epoch']
        epoch_samples = checkpoint['epoch_samples']
        step = checkpoint['step']
        best_psnr = checkpoint.get('best_psnr', -math.inf)  # Older runs picked best.pt by val loss
        log(f'Resumed after {start_epoch} finished epochs, {epoch_samples} images into '
            f'epoch {start_epoch+1}, step={step}')
    else:
        latent = latent_statistics(vae, train_set, args.latent_crops, args, rank, world, device)
    # DDP broadcasts rank 0's weights into `model`, so every rank's EMA starts from the same copy
    train_model = DistributedDataParallel(model, device_ids=[device.index]) if distributed else model
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if checkpoint is not None:
        ema.load_state_dict(checkpoint['ema'])
    del checkpoint
    log(f'train images={len(train_set)} val images={len(val_set)} gpus={world} '
        f'unet params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n'
        f'latent mean={latent["mean"].cpu().numpy().round(3)}\nlatent std={latent["std"].cpu().numpy().round(3)}')

    def save(path, epoch, epoch_samples):
        # epoch = epochs fully trained; epoch_samples = images already trained in the next one.
        save_checkpoint(path, model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                        epoch=epoch, epoch_samples=epoch_samples, step=step, best_psnr=best_psnr,
                        config=config, latent={key: value.cpu() for key, value in latent.items()},
                        vae=vae_id, args=vars(args))

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch, skip=epoch_samples)
        model.train()
        train_sum, seen = 0., 0
        window_sum, window_seen = 0., 0
        last_time, last_step = time.time(), step
        for hr, upscaled in train_loader:
            x = to_unit(torch.cat([hr.to(device, non_blocking=True), upscaled.to(device, non_blocking=True)]))
            lr = learning_rate(epoch + epoch_samples / len(train_set), args)
            lr *= min(1., (step + 1) / max(args.warmup_steps, 1))
            for group in optimizer.param_groups:
                group['lr'] = lr  # Overrides the lr restored with the optimizer state
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                z0, cond = encode(vae, x, latent).chunk(2)
                t = torch.randint(0, diffusion.timesteps, (len(z0),), device=device)
                loss = diffusion.loss(train_model, z0, cond, t, torch.randn_like(z0))
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            update_ema(ema, model, min(args.ema_decay, (1 + step) / (10 + step)))
            loss_sum = loss.item() * len(z0)
            train_sum += loss_sum
            window_sum += loss_sum
            seen += len(z0)
            window_seen += len(z0)
            step += 1
            epoch_samples += len(z0) * world
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                if rank == 0:
                    save(out / 'latest.pt', epoch, epoch_samples)
                now = time.time()
                log(f'epoch {epoch+1} {epoch_samples}/{len(train_set)} images step={step} '
                    f'loss={window_sum / window_seen:.5f} (rank 0) lr={lr:.2e} '
                    f'{(now - last_time) / (step - last_step):.3f}s/step', flush=True)
                last_time, last_step = now, step
                window_sum, window_seen = 0., 0
        epoch_samples = 0

        # Val loss of the EMA weights, with each batch's timesteps and noise fixed across epochs
        val_sum, count = 0., 0
        sample_text = ''
        with torch.inference_mode():
            for batch_number, (hr, upscaled) in enumerate(val_loader):
                hr = to_unit(hr.to(device, non_blocking=True))
                upscaled = to_unit(upscaled.to(device, non_blocking=True))
                generator = torch.Generator(device).manual_seed(args.seed + batch_number * world + rank)
                with autocast():
                    z0, cond = encode(vae, torch.cat([hr, upscaled]), latent).chunk(2)
                    t = torch.randint(0, diffusion.timesteps, (len(z0),), device=device, generator=generator)
                    noise = torch.randn(z0.shape, device=device, generator=generator)
                    loss = diffusion.loss(ema, z0, cond, t, noise)
                val_sum += loss.item() * len(z0)
                count += len(z0)
                if batch_number == 0 and rank == 0:
                    # Full DDIM samples of the first batch, decoded and scored against HR
                    generator = torch.Generator(device).manual_seed(args.seed)
                    with autocast():
                        z = diffusion.ddim_sample(ema, cond, args.sample_steps, generator)
                        sample = restore_low_frequencies(decode(vae, z, latent), upscaled).clamp(-1, 1)
                    save_preview(upscaled, sample, hr, out / 'preview.png')
                    sample_text = (f' | {len(hr)} DDIM-{args.sample_steps} samples PSNR={psnr(sample, hr):.2f}dB '
                                   f'(bicubic {psnr(upscaled, hr):.2f}dB)')
            # Full frames as unet_inference.py upscales them (tile = crop, overlap 64 at crop
            # 256, one noise field per frame), split over the ranks
            frame_sums = dict(sr=0., bicubic=0., frames=0)
            for path in val_frames[rank::world]:
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert('RGB')
                width, height = image.size
                original = to_tensor(image).to(device)
                bicubic = to_tensor(upscaled_lr_crop(image, lr_size, (0, 0, width, height))).to(device)
                upscaled = F.pad(to_unit(bicubic), (0, (-width) % 8, 0, (-height) % 8), mode='replicate')
                generator = torch.Generator(device).manual_seed(args.seed)
                noise = torch.randn(1, channels, upscaled.shape[-2] // 8, upscaled.shape[-1] // 8,
                                    generator=generator, device=device)
                with autocast():
                    sr = tiled_upscale(vae, ema, diffusion, latent, upscaled, noise, args.crop,
                                       args.crop // 4 // 8 * 8, args.batch_size, args.sample_steps)
                sr = to_uint8(sr[..., :height, :width])
                frame_sums['sr'] += psnr(to_unit(sr), to_unit(original))
                frame_sums['bicubic'] += psnr(to_unit(bicubic), to_unit(original))
                frame_sums['frames'] += 1
        frame_sums = all_reduce_sums(frame_sums, device)
        frame_psnr = frame_sums['sr'] / frame_sums['frames']
        val_sum = all_reduce_sums(dict(loss=val_sum, count=count), device)
        val_loss = val_sum['loss'] / val_sum['count']
        train_sum = all_reduce_sums(dict(loss=train_sum, seen=seen), device)
        train_loss = train_sum['loss'] / max(train_sum['seen'], 1)  # 0 seen if resumed at an epoch's end
        log(f'epoch {epoch+1}/{args.epochs} train loss={train_loss:.5f} val loss (EMA)={val_loss:.5f}'
            + sample_text + f' | {round(frame_sums["frames"])} full frames PSNR={frame_psnr:.2f}dB '
            f'(bicubic {frame_sums["bicubic"] / frame_sums["frames"]:.2f}dB)', flush=True)
        improved = frame_psnr > best_psnr
        if improved:
            best_psnr = frame_psnr
        if rank == 0:
            save(out / 'latest.pt', epoch + 1, 0)
            if improved:
                save(out / 'best.pt', epoch + 1, 0)
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
