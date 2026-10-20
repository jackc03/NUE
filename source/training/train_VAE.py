"""Pretrain the MBConv VAE on HR frames/images, with validation and resume.

Example: python source/training/train_VAE.py --train-dir data/train --val-dir data/val --out-dir runs/mbconv
Multi-GPU: torchrun --standalone --nproc_per_node=4 source/training/train_VAE.py ...
"""

import argparse
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
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Repo root, wherever it is cloned
from source.models.mbconv_vae import MBConvVAE, vae_loss

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


class ImageCrops(Dataset):
    def __init__(self, root, crop, training):
        self.paths = sorted(p for p in Path(root).rglob('*') if p.suffix.lower() in EXTENSIONS)
        if not self.paths:
            raise ValueError(f'No supported images in {root}')
        self.crop = crop
        self.training = training

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        width, height = image.size
        if width < self.crop or height < self.crop:
            raise ValueError(f'{path} is smaller than the {self.crop}px crop')
        if self.training:
            left = random.randint(0, width - self.crop)
            top = random.randint(0, height - self.crop)
        else:
            left = (width - self.crop) // 2
            top = (height - self.crop) // 2
        image = image.crop((left, top, left + self.crop, top + self.crop))
        if self.training and random.random() < .5:
            image = ImageOps.mirror(image)
        array = np.asarray(image, dtype=np.float32).copy()
        return torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1


class EpochSampler(Sampler):
    # Every rank draws the same per-epoch order and takes every world-th index. `skip`
    # drops samples already trained this epoch, so a resume continues mid-epoch, even
    # with a different GPU count.
    def __init__(self, length, rank, world, shuffle, seed):
        self.length, self.rank, self.world = length, rank, world
        self.shuffle, self.seed = shuffle, seed
        self.epoch, self.skip = 0, 0

    def set_epoch(self, epoch, skip=0):
        self.epoch, self.skip = epoch, skip

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(self.length, generator=generator)[self.skip:]
            order = order[:len(order) - len(order) % self.world]  # Equal shares keep DDP ranks in step
        else:
            order = torch.arange(self.length)[self.skip:]
        return iter(order[self.rank::self.world].tolist())

    def __len__(self):
        remaining = max(self.length - self.skip, 0)
        if self.shuffle:
            return remaining // self.world
        return len(range(self.rank, remaining, self.world))



def save_preview(target, reconstruction, path):
    # Side-by-side target and deterministic reconstruction, first val example.
    panels = []
    for tensor in (target[0], reconstruction[0]):
        array = ((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5)
        panels.append(array.permute(1, 2, 0).numpy().round().astype(np.uint8))
    Image.fromarray(np.concatenate(panels, axis=1)).save(path)



def save_checkpoint(path, model, optimizer, epoch, epoch_samples, step, best_psnr, config, args):
    # epoch = epochs fully trained; epoch_samples = images already trained in the next one.
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                   epoch=epoch, epoch_samples=epoch_samples, step=step,
                   best_psnr=best_psnr, config=config, args=vars(args))
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def all_reduce_sums(sums, device):
    # Sum each rank's running totals so logged metrics cover every GPU's samples.
    if not dist.is_initialized():
        return sums
    totals = torch.tensor(list(sums.values()), dtype=torch.float64, device=device)
    dist.all_reduce(totals)
    return dict(zip(sums, totals.tolist()))




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--val-dir', required=True)
    parser.add_argument('--out-dir', default='runs/mbconv_vae')
    parser.add_argument('--resume', default=None, help='A checkpoint from this script')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4, help='Per GPU')
    parser.add_argument('--crop', type=int, default=256)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta', type=float, default=1e-4)
    parser.add_argument('--kl-warmup-steps', type=int, default=10000)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--growth', type=int, default=16)
    parser.add_argument('--blocks-per-stage', type=int, default=2)
    parser.add_argument('--latent-channels', type=int, default=16)
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
    if args.crop < 8 or args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error('crop >= 8, epochs >= 1, batch-size >= 1, workers >= 0 required')
    if args.checkpoint_every < 0:
        parser.error('checkpoint-every >= 0 required')
    if args.beta < 0 or args.lr <= 0 or args.kl_warmup_steps < 0:
        parser.error('beta >= 0, lr > 0, kl-warmup-steps >= 0 required')
    if args.amp and not args.device.startswith('cuda'):
        parser.error('--amp is supported only with CUDA')
    # Per-rank seeds vary the crops; DDP copies rank 0's initial weights to every rank.
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_set = ImageCrops(args.train_dir, args.crop, training=True)
    val_set = ImageCrops(args.val_dir, args.crop, training=False)
    if set(train_set.paths) & set(val_set.paths):
        raise ValueError('Training and validation paths overlap')
    log(f'train images={len(train_set)} val images={len(val_set)} gpus={world}')
    train_sampler = EpochSampler(len(train_set), rank, world, shuffle=True, seed=args.seed)
    val_sampler = EpochSampler(len(val_set), rank, world, shuffle=False, seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                              drop_last=True, num_workers=args.workers,
                              pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=args.workers, pin_memory=device.type == 'cuda')
    config = dict(base_channels=args.base_channels,
                  blocks_per_stage=args.blocks_per_stage,
                  latent_channels=args.latent_channels)
    start_epoch, epoch_samples, step, best_psnr = 0, 0, 0, -math.inf
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if config != checkpoint['config']:
            parser.error('Architecture arguments must match the resumed checkpoint config')
    model = MBConvVAE(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']
        epoch_samples = checkpoint.get('epoch_samples', 0)
        step = checkpoint['step']
        best_psnr = checkpoint['best_psnr']
        log(f'Resumed after {start_epoch} finished epochs, {epoch_samples} images into '
            f'epoch {start_epoch+1}, step={step}')
    train_model = DistributedDataParallel(model, device_ids=[device.index]) if distributed else model
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch, skip=epoch_samples)
        model.train()
        train_sum = dict(loss=0., l1=0., kl=0.)
        seen = 0
        last_time, last_step = time.time(), step
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            beta = args.beta * min(1., (step + 1) / max(args.kl_warmup_steps, 1))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=args.amp, dtype=torch.bfloat16):
                reconstruction, mu, logvar = train_model(x)
                loss, metrics = vae_loss(reconstruction, x, mu, logvar, beta)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            for key in train_sum:
                train_sum[key] += metrics[key].item() * x.size(0)
            seen += x.size(0)
            step += 1
            epoch_samples += x.size(0) * world
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                if rank == 0:
                    save_checkpoint(out / 'latest.pt', model, optimizer, epoch, epoch_samples,
                                    step, best_psnr, config, args)
                now = time.time()
                log(f'epoch {epoch+1} {epoch_samples}/{len(train_set)} images step={step} '
                    f'{(now - last_time) / (step - last_step):.3f}s/step', flush=True)
                last_time, last_step = now, step
        epoch_samples = 0
        model.eval()
        val_sum = dict(loss=0., l1=0., kl=0., mse=0.)
        count = 0
        with torch.inference_mode():
            for batch_number, x in enumerate(val_loader):
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type='cuda', enabled=args.amp):
                    reconstruction, mu, logvar = model(x, sample=False)
                    loss, metrics = vae_loss(reconstruction, x, mu, logvar, args.beta)
                for key in ('loss', 'l1', 'kl'):
                    val_sum[key] += metrics[key].item() * x.size(0)
                # PSNR in [0,1] RGB, computed over all pixels and images.
                mse = ((reconstruction.float().clamp(-1, 1) - x.float()) / 2).square()
                val_sum['mse'] += mse.sum().item()
                count += x.numel()
                if batch_number == 0 and rank == 0:
                    save_preview(x, reconstruction, out / 'preview.png')
        val_sum = all_reduce_sums(dict(val_sum, count=count), device)
        count = val_sum.pop('count')
        train_sum = all_reduce_sums(dict(train_sum, seen=seen), device)
        seen = max(train_sum.pop('seen'), 1)  # 0 if resumed exactly at an epoch's end
        psnr = -10 * math.log10(max(val_sum['mse'] / count, 1e-12))
        train_text = ' '.join(f'{key}={value/seen:.5f}' for key, value in train_sum.items())
        val_text = ' '.join(f'{key}={val_sum[key]/len(val_set):.5f}' for key in ('loss','l1','kl'))
        log(f'epoch {epoch+1}/{args.epochs} train {train_text} val {val_text} PSNR={psnr:.2f}dB', flush=True)
        improved = psnr > best_psnr
        if improved:
            best_psnr = psnr
        if rank == 0:
            save_checkpoint(out / 'latest.pt', model, optimizer, epoch+1, 0,
                            step, best_psnr, config, args)
            if improved:
                save_checkpoint(out / 'best.pt', model, optimizer, epoch+1, 0,
                                step, best_psnr, config, args)
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
