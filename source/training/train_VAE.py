"""Pretrain the MBConv VAE on HR frames/images, with validation and resume.

Example: python train_VAE.py --train-dir data/train --val-dir data/val --out-dir runs/mbconv
"""

import sys
sys.path.insert(0, "/home/hice1/yyu496/AI_Enigneer")


import os
import argparse
import math
import random
from pathlib import Path
import json
from itertools import chain, islice
from tqdm.auto import tqdm
import hashlib
import multiprocessing as mp
from datetime import timedelta

import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, BatchSampler, Sampler
from torch.utils.data.distributed import DistributedSampler

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


from NUE.source.models.mbconv_vae import MBConvVAE, vae_loss

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


class ImageCrops(Dataset):
    def __init__(self, root, crop, training, seed=42, rank=0):
        self.paths = sorted(p for p in Path(root).rglob('*') if p.suffix.lower() in EXTENSIONS)

        if not self.paths:
            raise ValueError(f'No supported images in {root}')
        
        self.crop = crop
        self.training = training
        self.seed = seed
        self.rank = rank
        self.shared_epoch = mp.Value("q", 0)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')

        width, height = image.size
        if width < self.crop or height < self.crop:
            raise ValueError(f'{path} is smaller than the {self.crop}px crop')

        epoch = self.shared_epoch.value
        seed_bytes = hashlib.blake2b(
            f"{self.seed}:{epoch}:{self.rank}:{index}".encode(),
            digest_size=8
        ).digest()

        sample_seed = int.from_bytes(seed_bytes, "little")
        rng = random.Random(sample_seed)
        
        if self.training:
            left = rng.randint(0, width - self.crop)
            top = rng.randint(0, height - self.crop)
        else:
            left = (width - self.crop) // 2
            top = (height - self.crop) // 2

        image = image.crop((left, top, left + self.crop, top + self.crop))

        if self.training and rng.random() < .5:
            image = ImageOps.mirror(image)

        array = np.asarray(image, dtype=np.float32).copy()
        return torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1

    def set_epoch(self, epoch):
        self.shared_epoch.value = epoch


def create_image_manifest(root, manifest_path):
    root = Path(root)
    manifest_path = Path(manifest_path)

    if manifest_path.exists():
        return

    temporary = manifest_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for directory, subdirs, files in os.walk(root):
            subdirs.sort()

            for filename in sorted(files):
                if Path(filename).suffix.lower() in EXTENSIONS:
                    f.write(str(Path(directory) / filename) + "\n")

    temporary.replace(manifest_path)



class ImageNetDataset(Dataset):
    def __init__(self, root, crop=256, training=True, manifest_path=None, seed=42, rank=0):
        self.root = Path(root)
        self.training = training

        self.crop = crop
        self.seed = seed
        self.rank = rank
        self.shared_epoch = mp.Value("q", 0)

        if manifest_path is not None:
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.paths = [
                    Path(line.rstrip("\n"))
                    for line in f
                ]
        else:
            self.paths = sorted(
                p for p in self.root.rglob("*")
                if p.suffix.lower() in EXTENSIONS
            )

        if not self.paths:
            raise ValueError(
                f"No images found in {self.root}"
            )

        if training:
            self.transform = T.Compose([
                T.Resize(crop + 32, interpolation=InterpolationMode.BILINEAR),
                T.RandomCrop(crop),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5)
                )
            ])
        else:
            self.transform = T.Compose([
                T.Resize(crop + 32, interpolation=InterpolationMode.BILINEAR),
                T.CenterCrop(crop),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5)
                ),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        if not self.training:
            return self.transform(image)

        epoch = self.shared_epoch.value
        seed_bytes = hashlib.blake2b(
            f"{self.seed}:{epoch}:{self.rank}:{index}".encode(),
            digest_size=8
        ).digest()

        sample_seed = int.from_bytes(seed_bytes, "little")
        rng = random.Random(sample_seed)

        image = TF.resize(
            image,
            self.crop + 32,
            interpolation=InterpolationMode.BILINEAR,
        )

        width, height = image.size
        top = rng.randint(0, height - self.crop)
        left = rng.randint(0, width - self.crop)
        image = TF.crop(
            image,
            top,
            left,
            self.crop,
            self.crop,
        )

        if rng.random() < 0.5:
            image = TF.hflip(image)
        tensor = TF.to_tensor(image)
        return TF.normalize(
            tensor,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

    def set_epoch(self, epoch):
        self.shared_epoch.value = epoch



class SkipBatchSampler:
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.batch_sampler = BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=False
        )
        self.skip_batches = skip_batches

    def __iter__(self):
        yield from islice(
            iter(self.batch_sampler),
            self.skip_batches,
            None,
        )

    def __len__(self):
        return max(0, len(self.batch_sampler) - self.skip_batches)



class DistributedEvalSampler(Sampler):
    """
    Distributed validation samples across ranks without
    duplicatting or dropping any samples
    """
    def __init__(self, dataset, rank=0, world_size=1):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(
            range(self.rank, len(self.dataset), self.world_size)
        )

    def __len__(self):
        return len(
            range(self.rank, len(self.dataset), self.world_size)
        )



def save_preview(target, reconstruction, path):
    # Side-by-side target and deterministic reconstruction, first val example.
    panels = []
    for tensor in (target[0], reconstruction[0]):
        array = ((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5)
        panels.append(array.permute(1, 2, 0).numpy().round().astype(np.uint8))
    Image.fromarray(np.concatenate(panels, axis=1)).save(path)



def save_checkpoint(path, model, optimizer, epoch, step, best_psnr, config, args, batch_in_epoch=0, rank_states=None):
    payload = {
        "model" : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "epoch" : epoch,
        "batch_in_epoch" : batch_in_epoch,
        "step" : step,
        "best_psnr" : best_psnr,
        "config" : config,
        "args" : vars(args),
        "rank_states" : rank_states,
        "world_size" : len(rank_states) if rank_states else 1
    }

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)



def compute_model_size(model):
    bytes = 0
    seen = set()

    for tensor in chain(model.parameters(), model.buffers()):
        if tensor.is_meta:
            continue

        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())

        if key not in seen:
            seen.add(key)
            bytes += storage.nbytes()
    return bytes


def compute_model_parameters(model):
    total = 0

    for p in model.parameters():
        total += p.numel()

    return total


def capture_rank_state(train_sum, seen, device):
    return {
        "python_rng" : random.getstate(),
        "numpy_rng" : np.random.get_state(),
        "torch_cpu_rng" : torch.get_rng_state(),
        "torch_cuda_rng" : (
            torch.cuda.get_rng_state(device)
            if device.type == "cuda"
            else None
        ),
        "train_sum" : dict(train_sum),
        "seen" : int(seen)
    }

def restore_rank_rng(state, device):
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_cpu_rng"])

    if device.type == "cuda":
        torch.cuda.set_rng_state(
            state["torch_cuda_rng"],
            device=device
        )

def collect_rank_states(local_state, distributed, world_size):
    if distributed:
        states = [None] * world_size
        dist.all_gather_object(states, local_state)
        return states
    return [local_state]



def create_validation_metrics(device, distributed):
    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=1.0,
        sync_on_compute=distributed
    ).to(device)

    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        normalize=False,
        sync_on_compute=distributed
    ).to(device)
    lpips_metric.eval()

    return ssim_metric, lpips_metric




def validate_vae(
        model,
        valid_loader,
        ssim_metric,
        lpips_metroc,
        args,
        device,
        is_main,
        distributed,
        out,
        description = "Validation"
):
    model.eval()
    model = model.module if isinstance(model, DDP) else model

    val_stats = torch.zeros(6, dtype=torch.float64, device=device)

    valid_bar = tqdm(
        valid_loader,
        desc=description,
        disable=not is_main,
        dynamic_ncols=True,
        leave=False
    )
    with torch.inference_mode():
        for batch_idx, x, in enumerate(valid_bar):
            x = x.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
                reconstruction, mu, logvar = model(x, sample=False)
                loss, metrics = vae_loss(reconstruction, x, mu, logvar, args.beta)

            batch_size = x.size(0)
            for i, key in enumerate(("loss", "l1", "kl")):
                val_stats[i] += metrics[key].detach().double() * batch_size
            val_stats[3] += batch_size

            pred = reconstruction.detach().float().clamp(-1, 1)
            target = x.float().clamp(-1, 1)

            squared_error = ((pred - target) / 2.0).square()

            val_stats[4] += squared_error.sum(dtype=torch.float64)
            val_stats[5] += target.numel()

            pred_01 = (pred + 1.0) * 0.5
            target_01 = (target + 1.0) * 0.5
            ssim_metric.update(pred_01, target_01)
            lpips_metroc.update(pred, target)

            if is_main and batch_idx == 0:
                save_preview(x, reconstruction, out / "preview.png")
            if is_main and batch_idx % 10 == 0:
                valid_bar.set_postfix(
                    loss=f"{metrics['loss'].item():.4f}",
                    l1=f"{metrics['l1'].item():.4f}",
                    kl=f"{metrics['kl'].item():.4f}"
                )

    if distributed:
        dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)

    ssim = ssim_metric.compute().item()
    lpips = lpips_metroc.compute().item()

    ssim_metric.reset()
    lpips_metroc.reset()

    num_images = val_stats[3].item()
    squared_error = val_stats[4].item()
    pixel_count = val_stats[5].item()

    mse = squared_error / pixel_count
    psnr = -10.0 * math.log10(max(mse, 1e-12))

    results = {
        "loss": val_stats[0].item() / num_images,
        "l1": val_stats[1].item() / num_images,
        "kl": val_stats[2].item() / num_images,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--val-dir', required=True)
    parser.add_argument('--out-dir', default='runs/mbconv_vae')
    parser.add_argument('--resume', default=None, help='A checkpoint from this script')
    parser.add_argument('--save_every_steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--compile', action='store_true', help='compile the model')
    parser.add_argument('--train', action='store_true', help='training mode')
    parser.add_argument('--imagenet', action='store_true', help='use imagenet as training data')
    parser.add_argument('--crop', type=int, default=256)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--beta', type=float, default=1e-4)
    parser.add_argument('--kl-warmup-steps', type=int, default=10000)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--max_channels', type=int, default=512)
    parser.add_argument('--growth', type=int, default=16)
    parser.add_argument('--blocks-per-stage', type=int, default=2)
    parser.add_argument('--latent-channels', type=int, default=4)
    parser.add_argument('--amp', action='store_true', help='CUDA mixed precision')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    if args.crop < 8 or args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error('crop >= 8, epochs >= 1, batch-size >= 1, workers >= 0 required')
    if args.beta < 0 or args.lr <= 0 or args.kl_warmup_steps < 0:
        parser.error('beta >= 0, lr > 0, kl-warmup-steps >= 0 required')
    if args.amp and not args.device.startswith('cuda'):
        parser.error('--amp is supported only with CUDA')
    
    
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    
    if distributed:
        if not torch.cuda.is_available():
            parser.error("CUDA is required for the DDP configuration")
        
        if not args.device.startswith("cuda"):
            parser.error("Use a CUDA device with torchrun")
            
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=device, timeout=timedelta(hours=1))
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device(args.device)
        
    is_main = rank == 0
    
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.imagenet:
        train_manifest = out / "imagenet_train_paths.txt"
        valid_manifest = out / "imagenet_valid_paths.txt"

        if is_main:
            print("Preparing ImageNet file manifest...", flush=True)

            create_image_manifest(args.train_dir, train_manifest)
            create_image_manifest(args.val_dir, valid_manifest)

            print("ImageNet manifests ready.", flush=True)

        if distributed:
            dist.barrier(device_ids=[local_rank])

        train_set = ImageNetDataset(args.train_dir, args.crop, training=True, manifest_path=train_manifest, seed=args.seed, rank=rank,)
        valid_set = ImageNetDataset(args.val_dir, args.crop, training=False, manifest_path=valid_manifest, seed=args.seed, rank=rank,)
    else:
        train_set = ImageCrops(args.train_dir, args.crop, training=True)
        valid_set = ImageCrops(args.val_dir, args.crop, training=False)
    
    if set(train_set.paths) & set(valid_set.paths):
        raise ValueError("Training and validation paths overlapped")
        
    if is_main:
        print(
            f"train images = {len(train_set)} "
            f"valid images = {len(valid_set)} "
            f"world size = {world_size}",
            flush=True
        )
        
    train_sampler = (
        DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed
        )
        if distributed else None
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed + rank)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
        generator=loader_generator
    )

    valid_sampler = DistributedEvalSampler(
        valid_set, rank=rank, world_size=world_size
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed + 1000 + rank)
    )
        
    
    
    config = dict(
        base_channels=args.base_channels,
        blocks_per_stage=args.blocks_per_stage,
        latent_channels=args.latent_channels
    )
    
    
    start_epoch = 0
    start_batch = 0
    step = 0
    best_psnr = -math.inf

    resume_rank_state = None
    
    
    base_model = MBConvVAE(**config).to(device)
    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=args.lr
    )

    if is_main:
        total_model_bytes = compute_model_size(base_model)
        total_parameters = compute_model_parameters(base_model)
        print(f"Model storage size: {total_model_bytes / (1024 ** 3)} GB")
        print(f"Total parameters: {total_parameters:,} ({total_parameters / 1e6:.2f}M)")
        


    if is_main:
        ssim_metric, lpips_metric = create_validation_metrics(device, distributed)
    if distributed:
        dist.barrier(device_ids=[local_rank])
    if not is_main:
        ssim_metric, lpips_metric = create_validation_metrics(device, distributed)
    
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        
        if config != checkpoint["config"]:
            parser.error(
                "Architecture arguments must match "
                "the resumed checkpoint config"
            )

        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint.get("batch_in_epoch", 0))
        step = int(checkpoint["step"])
        best_psnr = checkpoint["best_psnr"]

        if checkpoint["world_size"] != world_size:
            raise ValueError("Exact mid-epoch resume requires the same world size")
        if checkpoint["args"]["batch_size"] != args.batch_size:
            raise ValueError("Exact mid-epoch resume requres the same per-GPU batch size")
        if checkpoint["args"]["seed"] != args.seed:
            raise ValueError("Exact resume resuires the original random seed")

        rank_states = checkpoint.get("rank_states")
        if not rank_states:
            raise ValueError("Checkpoint does not contain per-rank resume state")
        resume_rank_state = rank_states[rank]
        
        base_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        
        if is_main:
            print(
                f"Resumed epoch={start_epoch + 1}, "
                f"batch={start_batch}, "
                f"global_step={step}",
                flush=True,
            )
    
    if args.compile:
        base_model.compile(fullgraph=True, mode="reduce-overhead")
        
    if distributed:
        model = DDP(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank
        )
    else:
        model = base_model

    if args.train:
        for epoch in range(start_epoch, args.epochs):
            if isinstance(train_set, (ImageNetDataset, ImageCrops)):
                train_set.set_epoch(epoch)
        
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            skip_batches = start_batch if epoch == start_epoch else 0
            if epoch == start_epoch and skip_batches > 0:
                train_sum = dict(resume_rank_state["train_sum"])
                seen = int(resume_rank_state["seen"])
            else:
                train_sum = dict(loss=0., l1=0., kl=0.)
                seen = 0

            skip_batches = start_batch if epoch == start_epoch else 0
            if skip_batches > 0:
                if train_sampler is None:
                    raise RuntimeError("This mid-epoch resume path requires a deterministic sampler")
                epoch_loader = DataLoader(
                    train_set,
                    batch_sampler=SkipBatchSampler(
                        train_sampler,
                        args.batch_size,
                        skip_batches=skip_batches
                    ),
                    num_workers=args.workers,
                    persistent_workers=args.workers > 0,
                    pin_memory=device.type == "cuda",
                )
            else:
                epoch_loader = train_loader
            loader_iter = iter(epoch_loader)

            if epoch == start_epoch and resume_rank_state is not None:
                restore_rank_rng(resume_rank_state, device)
                resume_rank_state = None
                
            model.train()

            train_bar = tqdm(
                loader_iter,
                desc=f"Train {epoch + 1}/{args.epochs}",
                initial=skip_batches,
                total=len(train_loader),
                disable=not is_main,
                dynamic_ncols=True,
                leave=False,
            )

            for batch_idx, x in enumerate(train_bar, start=skip_batches):
                x = x.to(device, non_blocking=True)
                
                beta = args.beta * min(1., (step + 1) / max(args.kl_warmup_steps, 1))
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
                    reconstruction, mu, logvar = model(x)
                    loss, metrics = vae_loss(reconstruction, x, mu, logvar, beta)
                    
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finit training loss"
                    )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.,
                    error_if_nonfinite=True
                )
                optimizer.step()
                
                
                batch_metrics = {
                    key : metrics[key].detach().item()
                    for key in train_sum
                }

                for key, value in batch_metrics.items():
                    train_sum[key] += value * x.size(0)

                seen += x.size(0)
                step += 1

                if is_main and step % 10 == 0:
                    train_bar.set_postfix(
                        loss=f"{batch_metrics['loss']:.4f}",
                        l1=f"{batch_metrics['l1']:.4f}",
                        kl=f"{batch_metrics['kl']:.4f}",
                        beta=f"{beta:.2e}",
                    )

                if step % args.save_every_steps == 0:
                    local_state = capture_rank_state(train_sum, seen, device)
                    rank_states = collect_rank_states(local_state, distributed, world_size)

                    if is_main:
                        save_checkpoint(
                        out / "step_level_latest.pt",
                        base_model,
                        optimizer,
                        epoch,
                        step,
                        best_psnr,
                        config,
                        args,
                        batch_in_epoch=batch_idx + 1,
                        rank_states=rank_states,
                    )

                    if distributed:
                        dist.barrier(device_ids=[local_rank])
            
            
            train_stats = torch.tensor(
                [
                    train_sum["loss"],
                    train_sum["l1"],
                    train_sum["kl"],
                    seen
                ],
                dtype=torch.float64,
                device=device
            )
            
            if distributed:
                dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
                
            
            
            # validation
            results = validate_vae(
                model=model,
                valid_loader=valid_loader,
                ssim_metric=ssim_metric,
                lpips_metroc=lpips_metric,
                args=args,
                device=device,
                is_main=is_main,
                distributed=distributed,
                out=out,
                description=f"Valid {epoch + 1}/{args.epochs}"
            )


            psnr = results["psnr"]
            ssim = results["ssim"]
            lpips = results["lpips"]

            if is_main:
                loss_sum, l1_sum, kl_sum, train_count = train_stats.tolist()

                train_text = (
                    f"loss={loss_sum / train_count:.5f} "
                    f"l1={l1_sum / train_count:.5f} "
                    f"kl={kl_sum / train_count:.5f}"
                )

                valid_text = (
                    f"loss={results['loss']:.5f} "
                    f"l1={results['l1']:.5f} "
                    f"kl={results['kl']:.5f}"
                )

                tqdm.write(
                    f"Epoch {epoch + 1}/{args.epochs} "
                    f"train {train_text} "
                    f"valid {valid_text} "
                    f"PSNR={psnr:.2f}dB "
                    f"SSIM={ssim:.4f} "
                    f"LPIPS={lpips:.4f}"
                )

            local_state = capture_rank_state(train_sum, seen, device)
            rank_states = collect_rank_states(local_state, distributed, world_size)


            if is_main:    
                improved = psnr > best_psnr
                if improved:
                    best_psnr = psnr
                    
                save_checkpoint(
                    out / "latest.pt",
                    base_model,
                    optimizer,
                    epoch + 1,
                    step,
                    best_psnr,
                    config,
                    args,
                    batch_in_epoch=0,
                    rank_states=rank_states
                )
                
                if improved:
                    save_checkpoint(
                        out / "best.pt",
                        base_model,
                        optimizer,
                        epoch + 1,
                        step,
                        best_psnr,
                        config,
                        args,
                        batch_in_epoch=0,
                        rank_states=rank_states
                    )
                    
                    with open(out / "result.json", "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "Best-PSNR": float(best_psnr),
                                "SSIM": float(ssim),
                                "LPIPS": float(lpips),
                                "epoch": epoch + 1,
                            },
                            f,
                            indent=4,
                        )
        
            if distributed:
                dist.barrier(device_ids=[local_rank])
    else:
        if not args.resume:
            raise ValueError("The solely validation run requirement resume")

        results = validate_vae(
            model=model,
            valid_loader=valid_loader,
            ssim_metric=ssim_metric,
            lpips_metroc=lpips_metric,
            args=args,
            device=device,
            is_main=is_main,
            distributed=distributed,
            out=out,
            description=f"Validation checkpoint Epoch {start_epoch} step {step}"
        )

        if is_main:
            tqdm.write(
                f"Validation results: "
                f"loss={results['loss']:.5f} "
                f"l1={results['l1']:.5f} "
                f"kl={results['kl']:.5f} "
                f"PSNR={results['psnr']:.2f}dB "
                f"SSIM={results['ssim']:.4f} "
                f"LPIPS={results['lpips']:.4f}"
            )

        if distributed:
            dist.barrier(device_ids=[local_rank])
            
    if distributed:
        dist.destroy_process_group()
    
                    
        
        
        
        
        
        
        
        
        
    
    




if __name__ == "__main__":
    main()