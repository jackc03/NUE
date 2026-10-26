"""Full-frame 4x super-resolution (960x540 -> 4K) with the latent-diffusion UNet, tile by tile.

Each frame is degraded exactly as in training (train_unet.upscaled_lr_crop): bicubic
downsampling to the checkpoint's LR size, then bicubic upsampling back to the frame size. The
upscaled frame is cut into overlapping tiles on the 8px latent grid, and every tile runs the
training pipeline on its own: VAE-encode the upscaled-LR tile, DDIM-sample its HR latent with
the EMA UNet, decode. Decoded tiles are cross-faded in pixel space. Latents are never blended:
the VAE's GroupNorm and squeeze-excite statistics span a tile, so neighbors encode a shared
region differently and decoding their blend is much worse (see vae_inference.py). All tiles
start from crops of one frame-wide noise field, so overlapping tiles agree where they overlap,
and every frame gets the same field, which steadies video. The stitched frame keeps only the
samples' high frequencies, over the upscaled frame's low ones: sampled tiles drift in
brightness and color, which had cost flat frames 6-15dB against bicubic, and 3.6dB on
average. --tile 256 matches the training crops. Metrics score the result and the bicubic
baseline against the original frame.

Example: python source/inference/unet_inference.py --input /storage/ice1/4/8/jcochran66/Inter4K_frames/val --limit 20 --amp
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Repo root, wherever it is cloned
from source.inference.vae_inference import EXTENSIONS, compare, to_uint8
from source.models.diffusion import GaussianDiffusion
from source.models.mbconv_vae import MBConvVAE
from source.models.unet import UNet
from source.training.train_unet import tiled_upscale, to_tensor, to_unit, upscaled_lr_crop

DEFAULT_CHECKPOINT = Path.home() / 'scratch/runs/latent_unet/best.pt'


def save_images(original, bicubic, sr, out, name, crop):
    # Full-res result, plus a native-res center crop of bicubic | result | original, since 4K
    # detail is lost when a frame is scaled down.
    original, bicubic, sr = (x[0].permute(1, 2, 0).cpu().numpy() for x in (original, bicubic, sr))
    Image.fromarray(sr).save(out / f'{name}_sr.png')
    height, width = sr.shape[:2]
    size_h, size_w = min(crop, height), min(crop, width)
    top, left = (height - size_h) // 2, (width - size_w) // 2
    panels = [a[top:top + size_h, left:left + size_w] for a in (bicubic, sr, original)]
    Image.fromarray(np.concatenate(panels, axis=1)).save(out / f'{name}_crop.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='An HR image, or a directory searched recursively')
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT), help='A checkpoint from train_unet.py')
    parser.add_argument('--vae', default=None,
                        help="The UNet's frozen VAE checkpoint, if it has moved (default: the path in --checkpoint)")
    parser.add_argument('--out-dir', default=None,
                        help='Default: inference_<checkpoint name>/ next to the checkpoint')
    parser.add_argument('--limit', type=int, default=20,
                        help='Frames to upscale, spread evenly over the sorted list (0 = all)')
    parser.add_argument('--save-images', type=int, default=4,
                        help='Save the 4K result and a comparison crop for this many frames')
    parser.add_argument('--crop', type=int, default=512, help='Side of the saved comparison crop')
    parser.add_argument('--steps', type=int, default=50, help='DDIM sampling steps')
    parser.add_argument('--seed', type=int, default=0, help='Seed of the noise field, the same for every frame')
    parser.add_argument('--lpips', action='store_true', help='Also report LPIPS (AlexNet); needs pip install lpips')
    parser.add_argument('--tile', type=int, default=256, help='Tile side in pixels, a multiple of 8')
    parser.add_argument('--overlap', type=int, default=64,
                        help='Pixels shared by neighboring tiles and cross-faded, a multiple of 8')
    parser.add_argument('--tile-batch', type=int, default=16, help='Tiles per forward pass')
    parser.add_argument('--amp', action='store_true', help='bfloat16 autocast, as in training (CUDA only)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.limit < 0 or args.save_images < 0 or args.crop < 1 or args.tile_batch < 1:
        parser.error('limit >= 0, save-images >= 0, crop >= 1, tile-batch >= 1 required')
    if args.tile % 8 or args.overlap < 0 or args.overlap % 8 or args.overlap >= args.tile:
        parser.error('tile and overlap must be multiples of 8, with 0 <= overlap < tile')
    if args.amp and not args.device.startswith('cuda'):
        parser.error('--amp is supported only with CUDA')
    source = Path(args.input)
    if source.is_dir():
        paths = sorted(p for p in source.rglob('*') if p.suffix.lower() in EXTENSIONS)
    elif source.is_file():
        paths, source = [source], source.parent
    else:
        parser.error(f'{source} does not exist')
    if not paths:
        parser.error(f'No supported images in {source}')
    if args.limit and len(paths) > args.limit:
        # Even spacing spans many clips instead of the first clip's first frames.
        paths = [paths[i * len(paths) // args.limit] for i in range(args.limit)]
    device = torch.device(args.device)
    lpips_model = None
    if args.lpips:
        try:
            import lpips
        except ImportError:
            parser.error('--lpips needs the lpips package: pip install lpips')
        lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device).eval()

    # mmap leaves the optimizer state and training weights on disk; only the EMA weights are read.
    checkpoint_path = Path(args.checkpoint).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
    config = checkpoint['config']
    model = UNet(**config['unet'])
    model.load_state_dict(checkpoint['ema'])
    model.to(device).eval()
    if args.tile < 8 * model.factor:
        parser.error(f'tile must be at least {8 * model.factor} for this UNet')
    diffusion = GaussianDiffusion(**config['diffusion'])
    latent = {key: value.to(device) for key, value in checkpoint['latent'].items()}
    lr_size = tuple(config['lr_size'])
    vae_path = Path(args.vae).expanduser() if args.vae else Path(checkpoint['vae']['path'])
    vae_checkpoint = torch.load(vae_path, map_location='cpu', weights_only=True)
    if vae_checkpoint['step'] != checkpoint['vae']['step']:
        parser.error(f'{vae_path} is from VAE step {vae_checkpoint["step"]}, but the UNet was '
                     f'trained on the latents of step {checkpoint["vae"]["step"]}')
    vae = MBConvVAE(**vae_checkpoint['config'])
    vae.load_state_dict(vae_checkpoint['model'])
    vae.to(device).eval()
    del vae_checkpoint
    out = Path(args.out_dir) if args.out_dir else checkpoint_path.parent / f'inference_{checkpoint_path.stem}'
    out.mkdir(parents=True, exist_ok=True)
    # Checkpoints from before full-frame validation were picked by val loss
    score = (f'full-frame val PSNR={checkpoint["best_psnr"]:.2f}dB' if 'best_psnr' in checkpoint
             else f'val loss={checkpoint["best_loss"]:.5f}')
    print(f'{checkpoint_path}: epoch={checkpoint["epoch"]} step={checkpoint["step"]} '
          f'{score} | {len(paths)} frames via {lr_size[0]}x{lr_size[1]}, '
          f'DDIM-{args.steps}, tile {args.tile}/{args.overlap} -> {out}', flush=True)

    rows = []
    for index, path in enumerate(paths):
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert('RGB')
        width, height = image.size
        if min(width, height) < 8 * model.factor:
            parser.error(f'{path} is smaller than {8 * model.factor}px')
        name = '_'.join(path.relative_to(source).with_suffix('').parts)
        original = to_tensor(image).to(device)
        bicubic = to_tensor(upscaled_lr_crop(image, lr_size, (0, 0, width, height))).to(device)
        upscaled = F.pad(to_unit(bicubic), (0, (-width) % 8, 0, (-height) % 8), mode='replicate')
        generator = torch.Generator(device).manual_seed(args.seed)
        noise = torch.randn(1, model.cond_channels, upscaled.shape[-2] // 8, upscaled.shape[-1] // 8,
                            generator=generator, device=device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            sr = tiled_upscale(vae, model, diffusion, latent, upscaled, noise,
                               args.tile, args.overlap, args.tile_batch, args.steps)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        sr = to_uint8(sr[..., :height, :width])
        row = dict(frame=str(path.relative_to(source)), height=height, width=width)
        for method, result in (('sr', sr), ('bicubic', bicubic)):
            row.update({f'{method}_{key}': value for key, value in compare(original, result).items()})
            if lpips_model is not None:
                with torch.inference_mode():
                    row[f'{method}_lpips'] = lpips_model(to_unit(original), to_unit(result)).item()
        row['seconds'] = seconds
        rows.append(row)
        print(f'[{index+1}/{len(paths)}] {row["frame"]} ' + ' '.join(
            f'{key}={value:.4f}' for key, value in row.items() if isinstance(value, float)), flush=True)
        if index < args.save_images:
            save_images(original, bicubic, sr, out, name, args.crop)
        del original, bicubic, upscaled, noise, sr

    metrics = [key for key, value in rows[0].items() if isinstance(value, float)]
    summary = {key: dict(mean=float(np.mean([row[key] for row in rows])),
                         std=float(np.std([row[key] for row in rows]))) for key in metrics}
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == 'cuda' else None
    with open(out / 'metrics.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(out / 'summary.json', 'w') as file:
        json.dump(dict(checkpoint=str(checkpoint_path), epoch=checkpoint['epoch'], step=checkpoint['step'],
                       input=str(Path(args.input)), frames=len(rows), device=str(device), amp=args.amp,
                       lr_size=lr_size, steps=args.steps, seed=args.seed, tile=args.tile,
                       overlap=args.overlap, peak_gpu_gb=peak_gb, metrics=summary),
                  file, indent=2)
    print(f'\nMean over {len(rows)} full frames' + (f', peak GPU memory {peak_gb:.1f}GB' if peak_gb else ''))
    for key, value in summary.items():
        print(f'  {key:<16} {value["mean"]:.4f} ± {value["std"]:.4f}')
    print(f'Wrote {out / "metrics.csv"} and {out / "summary.json"}')


if __name__ == '__main__':
    main()
