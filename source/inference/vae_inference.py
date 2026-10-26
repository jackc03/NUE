"""Full-frame MBConv VAE reconstruction quality: PSNR, SSIM, MS-SSIM, LPIPS and more.

Each frame is split into overlapping tiles; every tile is encoded through the latent mean
and decoded on its own, and the reconstructions are cross-faded into one full-frame image,
so peak memory depends on the tile size, not the frame size. Tile latents are only blended
for the kl and latent_std statistics: neighboring tiles encode a shared region differently,
and decoding their blend instead cost ~10dB PSNR. GroupNorm and squeeze-excite statistics
span a tile, so tiles near the 256px training crops match training most closely.
--tile 0 runs whole frames untiled.

Example: python source/inference/vae_inference.py --input /storage/ice1/4/8/jcochran66/Inter4K_frames/val --limit 20
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Repo root, wherever it is cloned
from source.models.mbconv_vae import MBConvVAE, vae_loss

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
DEFAULT_CHECKPOINT = Path.home() / 'scratch/runs/mbconv_vae/best.pt'


def load_image(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0)  # uint8 [1, 3, H, W]


def to_uint8(tensor):
    # Model output in [-1, 1] -> the 8-bit image a saved PNG would hold; metrics use this.
    return ((tensor.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def tile_starts(size, tile, overlap):
    # Offsets of tiles covering [0, size); the last tile sits flush with the far edge.
    if size <= tile:
        return [0]
    return list(range(0, size - tile, tile - overlap)) + [size - tile]


def ramp(length, fade, rise, fall):
    # Blend weights along one tile axis: linear cross-fades over `fade` samples on the
    # sides shared with a neighbor, never zero so every output sample has weight.
    weight = torch.ones(length)
    steps = torch.arange(1, fade + 1) / (fade + 1)
    if fade and rise:
        weight[:fade] = torch.minimum(weight[:fade], steps)
    if fade and fall:
        weight[-fade:] = torch.minimum(weight[-fade:], steps.flip(0))
    return weight


def tiled(function, x, tile, overlap, scales, batch):
    # Apply `function` to overlapping tiles of x [1, C, H, W], `batch` tiles at a time, and
    # blend each of its outputs in float32. For a [B, C, h, w] input, `function` returns a
    # tuple whose i-th tensor is [B, C_i, h*scales[i], w*scales[i]].
    height, width = x.shape[-2:]
    tile_h, tile_w = min(tile, height), min(tile, width)
    boxes = [(top, left) for top in tile_starts(height, tile_h, overlap)
             for left in tile_starts(width, tile_w, overlap)]
    outs = weights = None
    for first in range(0, len(boxes), batch):
        chunk = boxes[first:first + batch]
        ys = function(torch.cat([x[..., top:top + tile_h, left:left + tile_w] for top, left in chunk]))
        if outs is None:
            outs = [torch.zeros(1, y.shape[1], round(height * scale), round(width * scale), device=x.device)
                    for y, scale in zip(ys, scales)]
            weights = [torch.zeros(1, 1, *out.shape[-2:], device=x.device) for out in outs]
        for y, scale, out, total in zip(ys, scales, outs, weights):
            s = lambda v: round(v * scale)
            for (top, left), y_tile in zip(chunk, y):
                weight = (ramp(s(tile_h), s(overlap), top > 0, top + tile_h < height)[:, None]
                          * ramp(s(tile_w), s(overlap), left > 0, left + tile_w < width)).to(x.device)
                region = (..., slice(s(top), s(top + tile_h)), slice(s(left), s(left + tile_w)))
                out[region] += weight * y_tile.float()
                total[region] += weight
    return [out / total for out, total in zip(outs, weights)]


def tiled_reconstruct(model, x, tile, overlap, batch):
    # Encode and decode each tile on its own, then blend the reconstructions. The blended
    # latent moments are for statistics only; decoding them is much worse. Tiles are
    # aligned to the 8x latent grid; tile=0 encodes and decodes the whole frame at once.
    height, width = x.shape[-2:]
    x = F.pad(x, (0, (-width) % 8, 0, (-height) % 8), mode='replicate')
    tile = tile or max(x.shape[-2:])

    def reconstruct(tiles):
        mu, logvar = model.encode(tiles)
        return model.decode(mu), torch.cat((mu, logvar), 1)

    decoded, moments = tiled(reconstruct, x, tile, overlap, (1, 1 / 8), batch)
    mu, logvar = moments.chunk(2, dim=1)
    return decoded[..., :height, :width], mu, logvar


def luma(x):
    # ITU-R BT.601 Y (MATLAB rgb2ycbcr), the usual convention in super-resolution papers.
    r, g, b = x.unbind(1)
    return ((16 + 65.481 * r + 128.553 * g + 24.966 * b) / 255).unsqueeze(1)


def psnr(x, y):
    return -10 * math.log10(max((x - y).square().mean().item(), 1e-12))


def ssim_terms(x, y, size=11, sigma=1.5):
    # Wang et al. 2004: 11x11 Gaussian window, 'valid' filtering, data range 1. Returns the
    # mean SSIM and contrast-structure terms per image and channel, [B, C].
    channels = x.shape[1]
    coords = torch.arange(size, dtype=x.dtype, device=x.device) - size // 2
    window = torch.exp(-coords.square() / (2 * sigma ** 2))
    window = (window / window.sum()).repeat(channels, 1, 1, 1)
    blur = lambda t: F.conv2d(F.conv2d(t, window.view(channels, 1, -1, 1), groups=channels),
                              window.view(channels, 1, 1, -1), groups=channels)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x.square()
    var_y = blur(y * y) - mu_y.square()
    cov = blur(x * y) - mu_x * mu_y
    cs = (2 * cov + c2) / (var_x + var_y + c2)
    ssim = (2 * mu_x * mu_y + c1) / (mu_x.square() + mu_y.square() + c1) * cs
    return ssim.mean((-2, -1)), cs.mean((-2, -1))


def ms_ssim(x, y):
    # Wang et al. 2003, five scales; the coarsest must still fit the 11px window.
    if min(x.shape[-2:]) < 11 * 2 ** (len(MS_SSIM_WEIGHTS) - 1):
        return math.nan
    terms = []
    for weight in MS_SSIM_WEIGHTS[:-1]:
        terms.append(ssim_terms(x, y)[1].relu() ** weight)
        x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
    terms.append(ssim_terms(x, y)[0].relu() ** MS_SSIM_WEIGHTS[-1])
    return torch.stack(terms).prod(0).mean().item()


def compare(original, reconstruction):
    # uint8 [1, 3, H, W] pair -> fidelity metrics on [0, 1] intensities, in float64.
    x, y = original.double() / 255, reconstruction.double() / 255
    x_luma, y_luma = luma(x), luma(y)
    return dict(psnr=psnr(x, y), psnr_y=psnr(x_luma, y_luma),
                ssim=ssim_terms(x, y)[0].mean().item(), ssim_y=ssim_terms(x_luma, y_luma)[0].mean().item(),
                ms_ssim=ms_ssim(x, y))


def save_images(original, reconstruction, out, name, crop, gain):
    # Full-res reconstruction and |error| map, plus a native-res center crop of
    # original | reconstruction | error, since 4K detail is lost when a frame is scaled down.
    original = original[0].permute(1, 2, 0).cpu().numpy()
    reconstruction = reconstruction[0].permute(1, 2, 0).cpu().numpy()
    error = np.abs(original.astype(np.int16) - reconstruction).mean(2) * gain
    error = np.repeat(error.clip(0, 255).round().astype(np.uint8)[..., None], 3, axis=2)
    Image.fromarray(reconstruction).save(out / f'{name}_recon.png')
    Image.fromarray(error[..., 0]).save(out / f'{name}_error.png')
    height, width = error.shape[:2]
    size_h, size_w = min(crop, height), min(crop, width)
    top, left = (height - size_h) // 2, (width - size_w) // 2
    panels = [a[top:top + size_h, left:left + size_w] for a in (original, reconstruction, error)]
    Image.fromarray(np.concatenate(panels, axis=1)).save(out / f'{name}_crop.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='An image, or a directory searched recursively')
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    parser.add_argument('--out-dir', default=None,
                        help='Default: inference_<checkpoint name>/ next to the checkpoint')
    parser.add_argument('--limit', type=int, default=20,
                        help='Frames to evaluate, spread evenly over the sorted list (0 = all)')
    parser.add_argument('--save-images', type=int, default=4,
                        help='Save reconstruction, error map and crop for this many frames')
    parser.add_argument('--crop', type=int, default=512, help='Side of the saved comparison crop')
    parser.add_argument('--error-gain', type=float, default=8.,
                        help='Error map brightness per 8-bit level of mean absolute error')
    parser.add_argument('--lpips', action='store_true', help='Also report LPIPS (AlexNet); needs pip install lpips')
    parser.add_argument('--tile', type=int, default=512,
                        help='Tile side in pixels, a multiple of 8 (0 = whole frame, untiled)')
    parser.add_argument('--overlap', type=int, default=128,
                        help='Pixels shared by neighboring tiles and cross-faded, a multiple of 8')
    parser.add_argument('--tile-batch', type=int, default=4, help='Tiles per forward pass')
    parser.add_argument('--amp', action='store_true', help='bfloat16 autocast, as in training (CUDA only)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.limit < 0 or args.save_images < 0 or args.crop < 1 or args.error_gain <= 0:
        parser.error('limit >= 0, save-images >= 0, crop >= 1, error-gain > 0 required')
    if args.tile < 0 or args.tile % 8 or args.overlap < 0 or args.overlap % 8 or args.tile_batch < 1:
        parser.error('tile and overlap must be non-negative multiples of 8, tile-batch >= 1')
    if args.tile and not (8 <= args.tile and args.overlap < args.tile):
        parser.error('tile >= 8 and overlap < tile required')
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

    checkpoint_path = Path(args.checkpoint).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = MBConvVAE(**checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    out = Path(args.out_dir) if args.out_dir else checkpoint_path.parent / f'inference_{checkpoint_path.stem}'
    out.mkdir(parents=True, exist_ok=True)
    print(f'{checkpoint_path}: epoch={checkpoint["epoch"]} step={checkpoint["step"]} '
          f'crop val PSNR={checkpoint["best_psnr"]:.2f}dB | {len(paths)} frames -> {out}', flush=True)

    rows = []
    for index, path in enumerate(paths):
        original = load_image(path)
        name = '_'.join(path.relative_to(source).with_suffix('').parts)
        x = original.to(device).float() / 127.5 - 1
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            decoded, mu, logvar = tiled_reconstruct(model, x, args.tile, args.overlap, args.tile_batch)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        reconstruction = to_uint8(decoded)
        # l1 and kl as the training log reports them (l1 on the [-1, 1] scale).
        losses = vae_loss(decoded, x, mu, logvar)[1]
        row = dict(frame=str(path.relative_to(source)), height=x.shape[-2], width=x.shape[-1],
                   **compare(original.to(device), reconstruction))
        if lpips_model is not None:
            with torch.inference_mode():
                row['lpips'] = lpips_model(x, reconstruction.float() / 127.5 - 1).item()
        row.update(l1=losses['l1'].item(), kl=losses['kl'].item(),
                   latent_std=mu.float().std().item(), seconds=seconds)
        rows.append(row)
        print(f'[{index+1}/{len(paths)}] {row["frame"]} ' + ' '.join(
            f'{key}={value:.4f}' for key, value in row.items() if isinstance(value, float)), flush=True)
        if index < args.save_images:
            save_images(original, reconstruction, out, name, args.crop, args.error_gain)
        del x, mu, logvar, decoded, reconstruction

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
                       tile=args.tile, overlap=args.overlap, peak_gpu_gb=peak_gb, metrics=summary),
                  file, indent=2)
    print(f'\nMean over {len(rows)} full frames' + (f', peak GPU memory {peak_gb:.1f}GB' if peak_gb else ''))
    for key, value in summary.items():
        print(f'  {key:<10} {value["mean"]:.4f} ± {value["std"]:.4f}')
    print(f'Wrote {out / "metrics.csv"} and {out / "summary.json"}')


if __name__ == '__main__':
    main()
