"""Make low-resolution training inputs from extracted HR frames.

Each HR frame is downscaled with a random interpolation (area, bilinear or bicubic),
gets Gaussian or Poisson noise (optionally gray, i.e. the same in every channel) and is
JPEG-compressed at a random quality: the resize, noise and JPEG steps of the Real-ESRGAN
degradation (arXiv 2107.10833) with its parameter ranges. Parameters are sampled per frame. The output tree
mirrors the input tree, so <hr>/<split>/<clip>/<frame>.jpg pairs with
<out>/<split>/<clip>/<frame>.jpg. A manifest.csv records what was applied to each frame.

Resumable: frames whose output already exists are skipped unless --overwrite is given.
Reproducible: each frame's randomness is derived from --seed and its relative path,
so results do not depend on iteration order or --workers.

Example: python make_lr_pairs.py --hr data/frames --out data/lr --workers 8
"""
import argparse
import csv
import hashlib
import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
INTERPOLATIONS = {'area': Image.BOX, 'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
FIELDS = ['split', 'relative_path', 'interp', 'jpeg_quality', 'noise', 'noise_level', 'gray']


def frame_rng(seed, relative_path):
    digest = hashlib.sha256(f'{seed}:{relative_path.as_posix()}'.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], 'big'))


def add_noise(image, rng, noise):
    """Real-ESRGAN style noise. Returns (image, kind, level, gray).

    Gaussian: sigma on the 0-255 scale. Poisson: shot noise scaled by `level`.
    Gray noise uses one noise map for all three channels.
    """
    prob, sigma_range, scale_range, gray_prob = noise
    if rng.random() >= prob:
        return image, 'none', 0., False
    array = np.asarray(image, dtype=np.float32) / 255.
    generator = np.random.default_rng(rng.getrandbits(64))
    gray = rng.random() < gray_prob
    if rng.random() < .5:
        kind, level = 'gaussian', rng.uniform(*sigma_range)
        shape = array.shape[:2] + ((1,) if gray else (3,))
        delta = generator.normal(0., level / 255., shape).astype(np.float32)
    else:
        kind, level = 'poisson', rng.uniform(*scale_range)
        source = array.mean(axis=2, keepdims=True) if gray else array
        levels = 2 ** math.ceil(math.log2(len(np.unique(np.round(source * 255.)))))
        delta = (generator.poisson(source * levels) / levels - source) * level
    array = np.clip(array + delta, 0., 1.)
    return Image.fromarray(np.round(array * 255.).astype(np.uint8)), kind, round(level, 2), gray


def degrade(image, rng, scale, qualities, interps, noise):
    """Return (lr_image, params dict). The JPEG step happens on save."""
    width, height = image.size
    if width % scale or height % scale:
        raise ValueError(f'{width}x{height} is not divisible by scale {scale}')
    interp = rng.choice(interps)
    quality = rng.randint(*qualities)
    lr = image.resize((width // scale, height // scale), INTERPOLATIONS[interp])
    lr, kind, level, gray = add_noise(lr, rng, noise)
    return lr, dict(interp=interp, jpeg_quality=quality, noise=kind, noise_level=level, gray=int(gray))


def process(task):
    hr_path, lr_path, relative, seed, scale, qualities, interps, noise = task
    rng = frame_rng(seed, relative)
    with Image.open(hr_path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    try:
        lr, params = degrade(image, rng, scale, qualities, interps, noise)
    except ValueError as error:
        raise ValueError(f'{hr_path}: {error}') from None
    lr_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = lr_path.with_name(lr_path.name + '.tmp')
    lr.save(temporary, format='JPEG', quality=params['jpeg_quality'])
    temporary.replace(lr_path)
    return relative, params


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hr', required=True, help='Root of HR frames, with train/ and val/')
    parser.add_argument('--out', required=True, help='Root for LR frames, mirrors --hr')
    parser.add_argument('--scale', type=int, default=4, help='Downscale factor')
    parser.add_argument('--jpeg-quality', type=int, nargs=2, default=(30, 95), metavar=('LO', 'HI'),
                        help='Inclusive JPEG quality range sampled per frame')
    parser.add_argument('--interp', nargs='+', choices=sorted(INTERPOLATIONS), default=sorted(INTERPOLATIONS),
                        help='Resize interpolations sampled per frame')
    parser.add_argument('--noise-prob', type=float, default=1., help='Chance a frame gets noise (0 disables)')
    parser.add_argument('--gaussian-sigma', type=float, nargs=2, default=(1., 30.), metavar=('LO', 'HI'),
                        help='Gaussian noise sigma range on the 0-255 scale')
    parser.add_argument('--poisson-scale', type=float, nargs=2, default=(.05, 3.), metavar=('LO', 'HI'),
                        help='Poisson noise scale range')
    parser.add_argument('--gray-prob', type=float, default=.4, help='Chance the noise is gray (same in all channels)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=0, help='Worker processes; 0 runs in this process')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate frames that already exist')
    args = parser.parse_args()
    lo, hi = args.jpeg_quality
    if args.scale < 2 or args.workers < 0 or not 1 <= lo <= hi <= 100:
        parser.error('scale >= 2, workers >= 0 and 1 <= LO <= HI <= 100 required')
    if not (0 <= args.noise_prob <= 1 and 0 <= args.gray_prob <= 1
            and 0 <= args.gaussian_sigma[0] <= args.gaussian_sigma[1]
            and 0 <= args.poisson_scale[0] <= args.poisson_scale[1]):
        parser.error('probabilities must be in [0, 1] and noise ranges must satisfy 0 <= LO <= HI')
    noise = (args.noise_prob, tuple(args.gaussian_sigma), tuple(args.poisson_scale), args.gray_prob)
    hr_root, out = Path(args.hr), Path(args.out)
    if not hr_root.is_dir():
        parser.error(f'{hr_root} is not a directory')
    if out.resolve() == hr_root.resolve():
        parser.error('--out must differ from --hr')
    frames = sorted(p for p in hr_root.rglob('*') if p.suffix.lower() in EXTENSIONS)
    if not frames:
        parser.error(f'No supported images under {hr_root}')
    interps = sorted(set(args.interp))
    tasks, skipped = [], 0
    for hr_path in frames:
        relative = hr_path.relative_to(hr_root).with_suffix('.jpg')
        lr_path = out / relative
        if lr_path.exists() and not args.overwrite:
            skipped += 1
            continue
        tasks.append((hr_path, lr_path, relative, args.seed, args.scale, (lo, hi), interps, noise))
    print(f'{len(frames)} HR frames, {skipped} already done, {len(tasks)} to generate', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.csv'
    rows = {}
    if manifest_path.exists():
        with open(manifest_path, newline='') as file:
            for row in csv.DictReader(file):
                rows[row['relative_path']] = row
    if tasks:
        if args.workers:
            with ProcessPoolExecutor(args.workers) as pool:
                results = pool.map(process, tasks, chunksize=4)
                done = consume(results, len(tasks))
        else:
            done = consume(map(process, tasks), len(tasks))
        for relative, params in done:
            split = relative.parts[0] if len(relative.parts) > 1 else ''
            rows[relative.as_posix()] = dict(split=split, relative_path=relative.as_posix(), **params)
        temporary = manifest_path.with_suffix('.csv.tmp')
        with open(temporary, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            for key in sorted(rows):
                writer.writerow(rows[key])
        temporary.replace(manifest_path)
    print(f'Done: {len(tasks)} LR frames written to {out}, manifest at {manifest_path}')


def consume(results, total):
    done = []
    for index, result in enumerate(results, 1):
        done.append(result)
        if index % 50 == 0 or index == total:
            print(f'\r{index}/{total}', end='', flush=True)
    if total:
        print()
    return done


if __name__ == '__main__':
    main()
