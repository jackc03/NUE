"""Make low-resolution video clips from the HR source videos.

Each clip is Gaussian-blurred at full resolution, downscaled with a random scaler
(area, bilinear or bicubic), gets temporal Gaussian noise of random strength, and is
re-encoded with H.264 at a random CRF: the blur, resize, noise and compression steps
of the Real-ESRGAN degradation, the video counterpart of make_lr_pairs.py.
Output names follow the frame folders (<index>_<stem>) so a clip pairs with its
frames: data/videos_lr/<split>/00006_27.mp4 <-> data/frames/<split>/00006_27/.
A manifest.csv records what was applied to each clip.

Requires FFmpeg in PATH or the imageio-ffmpeg package.

Example: python make_lr_videos.py --videos data/Inter4K --frames data/frames --out data/videos_lr
"""
import argparse
import csv
import hashlib
import random
import shutil
import subprocess
from pathlib import Path

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.webm'}
SCALERS = ['area', 'bilinear', 'bicubic']


def find_ffmpeg():
    found = shutil.which('ffmpeg')
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def clip_rng(seed, name):
    digest = hashlib.sha256(f'{seed}:{name}'.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], 'big'))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--videos', required=True, help='Directory of HR source clips')
    parser.add_argument('--frames', required=True, help='Frame root with train/ and val/, used for naming and split')
    parser.add_argument('--out', required=True, help='Output root, gets train/ and val/')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--crf', type=int, nargs=2, default=(28, 40), metavar=('LO', 'HI'),
                        help='Inclusive H.264 CRF range sampled per clip (higher = worse)')
    parser.add_argument('--blur', type=float, nargs=2, default=(.2, 3.), metavar=('LO', 'HI'),
                        help='Gaussian blur sigma range in HR pixels, applied before downscaling (0 0 disables)')
    parser.add_argument('--noise-prob', type=float, default=1., help='Chance a clip gets noise (0 disables)')
    parser.add_argument('--noise', type=int, nargs=2, default=(1, 30), metavar=('LO', 'HI'),
                        help='Inclusive FFmpeg noise strength range (0-100), new noise every frame')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        parser.error('FFmpeg not found; install it or pip install imageio-ffmpeg')
    videos = {p.stem: p for p in Path(args.videos).rglob('*') if p.suffix.lower() in VIDEO_EXTENSIONS}
    frame_dirs = sorted(d for d in Path(args.frames).glob('*/*') if d.is_dir())
    if not frame_dirs:
        parser.error(f'No <split>/<clip> folders under {args.frames}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, frame_dir in enumerate(frame_dirs, 1):
        split, name = frame_dir.parent.name, frame_dir.name
        stem = name.split('_', 1)[1]
        source = videos.get(stem)
        if source is None:
            raise FileNotFoundError(f'No source video with stem {stem} for {frame_dir}')
        target = out / split / f'{name}.mp4'
        rng = clip_rng(args.seed, name)
        scaler = rng.choice(SCALERS)
        blur = round(rng.uniform(*args.blur), 2)
        crf = rng.randint(*args.crf)
        noise = rng.randint(*args.noise) if rng.random() < args.noise_prob else 0
        rows.append(dict(split=split, clip=f'{split}/{name}.mp4', source=source.name,
                         blur=blur, scaler=scaler, noise=noise, crf=crf))
        if target.exists() and not args.overwrite:
            print(f'{index}/{len(frame_dirs)} skip {target}', flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.tmp.mp4')
        command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
                   '-vf', (f'gblur=sigma={blur},' if blur else '')
                          + f'scale=iw/{args.scale}:ih/{args.scale}:flags={scaler}'
                          + (f',noise=alls={noise}:allf=t' if noise else ''),
                   '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf), '-pix_fmt', 'yuv420p',
                   '-an', str(temporary)]
        subprocess.run(command, check=True)
        temporary.replace(target)
        print(f'{index}/{len(frame_dirs)} {source.name} -> {target} (blur={blur}, {scaler}, noise={noise}, crf={crf})', flush=True)
    with open(out / 'manifest.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['split', 'clip', 'source', 'blur', 'scaler', 'noise', 'crf'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Done: {len(rows)} clips, manifest at {out / "manifest.csv"}')


if __name__ == '__main__':
    main()
