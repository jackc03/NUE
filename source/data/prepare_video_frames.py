"""Split video clips, then extract frames for VAE training.

Requires FFmpeg in PATH, or the imageio-ffmpeg package (pip install imageio-ffmpeg),
which bundles a binary. Split is by source clip, never by frame.
"""
import argparse
import random
import shutil
import subprocess
from pathlib import Path

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.webm'}


def find_ffmpeg():
    found = shutil.which('ffmpeg')
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', required=True, help='Directory of source clips')
    parser.add_argument('--out', required=True, help='Output directory with train/ and val/')
    parser.add_argument('--fps', type=float, default=1., help='Frames per second to extract')
    parser.add_argument('--val-fraction', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.fps <= 0 or not 0 < args.val_fraction < 1:
        parser.error('fps > 0 and 0 < val-fraction < 1 required')
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        parser.error('FFmpeg is not available in PATH; install it or run: pip install imageio-ffmpeg')
    videos = sorted(p for p in Path(args.videos).rglob('*')
                    if p.suffix.lower() in VIDEO_EXTENSIONS)
    if len(videos) < 2:
        parser.error('At least two video clips are needed for a train/val split')
    random.Random(args.seed).shuffle(videos)
    val_count = max(1, round(len(videos) * args.val_fraction))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for index, video in enumerate(videos):
        split = 'val' if index < val_count else 'train'
        destination = out / split / f'{index:05d}_{video.stem}'
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f'{destination} already has files; use a fresh output directory')
        destination.mkdir(parents=True, exist_ok=True)
        command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-i', str(video),
                   '-vf', f'fps={args.fps:g}', '-q:v', '2',
                   str(destination / 'frame_%05d.jpg')]
        subprocess.run(command, check=True)
        if not any(destination.iterdir()):
            raise RuntimeError(f'No frames extracted from {video}')
        print(f'{split}: {video} -> {destination}', flush=True)


if __name__ == '__main__':
    main()
