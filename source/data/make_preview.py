"""Render contact sheets comparing HR frames with their LR counterparts.

sheet_crops.png: center crops of HR at 1:1 pixels next to the LR frame upscaled
x4 (bicubic), so degradation is visible at pixel level.
sheet_full.png: whole frames side by side, downsized to fit.
Captions come from the LR manifest.csv when present.

Example: python make_preview.py --hr data/frames --lr data/lr --out data/previews
"""
import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw

BAR = 24


def caption(draw, x, y, width, text):
    draw.rectangle((x, y, x + width, y + BAR), fill='black')
    draw.text((x + 6, y + 6), text, fill='white')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hr', required=True)
    parser.add_argument('--lr', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--rows', type=int, default=4)
    parser.add_argument('--crop', type=int, default=700, help='HR crop width; height is crop/2')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    hr_root, lr_root, out = Path(args.hr), Path(args.lr), Path(args.out)
    frames = sorted(p for p in lr_root.rglob('*.jpg'))
    if len(frames) < args.rows:
        parser.error(f'Need at least {args.rows} LR frames under {lr_root}')
    manifest = {}
    if (lr_root / 'manifest.csv').exists():
        with open(lr_root / 'manifest.csv', newline='') as file:
            manifest = {row['relative_path']: row for row in csv.DictReader(file)}
    picks = random.Random(args.seed).sample(frames, args.rows)
    out.mkdir(parents=True, exist_ok=True)
    cw, ch = args.crop, args.crop // 2
    crops = Image.new('RGB', (2 * cw, args.rows * (ch + BAR)), 'black')
    full_w = 640
    fulls = None
    draw_c = ImageDraw.Draw(crops)
    for row, lr_path in enumerate(picks):
        relative = lr_path.relative_to(lr_root)
        hr_path = hr_root / relative
        hr, lr = Image.open(hr_path).convert('RGB'), Image.open(lr_path).convert('RGB')
        scale = hr.width // lr.width
        info = manifest.get(relative.as_posix(), {})
        label = ', '.join(f'{k}={info[k]}' for k in ('interp', 'noise', 'noise_level', 'gray', 'jpeg_quality') if k in info)
        up = lr.resize(hr.size, Image.BICUBIC)
        box = ((hr.width - cw) // 2, (hr.height - ch) // 2, (hr.width + cw) // 2, (hr.height + ch) // 2)
        y = row * (ch + BAR)
        crops.paste(hr.crop(box), (0, y))
        crops.paste(up.crop(box), (cw, y))
        caption(draw_c, 0, y + ch, cw, 'HR center crop, 1:1 pixels')
        caption(draw_c, cw, y + ch, cw, f'LR x{scale} bicubic upscale ({label})')
        full_h = round(hr.height * full_w / hr.width)
        if fulls is None:
            fulls = Image.new('RGB', (2 * full_w, args.rows * (full_h + BAR)), 'black')
            draw_f = ImageDraw.Draw(fulls)
        fulls.paste(hr.resize((full_w, full_h), Image.LANCZOS), (0, y := row * (full_h + BAR) + BAR))
        fulls.paste(up.resize((full_w, full_h), Image.LANCZOS), (full_w, y))
        caption(draw_f, 0, y - BAR, full_w, f'HR {hr.width}x{hr.height}  {relative.as_posix()}')
        caption(draw_f, full_w, y - BAR, full_w, f'LR {lr.width}x{lr.height} ({label}) shown upscaled')
    crops.save(out / 'sheet_crops.png')
    fulls.save(out / 'sheet_full.png')
    print(f'Wrote {out / "sheet_crops.png"} and {out / "sheet_full.png"}')


if __name__ == '__main__':
    main()
