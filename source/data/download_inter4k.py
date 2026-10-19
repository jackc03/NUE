"""Download and extract the Inter4K dataset (1000 UHD 60fps clips, ~15.5GB zip).

Resumable: rerun the same command after an interruption. The zip is extracted to
<out>/Inter4K/60fps/UHD/*.mp4 (ready for prepare_video_frames.py --videos) and
then deleted. Peak disk use is ~31GB (archive plus extracted clips).
Dataset and licence: https://github.com/alexandrosstergiou/Inter4K
"""
import argparse
import http.client
import re
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

FILE_ID = '1YNdzhk0mxFln9_2MetHF1cJIsoeVCskh'
URL = f'https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t'
CLIP_COUNT = 1000
CHUNK = 1 << 20


def request(start, end=''):
    headers = {'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={start}-{end}'}
    response = urllib.request.urlopen(urllib.request.Request(URL, headers=headers), timeout=60)
    if response.headers.get_content_type() == 'text/html':
        response.close()
        raise RuntimeError('Google Drive returned a web page instead of the archive, likely '
                           'its download quota. Retry later, or download '
                           f'https://drive.google.com/file/d/{FILE_ID}/view in a browser '
                           'and pass the file with --zip')
    return response


def remote_size():
    with request(0, 0) as response:
        match = re.fullmatch(r'bytes 0-0/(\d+)', response.headers.get('Content-Range', ''))
        if response.status != 206 or match is None:
            raise RuntimeError('Server did not report the archive size')
    return int(match.group(1))


def require_space(path, needed):
    free = shutil.disk_usage(path).free
    if free < needed:
        raise SystemExit(f'Need {needed/1e9:.1f} GB free in {path}, only {free/1e9:.1f} GB available')


def download(part, total):
    offset = part.stat().st_size if part.exists() else 0
    if offset == total:
        return
    with request(offset) as response, open(part, 'ab') as file:
        if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
            raise RuntimeError(f'Server did not resume at byte {offset}; delete {part} and retry')
        done, started, last = offset, time.monotonic(), 0.
        while chunk := response.read(CHUNK):
            file.write(chunk)
            done += len(chunk)
            now = time.monotonic()
            if now - last >= 2 or done == total:
                rate = (done - offset) / max(now - started, 1e-9) / 1e6
                print(f'\r{done/1e9:.2f}/{total/1e9:.2f} GB ({100*done/total:.1f}%) {rate:.1f} MB/s',
                      end='', flush=True)
                last = now
        print()
    if done != total:
        raise ConnectionError(f'Download stopped at {done} of {total} bytes')


def extract(archive, out):
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        # Skip files already fully extracted; a partial file has the wrong size.
        pending = [m for m in members if not ((out / m.filename).is_file()
                                              and (out / m.filename).stat().st_size == m.file_size)]
        require_space(out, sum(m.file_size for m in pending))
        for index, member in enumerate(pending, 1):
            zf.extract(member, out)  # Raises BadZipFile on a CRC mismatch
            print(f'\rExtracted {index}/{len(pending)}', end='', flush=True)
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='Directory for the archive and extracted clips')
    parser.add_argument('--zip', default=None, help='Use an already downloaded Inter4K.zip')
    parser.add_argument('--keep-zip', action='store_true',
                        help='Keep the downloaded archive (deleted after extraction by default)')
    parser.add_argument('--retries', type=int, default=5, help='Resume attempts after network errors')
    args = parser.parse_args()
    if args.retries < 1:
        parser.error('retries >= 1 required')
    if args.zip and not Path(args.zip).is_file():
        parser.error(f'{args.zip} does not exist')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = out / 'Inter4K' / '60fps' / 'UHD'
    marker = out / 'Inter4K' / '.complete'
    if marker.exists():
        print(f'Inter4K already extracted to {videos}')
        return
    archive = Path(args.zip) if args.zip else out / 'Inter4K.zip'
    if not archive.exists():
        part = archive.with_name(archive.name + '.part')
        total = remote_size()
        offset = part.stat().st_size if part.exists() else 0
        if offset > total:
            part.unlink()
            offset = 0
        # Room for the rest of the download plus roughly the same again to extract.
        require_space(out, total - offset + total)
        print(f'Downloading {archive.name} ({total/1e9:.2f} GB) to {out}'
              + (f', resuming at {offset/1e9:.2f} GB' if offset else ''), flush=True)
        for attempt in range(1, args.retries + 1):
            try:
                download(part, total)
                break
            except (OSError, http.client.HTTPException) as error:
                if attempt == args.retries:
                    raise
                print(f'\nDownload interrupted ({error}); retrying in 10s', flush=True)
                time.sleep(10)
        part.replace(archive)
    print(f'Extracting {archive} to {out}', flush=True)
    extract(archive, out)
    count = sum(1 for p in videos.glob('*.mp4'))
    if count != CLIP_COUNT:
        raise RuntimeError(f'Expected {CLIP_COUNT} clips in {videos}, found {count}')
    marker.touch()
    if not args.keep_zip and not args.zip:  # Never delete an archive the user supplied
        archive.unlink()
        print(f'Deleted {archive}')
    print(f'Done: {count} clips in {videos}')


if __name__ == '__main__':
    main()
