"""Download and extract the Inter4K dataset (1000 UHD 60fps clips, ~15.5GB zip).

Resumable: rerun the same command after an interruption. The zip is extracted to
<out>/Inter4K/60fps/UHD/*.mp4 (ready for prepare_video_frames.py --videos) and
then deleted. Peak disk use is ~31GB (archive plus extracted clips).

--max-clips N fetches only clips 1..N (about 15.5MB each) directly out of the remote
archive with HTTP range requests, without downloading the whole zip. Useful for a
small test set; a later full run reuses the clips already on disk.
Dataset and licence: https://github.com/alexandrosstergiou/Inter4K
"""
import argparse
import http.client
import re
import shutil
import struct
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

FILE_ID = '1YNdzhk0mxFln9_2MetHF1cJIsoeVCskh'
URL = f'https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t'
CLIP_COUNT = 1000
CHUNK = 1 << 20
ZIP64_SENTINEL = 0xFFFFFFFF


def request(start, end=''):
    headers = {'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={start}-{end}'}
    response = urllib.request.urlopen(urllib.request.Request(URL, headers=headers), timeout=60)
    if response.headers.get_content_type() == 'text/html':
        with response:
            title = re.search(r'<title>(.*?)</title>', response.read().decode('utf-8', 'replace'), re.S)
        raise RuntimeError(f'Google Drive returned "{title.group(1).strip() if title else "a web page"}" '
                           'instead of the archive. Its download quota usually resets within 24 hours, '
                           'so rerun later, or copy an Inter4K.zip from another machine and pass it '
                           'with --zip')
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


def remote_clips(total):
    """Read the archive's table of contents over HTTP. Returns {clip number: (offset, method, csize, usize, crc)}."""
    with request(max(0, total - 65536), total - 1) as response:
        tail = response.read()
    eocd = tail.rfind(b'PK\x05\x06')
    if eocd < 0:
        raise RuntimeError('Could not find the archive index')
    count, cd_size, cd_offset = struct.unpack_from('<HII', tail, eocd + 10)
    if ZIP64_SENTINEL in (cd_size, cd_offset) or count == 0xFFFF:
        zip64 = tail.rfind(b'PK\x06\x06')
        if zip64 < 0:
            raise RuntimeError('Could not find the zip64 archive index')
        cd_size, cd_offset = struct.unpack_from('<QQ', tail, zip64 + 40)
    with request(cd_offset, cd_offset + cd_size - 1) as response:
        directory = response.read()
    clips, position = {}, 0
    while position + 46 <= len(directory) and directory[position:position + 4] == b'PK\x01\x02':
        (method, crc, csize, usize, name_length, extra_length, comment_length,
         offset) = struct.unpack_from('<HxxxxIIIHHHxxxxxxxxI', directory, position + 10)
        name = directory[position + 46:position + 46 + name_length].decode('utf-8')
        extra = directory[position + 46 + name_length:position + 46 + name_length + extra_length]
        # zip64 extra field holds, in order, only the fields whose 32-bit value is the sentinel.
        at = 0
        while at + 4 <= len(extra):
            field_id, field_size = struct.unpack_from('<HH', extra, at)
            if field_id == 1:
                values = iter(struct.unpack_from(f'<{field_size // 8}Q', extra, at + 4))
                usize = next(values) if usize == ZIP64_SENTINEL else usize
                csize = next(values) if csize == ZIP64_SENTINEL else csize
                offset = next(values) if offset == ZIP64_SENTINEL else offset
            at += 4 + field_size
        match = re.fullmatch(r'Inter4K/60fps/UHD/(\d+)\.mp4', name)
        if match:
            clips[int(match.group(1))] = (offset, method, csize, usize, crc)
        position += 46 + name_length + extra_length + comment_length
    if len(clips) != CLIP_COUNT:
        raise RuntimeError(f'Expected {CLIP_COUNT} clips in the archive index, found {len(clips)}')
    return clips


def fetch_clip(entry, destination):
    """Download one archive member by byte range, inflate it and verify its CRC."""
    offset, method, csize, usize, crc = entry
    if method not in (0, 8):
        raise RuntimeError(f'Unsupported compression method {method} for {destination.name}')
    temporary = destination.with_name(destination.name + '.part')
    # Local header is 30 bytes plus name and extra; over-request the header slack and stop early.
    with request(offset, offset + 30 + 65535 + csize) as response, open(temporary, 'wb') as file:
        header = response.read(30)
        if header[:4] != b'PK\x03\x04':
            raise RuntimeError(f'Bad local header for {destination.name}')
        name_length, extra_length = struct.unpack_from('<HH', header, 26)
        response.read(name_length + extra_length)
        inflater = zlib.decompressobj(-15) if method == 8 else None
        remaining, checksum, written = csize, 0, 0
        while remaining:
            chunk = response.read(min(CHUNK, remaining))
            if not chunk:
                raise ConnectionError(f'Connection closed with {remaining} bytes left of {destination.name}')
            remaining -= len(chunk)
            data = inflater.decompress(chunk) if inflater else chunk
            if inflater and not remaining:
                data += inflater.flush()
            checksum = zlib.crc32(data, checksum)
            written += len(data)
            file.write(data)
    if written != usize or checksum != crc:
        temporary.unlink()
        raise RuntimeError(f'{destination.name} failed verification; rerun to retry')
    temporary.replace(destination)


def download_subset(videos, count, retries):
    total = remote_size()
    clips = remote_clips(total)
    wanted = sorted(clips)[:count]
    pending = [n for n in wanted if not (videos / f'{n}.mp4').is_file()
               or (videos / f'{n}.mp4').stat().st_size != clips[n][3]]
    size = sum(clips[n][3] for n in pending)
    print(f'{len(wanted)} clips requested, {len(wanted) - len(pending)} already on disk, '
          f'{len(pending)} to fetch ({size/1e9:.2f} GB)', flush=True)
    require_space(videos, size)
    started, done = time.monotonic(), 0
    for index, number in enumerate(pending, 1):
        for attempt in range(1, retries + 1):
            try:
                fetch_clip(clips[number], videos / f'{number}.mp4')
                break
            except (OSError, http.client.HTTPException) as error:
                if attempt == retries:
                    raise
                print(f'\nClip {number} interrupted ({error}); retrying in 10s', flush=True)
                time.sleep(10)
        done += clips[number][3]
        rate = done / max(time.monotonic() - started, 1e-9) / 1e6
        print(f'\r{index}/{len(pending)} clips, {done/1e9:.2f}/{size/1e9:.2f} GB, {rate:.1f} MB/s', end='', flush=True)
    if pending:
        print()
    return len(wanted)


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
    parser.add_argument('--max-clips', type=int, default=None, metavar='N',
                        help='Fetch only clips 1..N from the remote archive instead of the whole zip')
    args = parser.parse_args()
    if args.retries < 1:
        parser.error('retries >= 1 required')
    if args.max_clips is not None and not 1 <= args.max_clips <= CLIP_COUNT:
        parser.error(f'1 <= max-clips <= {CLIP_COUNT} required')
    if args.max_clips is not None and args.zip:
        parser.error('--max-clips fetches from the remote archive; it cannot be combined with --zip')
    if args.zip and not Path(args.zip).is_file():
        parser.error(f'{args.zip} does not exist')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = out / 'Inter4K' / '60fps' / 'UHD'
    marker = out / 'Inter4K' / '.complete'
    if marker.exists():
        print(f'Inter4K already extracted to {videos}')
        return
    if args.max_clips is not None:
        videos.mkdir(parents=True, exist_ok=True)
        count = download_subset(videos, args.max_clips, args.retries)
        print(f'Done: {count} clips in {videos} (subset; rerun without --max-clips for all {CLIP_COUNT})')
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
