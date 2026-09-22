#!/usr/bin/env python3
"""Audit pinned LIBERO parquet footers without downloading image payloads."""
import concurrent.futures
import io
import json
import struct
import subprocess
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', default='86958911c0f959db2bbbdb107eb3e17c5f9c798e')
    args = parser.parse_args()
    import pyarrow.parquet as pq
    host = 'https://hf-mirror.com'
    def fetch(url, byte_range=None):
        command = ['curl', '-fLsS', '--retry', '2', '--connect-timeout', '15', '--max-time', '90']
        if byte_range:
            a, b = byte_range
            command += ['--max-filesize', '2097152', '-H', f'Range: bytes={a}-{b}']
            url += f'?footer_range={a}'
        value = subprocess.check_output(command + [url])
        if byte_range and len(value) != b-a+1:
            raise ValueError('Server did not return requested byte range')
        return value
    tree = json.loads(fetch(f'{host}/api/datasets/HuggingFaceVLA/libero/tree/{args.revision}?recursive=true&limit=1000'))
    files = [x for x in tree if x['type']=='file' and x['path'].startswith('data/') and x['path'].endswith('.parquet')]
    def inspect(item):
        size = item['size']
        url = f'{host}/datasets/HuggingFaceVLA/libero/resolve/{args.revision}/{item["path"]}'
        tail = fetch(url, (size-8,size-1))
        if tail[4:] != b'PAR1':
            raise ValueError(item['path'])
        length = struct.unpack('<I',tail[:4])[0]
        meta = pq.read_metadata(io.BytesIO(b'PAR1'+fetch(url,(size-8-length,size-1))))
        groups = []
        for i in range(meta.num_row_groups):
            group = meta.row_group(i)
            stats = {}
            for j in range(group.num_columns):
                col = group.column(j)
                if col.path_in_schema in ('index','episode_index','task_index'):
                    s = col.statistics
                    if s is None or not s.has_min_max:
                        raise ValueError('Missing index statistics')
                    stats[col.path_in_schema] = [s.min,s.max]
            groups.append(stats)
        return dict(path=item['path'],size=size,lfs=item.get('lfs'),rows=meta.num_rows,groups=groups)
    args.output.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(inspect,x):x for x in files}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            target = args.output / (result['path'].replace('/','__')+'.json')
            target.write_text(json.dumps(result,indent=2)+'\n')
            print(result['path'],result['rows'],flush=True)
    (args.output/'revision.txt').write_text(args.revision+'\n')

if __name__ == '__main__':
    main()
