#!/usr/bin/env python
import gzip
import os
import shutil
import sys

compress_file_endings = [
    'txt',
    'log',
    'data',
    'xml',
    'out',
    'dump',
    'lammps',
    'in'
]
min_size_in_MB=2
min_size_in_B = min_size_in_MB*1e6


def compress_than_delete(filepath):
    with open(filepath, 'rb') as f_in:
        with gzip.open(f'{filepath}.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    os.remove(filepath)

def recursive_check_dir(dirpath):
    for root, dirs, files in os.walk(dirpath,followlinks=False):
        for f in files:
            fullpath = os.path.join(root, f)
            if os.path.islink(fullpath):
                continue
            if os.path.getsize(fullpath) > min_size_in_B:
                if f.split('.')[-1] in compress_file_endings:
                    compress_than_delete(
                        fullpath
                    )
        for d in dirs:
            recursive_check_dir(
                os.path.join(root, d)
            )

if __name__ == '__main__':
    recursive_check_dir(sys.argv[1])

        