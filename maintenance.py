import os


def delete_files(list_file):
    with open(list_file) as f:
        for l in f:
            os.remove(l.strip())


def compress_files(list_file):
    with open(list_file) as f:
        for l in f:
            if ".bz2" in l:
                continue
            try:
                os.system(f"bzip2 {l}")
            except FileNotFoundError:
                pass
