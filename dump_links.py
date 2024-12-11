#!/usr/bin/env python

import argparse
import os


def create_symlinks(
    file, replace_dir, vars, symlink_dir=".", basename=None, symbol="*"
):
    if symlink_dir != ".":
        os.makedirs(symlink_dir, exist_ok=True)
    for var in vars:
        rdir = replace_dir.replace(symbol, str(var))
        filepath = f"{rdir}/{file}"
        if basename is None:
            if symlink_dir == ".":
                linkpath = f"{rdir}{file}_{var}".replace("/", "_")
            else:
                linkpath = f"{symlink_dir}/{file}_{var}"
        else:
            linkpath = f"{basename}_{var}"
        os.symlink(filepath, linkpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Symlinks",
        description=(
            "Create symlinks to multiple files in different directory in one directory"
            "In this case multiple lammps dump files in different directories are linked"
            "into a single directory to be read by lammps."
        ),
    )
    parser.add_argument(
        "--filename",
        default="dump.out",
        type=str,
        help="Basename of file that should be linked",
        required=False,
    )
    parser.add_argument(
        "--sdir",
        default=".",
        help="Directory into which files will be linked. Defaults to '.', replacing dir structure with long names",
        required=False,
    )
    parser.add_argument(
        "--rdir",
        help="Directory name where '*' will be replaced by input numbers to look for files",
        type=str,
    )

    parser.add_argument(
        "--basename",
        help="",
        default=None,
        required=False,
    )

    parser.add_argument("--n1", default=1, type=int, required=False)
    parser.add_argument("--n2", type=int, required=False)

    args = parser.parse_args()
    arg_lst = list(range(args.n1, args.n2 + 1))
    create_symlinks(
        file=args.filename,
        basename=args.basename,
        symlink_dir=args.sdir,
        replace_dir=args.rdir,
        vars=arg_lst,
        symbol="*",
    )
