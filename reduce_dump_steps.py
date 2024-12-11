#!/usr/bin/env python
import argparse

def reduce_lammps_dump(infile, outfile, every=10, start=0):
    with open(infile, "r") as fr:
        with open(outfile, "w") as fw:
            for l in fr:  # noqa: E741
                if l.startswith("ITEM: TIMESTEP"):
                    if start % every == 0:
                        write = True
                    else:
                        write = False
                    start += 1

                if write:
                    fw.write(l)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog="ReduceLammpsDump",
        description=(
            "Rewrite lammps dump file with reduced amount of steps"
        ),
    )
     
    parser.add_argument(
        "--infile",
        default="dump.out",
        type=str,
        help="original dump file",
        required=False,
    )
    parser.add_argument(
        "--outfile",
        default="reduce_dump.out",
        help="new dump file",
        required=False,
    )
    parser.add_argument(
        "--every",
        default=10,
        help="Write every nth step to the new file",
        type=int,
    )

    reduce_lammps_dump(
        infile=parser.infile,
        outfile=parser.outfile,
        every=parser.every,
        start=0
    )