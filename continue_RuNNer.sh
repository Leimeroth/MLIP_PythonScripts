#!/usr/bin/bash
set -e
if [[ $1 == "" ]] ; then
    echo 'No arguments given, exiting'
    exit 0
fi

ulimit -s unlimited
export OMP_NUM_THREADS=6

mkdir Continue
cp input.nn Continue/
cp *.data Continue/
cp 000$1.short.013.out Continue/weights.013.data
cp 000$1.short.029.out Continue/weights.029.data
cp 000$1.short.040.out Continue/weights.040.data

cd Continue
sed -i -e "s/#use_old_weights_short/use_old_weights_short/g" input.nn
RuNNer.x > mode2.out