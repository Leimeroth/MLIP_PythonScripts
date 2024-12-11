#!/usr/bin/bash
set -e
if [[ $1 == "" ]] ; then
    echo 'No arguments given, exiting'
    exit 0
fi

#ulimit -s unlimited
#export OMP_NUM_THREADS=1

mkdir Continue
cp input.nn Continue/
cp *.data Continue/

cp weights.013.000$1.out Continue/weights.013.data
cp weights.029.000$1.out Continue/weights.029.data
cp weights.040.000$1.out Continue/weights.040.data

cd Continue
#sed -i -e "s/#use_old_weights_short/use_old_weights_short/g" input.nn
#mpirun -np 16 nnp-train > mode2.out