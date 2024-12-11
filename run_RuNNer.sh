ulimit -s unlimited
export OMP_NUM_THREADS=4
sed -i -e "s/runner_mode.*/runner_mode 1/g" input.nn
RuNNer.x > mode1.out

# run the fit
sed -i -e "s/runner_mode.*/runner_mode 2/g" input.nn
RuNNer.x > mode2.out
