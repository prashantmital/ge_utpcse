#--This is a script that produces data that can be
#	used to demonstrate the load-balancing
#	effect of cyclic-striped partioning

rm screenout.strong_scaling

nrows=$1
nprocmax=$2

#perform serial run
ibrun -np 1 main.out nrows $nrows ncols $nrows nproc 1 nreps 1 >> screenout.strong_scaling


for (( i=2; i<=$nprocmax; i++ ));do
    if [ $(($nrows%$i)) == 0 ]; then
	nreps=$(($nrows/$i))
	ibrun -np $i main.out nrows $nrows ncols $nrows nproc $i nreps $nreps >> screenout.strong_scaling
    fi
done

