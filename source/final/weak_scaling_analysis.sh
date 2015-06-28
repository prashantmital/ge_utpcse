#--This is a script that produces data that can be
#	used to demonstrate the load-balancing
#	effect of cyclic-striped partioning

rm screenout.weak_scaling

nrowsinit=512
nrows=$nrowsinit

#perform serial run
ibrun -np 1 main.out nrows $nrows ncols $nrows nproc 1 nreps 1 >> screenout.weak_scaling

for (( i=1; i<=4; i++ ));do
    nproc=$((2**($i*2)))
    nrows=$(((2**$i)*$nrowsinit))
    alpha=$(($nrows*$nrows/$nproc))
    nreps=$(($nrows/$nproc))
    echo $nproc
    echo $nrows
    echo $alpha
    echo $nreps	
    ibrun -np $nproc main.out nrows $nrows ncols $nrows nproc $nproc nreps $nreps >> screenout.weak_scaling
done

