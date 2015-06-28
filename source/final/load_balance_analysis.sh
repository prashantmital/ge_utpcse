#--This is a script that produces data that can be
#	used to demonstrate the load-balancing
#	effect of cyclic-striped partioning

rm screenout.load_balance

nrows=$1
nproc=$2
num=$(($nrows/$nproc))

echo $num

for (( i=1; i<=$num; i++ ));do
    if [ $((num%$i)) == 0 ]; then
	ibrun -np $nproc main.out nrows $nrows ncols $nrows nproc $nproc nreps $i >> screenout.load_balance
    fi
done

