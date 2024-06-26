#!/bin/bash

echo "**************************************************************"
echo "*Tests for muliple-walker ABF: May take a few minutes to run."
echo "*by Jeff Comer <jeffcomer at gmail dot com>"
echo ""

if (( $# != 1 )); then
    echo "Usage: run_mw_tests.sh namd_exec_location"
    exit -1
fi
NAMD_EXEC=$1

# Runner for the netlrts version
RUNNER="${HOME}/progs/namd/Linux-x86_64-g++.netlrts/charmrun ++local +p4"

# Runner for the MPI version
#RUNNER="mpirun -n 4"

unset l

i=0
for f in mw_independent.namd mw_shared.namd mw_selection.namd; do
    nam=${f%.*}
    rm -f out_${nam}.*.log
    $RUNNER $NAMD_EXEC +replicas 4 $f +stdout out_${nam}.%d.log

    if [ $f = "mw_independent.namd" ]; then
      grad=out_${nam}.0.grad
    else
      grad=out_${nam}.0.all.grad
    fi
    rmsd=$( paste -d ' ' da10_12-32_indepSaga0.0.grad $grad | awk '!/^#/ && NF==4 {n++; s+=($4-$2)*($4-$2)}; END {print sqrt(s/n)}' )
    echo "RMSD $rmsd"
    l[$i]=$( printf "%s\t\t%.5f" $nam $rmsd )
    ((i++))
done

l[$i]=$( printf "EXPECTED" )
((i++))

for f in mw_independent.namd mw_shared.namd mw_selection.namd; do
    nam=${f%.*}
    grad=ExpectedResults/out_${nam}.0.grad
    rmsd=$( paste -d ' ' da10_12-32_indepSaga0.0.grad $grad | awk '!/^#/ && NF==4 {n++; s+=($4-$2)*($4-$2)}; END {print sqrt(s/n)}' )
    l[$i]=$( printf "%s\t\t%.5f" $nam $rmsd )
    ((i++))
done

echo ""
echo "Note that the values fluctuate considerably between runs, but"
echo "on average RMSD(selection) < RMSD(shared) < RMSD(independent)."
echo "Root-mean-square deviation from converged grad (kcal mol^-1 Å^-2)"
for i in ${!l[@]}; do
    echo "${l[$i]}"
done
