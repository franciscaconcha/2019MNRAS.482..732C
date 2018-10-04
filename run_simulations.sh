#!/bin/bash

Nruns=$5
Initstars=$1
Finalstars=$2
InitRvir=$8
FinalRvir=$9

for ((i=$Initstars; i<=$Finalstars; i=i+250)); do
    for k in 0.1 0.3 0.5 0.8 1.0; do
        for ((j=0; j<$Nruns; j++)); do
            amuse.sh cluster_with_viscous_disks.py -N $i -s $3 -a $4 -R $k -n $j -c $7 -l ${10} -k ${11} -e ${12}
        done
    done
done
