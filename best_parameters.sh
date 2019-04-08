#!/bin/sh
# USE THIS SCRIPT TO READ THE OUTPUT OF THE HYPER-OPTIMIZATION
echo "BEST PARAMETERS"
for i in $(awk '/^Av[ae]rage.*[0-9]$/ {print $3}' hyperoptimization.txt | sort | tail -n$1)
do
	grep -B151 $i hyperoptimization.txt | head -n1
	grep -B1 $i hyperoptimization.txt
	echo "______________"
done
echo
echo "Total number of evaluations:               $(grep TESTING hyperoptimization.txt | wc -l)"
echo "Total number of evaluations in the domain: $(grep Av[ae]rage hyperoptimization.txt | wc -l)"
