#!/bin/sh
# USE THIS SCRIPT TO READ THE OUTPUT OF THE HYPER-OPTIMIZATION WITH CROSS-VALIDATION
echo "BEST PARAMETERS"
for i in $(awk '/^Av[ae]rage fmeasure.*[0-9]$/ {print $3}' hyperoptimization.txt | sort | tail -n$1)
do
	grep -B53 $i hyperoptimization.txt | head -n1 $output
	grep -B53 $i hyperoptimization.txt | tail -n1 $output
	echo "______________"
done
echo
echo "Total number of evaluations:               $(grep TESTING hyperoptimization.txt | wc -l)"
echo "Total number of evaluations in the domain: $(grep "Av[ae]rage fmeasure" hyperoptimization.txt | wc -l)"
