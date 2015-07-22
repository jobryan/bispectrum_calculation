#!/bin/bash
for i in {1..2}
do
	python map_fnl_sims.py $i 30
	#echo $i
done
#for i in {1..2}
#do
#	python map_fnl_sims.py $i 10
#	#echo $i
#done
