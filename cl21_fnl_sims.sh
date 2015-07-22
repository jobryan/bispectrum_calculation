#!/bin/bash
for i in {1..10}
do
	mpirun -np 15 python cl21.py fnl $i 0
	#echo $i
done
for i in {1..10}
do
	mpirun -np 15 python cl21.py fnl $i 100
	#echo $i
done
