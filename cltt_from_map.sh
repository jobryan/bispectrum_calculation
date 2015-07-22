#!/bin/bash
for i in {1..11}
do
    python cltt_from_map.py sim $i
done
python cltt_from_map.py data 0
