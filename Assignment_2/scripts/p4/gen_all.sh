#!/bin/bash

rm -f zinp_*
for i in $(seq -f "%02g" 1 28); do
  python gen_rands.py $i
  echo $i
done
