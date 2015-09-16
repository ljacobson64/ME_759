#!/bin/bash

rm -f log
for i in $(seq -f "%02g" 1 28); do
  ../../bld/p3 ./zinp_$i 2>&1 | tee -a log
done
