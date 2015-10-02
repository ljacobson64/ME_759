#!/bin/bash

rm -f log
for i in $(seq -f "%02g" 1 20); do
  ../bld/p2 $i 32 2>&1 | tee -a log
done
for i in $(seq -f "%02g" 1 25); do
  ../bld/p2 $i 1024 2>&1 | tee -a log
done
