#!/bin/bash

if [ "x$1" == "x" ]
then
  echo "Missing index argument 1-3"
  exit 1
fi

gcc -o mmult mmult.c -lprofiler
./mmult $1
google-pprof --text mmult mmult.prof

exit 0
