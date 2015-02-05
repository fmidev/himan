#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

$HIMAN -d 5 -j 1 -f turbulence_hir.json -t grib --no-cuda 

if [ $? -eq 0 ];then
  echo turbulence/hirlam success on CPU!
else
  echo turbulence/hirlam failed on CPU
  exit 1
fi

