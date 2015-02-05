#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

$HIMAN -d 5 -j 1 -f turbulence_ec.json -t grib --no-cuda 

if [ $? -eq 0 ];then
  echo turbulence/ecmwf success on CPU!
else
  echo turbulence/ecmwf failed on CPU
  exit 1
fi

