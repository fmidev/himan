#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f himan*.grib

$HIMAN -d 5 -f tk2tc_ec1h.json -t grib --no-cuda source.grib

grib_compare result.grib himan_2013011800.grib

if [ $? -eq 0 ];then
  echo tk2tc/ec success!
else
  echo tk2tc/ec failed
  exit 1
fi

if [ "$CUDA_TOOLKIT_PATH" != "" ]; then

  mv himan_2013011800.grib himan_2013011800-CPU.grib

  $HIMAN -d 5 -f tk2tc_ec1h.json -t grib source.grib

  grib_compare -A 0.0001 himan_2013011800.grib himan_2013011800-CPU.grib

  if [ $? -eq 0 ];then
    echo tk2tc/ec success on GPU!
  else
    echo tk2tc/ec failed on GPU
    exit 1
  fi
fi

