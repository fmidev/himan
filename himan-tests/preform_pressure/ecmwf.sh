#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f PREC*grib

$HIMAN -d 5 -f preform_ec.json --no-cuda -s ec-stat -t grib ecmwf-source.grib

grib_compare ecmwf-result.grib PRECFORM-N_height_0_ll_2880_1441_0_004.grib

if [ $? -ne 0 ];then
  echo preform_pressure/ec failed on CPU
  exit 1
else
  echo preform_pressure/ec success on CPU
  exit 0
fi

