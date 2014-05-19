#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f PREC*grib

$HIMAN -d 5 -f preform_hirlam.json --no-cuda -s hirlam-stat -t grib  hirlam-source.grib

grib_compare hirlam-result.grib PRECFORM-N_height_0_rll_1030_816_0_004.grib

if [ $? -ne 0 ];then
  echo preform_pressure/hirlam failed on CPU
  exit 1
else
  echo preform_pressure/hirlam success on CPU
  exit 0
fi

