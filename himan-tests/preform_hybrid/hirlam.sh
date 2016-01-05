#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f PREC*grib

source_data=hirlam-source.grib

if [ ! -f "$source_data" ]; then
  echo "source data not present, copying it"
  scp "weto@scout:/masala/src/routines/run/himan/test-data/$source_data" .

  if [ $? -ne 0 ]; then
    exit 1
  fi
fi 

$HIMAN -d 5 -f hirlam-preform.json --no-cuda -s hirlam-stat $source_data

grib_compare hirlam-result.grib PRECFORM2-N_height_0_rll_1030_816_0_015.grib

if [ $? -ne 0 ];then
  echo preform_hybrid/hirlam failed on CPU
  exit 1
else
  echo preform_hybrid/hirlam success on CPU
  exit 0
fi

