#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/build/debug/himan"
fi

rm -f QNH*.grib

$HIMAN -d 5 -f qnh_ec.json -t grib --no-cuda ec_source.grib

grib_compare -A 0.001 ./QNH-HPA_height_0_ll_2880_1441_0_013.grib ec_result.grib
if [ $? -eq 0 ];then
  echo qnh/ecmwf success on CPU!
else
  echo qnh/ecmwf failed on CPU
  exit 1
fi

rm -f QNH*.grib
