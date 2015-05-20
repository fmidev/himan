#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/build/debug/himan"
fi

rm -f CL*.grib

$HIMAN -d 5 -f ceiling_ecmwf.json -t grib --no-cuda ecmwf_source.grib

grib_compare -A 0.001 CL-FT_ground_0_rll_529_461_0_001.grib ecmwf_result.grib
if [ $? -eq 0 ];then
  echo ceiling/ecmwf success on CPU!
else
  echo ceiling/ecmwf failed on CPU
  exit 1
fi

rm -f CL*.grib
