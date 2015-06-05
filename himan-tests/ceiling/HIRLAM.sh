#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/build/debug/himan"
fi

rm -f CL*.grib

$HIMAN -d 5 -f ceiling_hirlam.json -t grib --no-cuda hirlam_source.grib

grib_compare -A 0.001 CL-FT_ground_0_rll_1030_816_0_001.grib hirlam_result.grib
if [ $? -eq 0 ];then
  echo ceiling/hirlam success on CPU!
else
  echo ceiling/hirlam failed on CPU
  exit 1
fi

rm -f CL*.grib