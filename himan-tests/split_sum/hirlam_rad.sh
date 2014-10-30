#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f RR*grib

$HIMAN -d 5 -f radiation_hl.json -t grib hl_radiation_source.grib --no-cuda -s hl_nocuda

grib_compare hl_radiation_result.grib ./RTOPLW-WM2_height_0_rll_161_177_0_089.grib

if [ $? -ne 0 ];then
  echo radiation/hirlam failed on CPU
  exit 1
fi

echo radiation/hirlam success on CPU

