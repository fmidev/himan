#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f RR*grib

$HIMAN -d 5 -f precipitation_hl.json -t grib hl_source.grib --no-cuda -s hl_nocuda

grib_compare hl_result_24.grib RR-6-MM_height_0_rll_1030_816_0_024.grib

if [ $? -ne 0 ];then
  echo precipitation/hirlam failed on CPU
  exit 1
fi

grib_compare hl_result_25.grib RR-6-MM_height_0_rll_1030_816_0_025.grib

if [ $? -eq 0 ];then
  echo precipitation/hirlam success on CPU!
else
  echo precipitation/hirlam failed on CPU
  exit 1
fi
