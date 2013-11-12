#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f RR*grib

$HIMAN -d 5 -f precipitation_harmonie.json -t grib harmonie_source.grib --no-cuda -s harmonie_nocuda

grib_compare harmonie_result_1_60.grib RR-1-MM_height_0_rll_290_594_0_060.grib

if [ $? -ne 0 ];then
  echo precipitation/harmonie failed on CPU
  exit 1
fi

grib_compare harmonie_result_1_120.grib RR-1-MM_height_0_rll_290_594_0_120.grib

if [ $? -ne 0 ];then
  echo precipitation/harmonie failed on CPU
  exit 1
fi

grib_compare harmonie_result_1_180.grib RR-1-MM_height_0_rll_290_594_0_180.grib

if [ $? -ne 0 ];then
  echo precipitation/harmonie failed on CPU
  exit 1
fi

grib_compare harmonie_result_3_180.grib RR-3-MM_height_0_rll_290_594_0_180.grib

if [ $? -eq 0 ];then
  echo precipitation/harmonie success on CPU!
else
  echo precipitation/harmonie failed on CPU
  exit 1
fi


