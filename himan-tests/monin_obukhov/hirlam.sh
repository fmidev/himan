#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f monin_obukhov_hirlam.json.grib

$HIMAN -d 5 -j 1 -f monin_obukhov_hirlam.json --no-cuda 1_105_0_0_rll_1030_816_0_0_012.grib 121_105_0_0_rll_1030_816_4_0_011.grib 122_105_0_0_rll_1030_816_4_0_011.grib 227_105_0_0_rll_1030_816_0_0_012.grib 11_105_0_0_rll_1030_816_0_0_012.grib 121_105_0_0_rll_1030_816_4_0_012.grib 122_105_0_0_rll_1030_816_4_0_012.grib

grib_compare monin_obukhov_hirlam.json.grib result.grib 

if [ $? -eq 0 ];then
  echo monin-obukhov/hirlam success on CPU!
else
  echo monin-obukhov/hirlam failed on CPU
  exit 1
fi

rm -f monin_obukhov_hirlam.json.grib
