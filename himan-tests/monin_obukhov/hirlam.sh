#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f MOL-M*.grib

$HIMAN -d 5 -j 1 -f monin_obukhov_hirlam.json -t grib --no-cuda FLSEN-JM2_source_1.grib FLSEN-JM2_source_2.grib FRVEL-MS_source.grib P-PA_source.grib T-K_source.grib

grib_compare MOL-M_height_0_rll_1030_816_0_015.grib result.grib 

if [ $? -eq 0 ];then
  echo monin-obukhov/hirlam success on CPU!
else
  echo monin-obukhov/hirlam failed on CPU
  exit 1
fi

rm -f MOL-M*.grib
