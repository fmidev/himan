#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/build/release/himan"
fi

rm -f SR-M*.grib

$HIMAN -d 5 -f roughness_hirlam.json -t grib --no-cuda hirlam_SR_source.grib hirlam_SRMOM_source.grib

grib_compare ./SR-M_height_0_rll_1030_816_0_005.grib result_hirlam.grib 

if [ $? -eq 0 ];then
  echo roughness/hirlam success on CPU!
else
  echo roughness/hirlam failed on CPU
  exit 1
fi

rm -f SR-M*.grib
