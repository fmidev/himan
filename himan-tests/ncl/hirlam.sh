#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f H0C-M_*

$HIMAN -d 5 -f hirlam.json -t grib height_himan.grib t.grib

grib_compare result.grib H0C-M_height_0_rll_1030_816_0_001.grib

if [ $? -eq 0 ];then
  echo ncl/hirlam success!
else
  echo ncl/hirlam failed
  exit 1
fi
