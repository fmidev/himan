#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f H0C-M_*

$HIMAN -d 5 -f ec.json -t grib ec_source.grib2

grib_compare -A 0.3 ec_result.grib ./H0C-M_height_0_rll_465_461_0_003.grib

if [ $? -eq 0 ];then
  echo ncl/ec success!
else
  echo ncl/ec failed
  exit 1
fi
