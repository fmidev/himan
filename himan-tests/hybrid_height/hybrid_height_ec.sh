#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f HL-M_*

$HIMAN -d 5 -f hybrid_height_ec.json -t grib hl_source_geop_ec.grib

grib_compare result_ec.grib HL-M_hybrid_64_polster_290_225_0_003.grib

if [ $? -eq 0 ];then
  echo hybrid_height/ec success!
else
  echo hybrid_height/ec failed
  exit 1
fi
