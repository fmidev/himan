#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f HL-M_*

$HIMAN -d 5 -f hybrid_height.json -t grib hl_source_geop.grib

grib_compare result.grib HL-M_hybrid_64_polster_290_225_0_003.grib

if [ $? -eq 0 ];then
  echo hybrid_height/hirlam success!
else
  echo hybrid_height/hirlam failed
  exit 1
fi
