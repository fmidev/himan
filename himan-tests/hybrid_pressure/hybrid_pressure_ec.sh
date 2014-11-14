#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f hybrid_pressure_ec.json.grib

$HIMAN -d 5 -f hybrid_pressure_ec.json -t grib lnsp_ec.grib2 q_ec.grib2

grib_compare result_ec.grib hybrid_pressure_ec.json.grib

if [ $? -eq 0 ];then
  echo hybrid_pressure/ec success!
else
  echo hybrid_pressure/ec failed
  exit 1
fi

