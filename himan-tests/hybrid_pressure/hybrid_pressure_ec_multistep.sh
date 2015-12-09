#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f ecmwf-multistep.json.grib

$HIMAN -N -d 4 -f ecmwf-multistep.json lnsp_ec.grib2 q_ec.grib2 --no-cuda source_ecmwf_multistep.grib

grib_compare -A 0.001 result_ec_multistep.grib ecmwf-multistep.json.grib

if [ $? -eq 0 ];then
  echo hybrid_pressure_multistep/ec success!
else
  echo hybrid_pressure_multistep/ec failed
  exit 1
fi

