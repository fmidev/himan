#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f FRVEL*.grib RHO*.grib

$HIMAN -d 5 -f friction_velocity_ecmwf.json -t grib --no-cuda NSSS-NM2S_GROUND_0_0_ll_2880_1441_0_001_1.grib EWSS-NM2S_GROUND_0_0_ll_2880_1441_0_001_1.grib T-K_GROUND_0_0_ll_2880_1441_0_001_1.grib P-PA_GROUND_0_0_ll_2880_1441_0_001_1.grib

grib_compare -A 0.001 FRVEL-MS_ground_0_ll_2880_1441_0_001.grib result.grib
if [ $? -eq 0 ];then
  echo friction_velocity/ecmwf success on CPU!
else
  echo friction_velocity/ecmwf failed on CPU
  exit 1
fi

rm -f FRVEL*.grib RHO*.grib
