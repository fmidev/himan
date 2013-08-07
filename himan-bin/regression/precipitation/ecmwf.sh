#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f RR*grib

$HIMAN -d 5 -f precipitation_ec.json -t grib ec_source.grib --no-cuda -s ec_nocuda

grib_compare ec_result_rr1h_4.grib ./RR-1-MM_ground_0_rll_161_177_0_004.grib

if [ $? -ne 0 ];then
  echo precipitation/ecmwf failed on CPU
  exit 1
fi

grib_compare ec_result_rr1h_5.grib ./RR-1-MM_ground_0_rll_161_177_0_005.grib

if [ $? -ne 0 ];then
  echo precipitation/ecmwf failed on CPU
  exit 1
fi

grib_compare ec_result_rr1h_6.grib ./RR-1-MM_ground_0_rll_161_177_0_006.grib

if [ $? -eq 0 ];then
  echo precipitation/ecmwf success on CPU!
else
  echo precipitation/ecmwf failed on CPU
  exit 1
fi

