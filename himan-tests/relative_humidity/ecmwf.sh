#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f RH-PRCNT_height_2_rll_233_231_0_006.grib

$HIMAN -d 5 -f relative_humidity_ecmwf.json -t grib --no-cuda source_ecmwf.grib

grib_compare RH-PRCNT_height_2_rll_233_231_0_006.grib result_ecmwf.grib 

if [ $? -eq 0 ];then
  echo relative humidity/ecmwf success on CPU!
else
  echo relative humidity/ecmwf failed on CPU
  exit 1
fi

