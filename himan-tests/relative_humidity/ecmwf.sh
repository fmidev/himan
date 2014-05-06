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

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

rm -f RH-PRCNT_height_2_rll_233_231_0_006.grib

  $HIMAN -s cuda -d 5 -f relative_humidity_ecmwf.json -t grib source_ecmwf.grib

  grib_compare -b referenceValue -A 0.01 RH-PRCNT_height_2_rll_233_231_0_006.grib result_ecmwf.grib 

  if [ $? -eq 0 ];then
    echo relative_humidity/ecmwf success on GPU!
  else
    echo relative_humidity/ecmwf failed on GPU
  fi
else
  echo "no cuda device found for cuda tests"
fi
