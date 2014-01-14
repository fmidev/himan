#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f RH-PRCNT_hybrid_65_rll_1030_816_0_00*.grib P-HPA*grib

$HIMAN -d 5 -f relative_humidity_hirlam.json -t grib --no-cuda source_hirlam.grib

grib_compare RH-PRCNT_hybrid_65_rll_1030_816_0_004.grib result_hirlam_4.grib 

if [ $? -ne 0 ];then
  echo relative humidity/hirlam failed on CPU
  exit 1
fi

grib_compare RH-PRCNT_hybrid_65_rll_1030_816_0_005.grib result_hirlam_5.grib 

if [ $? -eq 0 ];then
  echo relative humidity/hirlam success on CPU!
else
  echo relative humidity/hirlam failed on CPU
  exit 1
fi

exit 0

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  mv relative humidity_hirlam.json.grib relative humidity_hirlam.json-CPU.grib

  $HIMAN -d 5 -f relative humidity_hirlam.json -t grib source.grib

#  grib_compare -A 0.001 relative humidity_hirlam.json.grib relative humidity_hirlam.json-CPU.grib

  if [ $? -eq 0 ];then
    echo relative humidity/hirlam success on GPU!
  else
    echo relative humidity/hirlam failed on GPU
  fi
else
  echo "no cuda device found for cuda tests"
fi

