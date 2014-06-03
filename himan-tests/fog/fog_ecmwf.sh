#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f fog_ecmwf.json.grib fog_ecmwf.json-CPU.grib

$HIMAN -f fog_ecmwf.json -t grib -a ec_ff10m.grib ec_ground.grib ec_dew.grib --no-cuda

grib_compare result_ecmwf.grib fog_ecmwf.json.grib

if [ $? -eq 0 ];then
  echo fog success on CPU!
else
  echo fog failed on CPU
  exit 1
fi

exit 0
# if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

#   mv fog.json.grib fog.json-CPU.grib

#   $HIMAN -f fog.json -t grib ff10m.grib ground.grib dew.grib

#   grib_compare -A 0.001 fog.json.grib fog.json-CPU.grib

#   if [ $? -eq 0 ];then
#     echo fog success on GPU!
#   else
#     echo fog failed on GPU
#   fi
# else
#   echo "no cuda device found for cuda tests"
# fi

