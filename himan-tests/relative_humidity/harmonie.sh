#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f RH-PRCNT*.grib 

$HIMAN -d 5 -f relative_humidity_harmonie.json -t grib --no-cuda source_harmonie.grib -s stat

grib_compare ./RH-PRCNT_hybrid_44_rll_720_800_0_060.grib result_harmonie.grib 

if [ $? -eq 0 ];then
  echo relative humidity/harmonie success on CPU!
else
  echo relative humidity/harmonie failed on CPU
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  rm -f RH-PRCNT*.grib

  $HIMAN -s cuda -d 5 -f relative_humidity_harmonie.json -t grib source_harmonie.grib 

  grib_compare -A 0.01 ./RH-PRCNT_hybrid_44_rll_720_800_0_060.grib result_harmonie.grib 

  if [ $? -eq 0 ];then
    echo relative humidity/harmonie success on GPU!
  else
    echo relative humidity/harmonie failed on GPU
  exit 1
fi

else
  echo "no cuda device found for cuda tests"
fi
