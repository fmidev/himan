#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f FFG2-MS_height_10_rll_661_576_0_005.grib

$HIMAN -d 4 -j 1 -f gust_ecmwf.json -t grib --no-cuda source.grib source.grib2

grib_compare FFG2-MS_height_10_rll_661_576_0_005.grib result.grib

if [ $? -eq 0 ];then
  echo gust/ecmwf success on CPU!
else
  echo gust/ecmwf failed on CPU
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  rm -f FFG2-MS_height_10_rll_661_576_0_005.grib

  $HIMAN -d 4 -j 1 -f gust_ecmwf.json -t grib source.grib source.grib2

  grib_compare -A 0.5 FFG2-MS_height_10_rll_661_576_0_005.grib result.grib

  if [ $? -eq 0 ];then
    echo gust/ecmwf success on GPU!
  else
    echo gust/ecmwf failed on GPU
    exit 1
  fi

fi
