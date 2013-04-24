#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f FFG-MS*.grib

$HIMAN -d 5 -f windvector_harmonie.json -t grib harmonie_gust_source.grib -s harmonie_nocuda --no-cuda

grib_compare -A 0.01 harmonie_gust_result.grib ./FFG-MS_height_10_rll_290_594_0_360.grib

if [ $? -eq 0 ];then
  echo windvector/harmonie gust speed success on CPU
else
  echo windvector/harmonie gust speed success on CPU
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  rm -f FFG-MS*.grib 

  $HIMAN -d 5 -f windvector_harmonie.json -t grib -s harmonie_cuda harmonie_gust_source.grib

  grib_compare -b referenceValue -A 0.001 harmonie_gust_result.grib ./FFG-MS_height_10_rll_290_594_0_360.grib

  if [ $? -eq 0 ];then
    echo windvector/harmonie gust speed success GPU
  else
    echo windvector/harmonie gust speed failed GPU
    exit 1
  fi

fi
