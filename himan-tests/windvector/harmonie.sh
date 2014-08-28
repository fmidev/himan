#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f FFG-MS*.grib

$HIMAN -d 5 -f windvector_harmonie.json -t grib harmonie_gust_source.grib -s harmonie_nocuda --no-cuda

grib_compare -A 0.01 harmonie_gust_result.grib ./FFG-MS_height_10_rll_720_800_0_360.grib

if [ $? -eq 0 ];then
  echo windvector/harmonie gust speed success on CPU
else
  echo windvector/harmonie gust speed failed on CPU
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  rm -f FFG-MS*.grib 

  $HIMAN -d 5 -f windvector_harmonie.json -t grib -s harmonie_cuda harmonie_gust_source.grib

  grib_compare -b referenceValue -A 0.001 harmonie_gust_result.grib ./FFG-MS_height_10_rll_720_800_0_360.grib

  if [ $? -eq 0 ];then
    echo windvector/harmonie gust speed success GPU
  else
    echo windvector/harmonie gust speed failed GPU
    exit 1
  fi

fi

rm -f FF-MS*grib DD-D*grib

$HIMAN -d 5 -f windvector_harmonie_wind.json -t grib harmonie_wind_source.grib -s harmonie_nocuda --no-cuda

grib_compare -A 0.01 harmonie_wind_result_FF.grib ./FF-MS_hybrid_15_rll_720_800_0_120.grib

if [ $? -eq 0 ];then
  echo windvector/harmonie wind speed success on CPU
else
  echo windvector/harmonie wind speed failed on CPU
  exit 1
fi

grib_compare -A 1 harmonie_wind_result_DD.grib ./DD-D_hybrid_15_rll_720_800_0_120.grib

if [ $? -eq 0 ];then
  echo windvector/harmonie wind direction success on CPU
else
  echo windvector/harmonie wind direction failed on CPU
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  rm -f FF-MS*.grib DD-D*grib

  $HIMAN -d 5 -f windvector_harmonie_wind.json -t grib -s harmonie_cuda harmonie_wind_source.grib

  grib_compare -b referenceValue -A 0.001 harmonie_wind_result_FF.grib ./FF-MS_hybrid_15_rll_720_800_0_120.grib

  if [ $? -eq 0 ];then
    echo windvector/harmonie wind speed success GPU
  else
    echo windvector/harmonie wind speed failed GPU
    exit 1
  fi

  grib_compare -A 1 harmonie_wind_result_DD.grib ./DD-D_hybrid_15_rll_720_800_0_120.grib

  if [ $? -eq 0 ];then
    echo windvector/harmonie wind direction success on GPU
  else
    echo windvector/harmonie wind direction failed on GPU
    exit 1
  fi

fi

