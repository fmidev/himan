#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f TD-C*grib

$HIMAN -s no-cuda -d 5 -f dewpoint_harmonie.json -t grib source.grib --no-cuda

grib_compare harmonie_result_height.grib TD-C_height_2_rll_720_800_0_1440.grib

if [ $? -ne 0 ];then
  echo dewpoint/harmonie failed on CPU
  exit 1
fi

grib_compare harmonie_result_pressure.grib TD-C_pressure_700_rll_720_800_0_1440.grib

if [ $? -eq 0 ];then
  echo dewpoint/harmonie success on CPU!
else
  echo dewpoint/harmonie failed on CPU
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  rm -f TD-C_*.grib

  $HIMAN -s cuda -d 5 -f dewpoint_harmonie.json -t grib source.grib

  grib_compare -A 0.001 harmonie_result_height.grib TD-C_height_2_rll_720_800_0_1440.grib

  if [ $? -ne 0 ];then
    echo dewpoint/harmonie failed on GPU
  fi

  grib_compare -A 0.001 harmonie_result_pressure.grib TD-C_pressure_700_rll_720_800_0_1440.grib

  if [ $? -eq 0 ];then
    echo dewpoint/harmonie success on GPU!
  else
    echo dewpoint/harmonie failed on GPU
  fi

else
  echo "no cuda device found for cuda tests"
fi

