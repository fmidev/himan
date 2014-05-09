#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f TD-C_pressure_700_rll_529_461_0_009.grib

$HIMAN -s no-cuda -d 5 -f dewpoint_ecmwf.json -t grib ecmwf_source.grib --no-cuda

grib_compare ecmwf_result.grib TD-C_pressure_700_rll_529_461_0_009.grib

if [ $? -eq 0 ];then
  echo dewpoint/ecmwf success on CPU!
else
  echo dewpoint/ecmwf failed on CPU
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  rm -f TD-C_pressure_700_rll_529_461_0_009.grib

  $HIMAN -s cuda -d 5 -f dewpoint_ecmwf.json -t grib ecmwf_source.grib

  grib_compare -A 0.001 ecmwf_result.grib TD-C_pressure_700_rll_529_461_0_009.grib

  if [ $? -eq 0 ];then
    echo dewpoint/ecmwf success on GPU!
  else
    echo dewpoint/ecmwf failed on GPU
  fi
else
  echo "no cuda device found for cuda tests"
fi

