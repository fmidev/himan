#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f DPDEF*grib

$HIMAN -d 5 -f ecmwf.json ecmwf_dpdef_source.grib -s stat --no-cuda

grib_compare ecmwf_dpdef_result.grib ./DPDEF-C_height_2_ll_2880_1441_0_001.grib

if [ $? -eq 0 ];then
  echo luatool/ecmwf success on CPU!
else
  echo luatool/ecmwf failed on CPU
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  rm -f DPDEF*grib

  $HIMAN -s stat -d 5 -f ecmwf.json ecmwf_dpdef_source.grib

  grib_compare -A 0.001 ecmwf_dpdef_result.grib ./DPDEF-C_height_2_ll_2880_1441_0_001.grib

  if [ $? -eq 0 ];then
    echo luatool/ecmwf success on GPU!
  else
    echo luatool/ecmwf failed on GPU
  fi
else
  echo "no cuda device found for cuda tests"
fi

