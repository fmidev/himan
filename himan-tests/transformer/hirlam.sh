#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f T-C_*.grib

$HIMAN -d 5 -f tk2tc_hl.json -t grib --no-cuda -s tk2tc_hl_nocuda hl_source.grib

grib_compare -A 0.0001 hl_result.grib ./T-C_height_2_rll_1030_816_0_001.grib

if [ $? -eq 0 ];then
  echo tk2tc/hl success!
else
  echo tk2tc/hl failed
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  rm -f T-C_*.grib

  $HIMAN -d 5 -f tk2tc_hl.json -t grib -s tk2tc_hl_cuda hl_source.grib

  grib_compare -A 0.0001 hl_result.grib ./T-C_height_2_rll_1030_816_0_001.grib

  if [ $? -eq 0 ];then
    echo tk2tc/hl success on GPU!
  else
    echo tk2tc/hl failed on GPU
    exit 1
  fi
else
  echo "no cuda device found for cuda tests"
fi

$HIMAN -d 5 -f tk2tc_hl_pres.json --no-cuda -s tk2tc_hl_pres_nocuda hl_source_pres.grib

grib_compare -A 0.0001 hl_result_pres.grib ./T-C_pressure_850_rll_582_448_0_001.grib

if [ $? -eq 0 ];then
  echo tk2tc_pres/hl success!
else
  echo tk2tc_pres/hl failed
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  rm -f T-C_*.grib

  $HIMAN -d 5 -f tk2tc_hl_pres.json -s tk2tc_hl_pres_cuda hl_source_pres.grib

  grib_compare -A 0.1 hl_result_pres.grib ./T-C_pressure_850_rll_582_448_0_001.grib

  if [ $? -eq 0 ];then
    echo tk2tc_pres/hl success on GPU!
  else
    echo tk2tc_pres/hl failed on GPU
    exit 1
  fi
else
  echo "no cuda device found for cuda tests"
fi


