#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f tk2tc_hl.json*grib tk2tc_hl_pres.json*.grib

$HIMAN -d 5 -f tk2tc_hl.json -t grib --no-cuda -s tk2tc_hl_nocuda hl_source.grib

grib_compare hl_result.grib tk2tc_hl.json.grib

if [ $? -eq 0 ];then
  echo tk2tc/hl success!
else
  echo tk2tc/hl failed
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  mv tk2tc_hl.json.grib tk2tc_hl.json-CPU.grib

  $HIMAN -d 5 -f tk2tc_hl.json -t grib -s tk2tc_hl_cuda hl_source.grib

  grib_compare -A 0.0001 tk2tc_hl.json.grib tk2tc_hl.json-CPU.grib

  if [ $? -eq 0 ];then
    echo tk2tc/hl success on GPU!
  else
    echo tk2tc/hl failed on GPU
    exit 1
  fi
else
  echo "no cuda device found for cuda tests"
fi

$HIMAN -d 5 -f tk2tc_hl_pres.json -t grib --no-cuda -s tk2tc_hl_pres_nocuda hl_source_pres.grib

grib_compare hl_result_pres.grib tk2tc_hl_pres.json.grib

if [ $? -eq 0 ];then
  echo tk2tc_pres/hl success!
else
  echo tk2tc_pres/hl failed
  exit 1
fi

