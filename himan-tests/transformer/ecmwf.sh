#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f tk2tc_ec.json.grib tk2tc_ec.json-CPU.grib

$HIMAN -d 5 -f tk2tc_ec.json -t grib --no-cuda -s tk2tc_ec_nocuda ec_source.grib

grib_compare -A 0.0001 ec_result.grib tk2tc_ec.json.grib


if [ $? -eq 0 ];then
  echo tk2tc/ec success!
else
  echo tk2tc/ec failed
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  mv tk2tc_ec.json.grib tk2tc_ec.json-CPU.grib

  $HIMAN -d 5 -f tk2tc_ec.json -t grib -s tk2tc_ec_cuda ec_source.grib

  grib_compare -A 0.0001 tk2tc_ec.json.grib tk2tc_ec.json-CPU.grib

  if [ $? -eq 0 ];then
    echo tk2tc/ec success on GPU!
  else
    echo tk2tc/ec failed on GPU
    exit 1
  fi
else
  echo "no cuda device found for cuda tests"
fi
