#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f dewpoint_arome.json.grib dewpoint_arome.json-CPU.grib

$HIMAN -d 5 -f dewpoint_arome.json -t grib source.grib --no-cuda

grib_compare result.grib dewpoint_arome.json.grib

if [ $? -eq 0 ];then
  echo dewpoint/arome success on CPU!
else
  echo dewpoint/arome failed on CPU
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  mv dewpoint_arome.json.grib dewpoint_arome.json-CPU.grib

  $HIMAN -d 5 -f dewpoint_arome.json -t grib source.grib

  grib_compare -A 0.001 dewpoint_arome.json.grib dewpoint_arome.json-CPU.grib

  if [ $? -eq 0 ];then
    echo dewpoint/arome success on GPU!
  else
    echo dewpoint/arome failed on GPU
  fi
else
  echo "no cuda device found for cuda tests"
fi

