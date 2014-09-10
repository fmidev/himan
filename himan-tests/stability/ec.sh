#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f *-N_height*001.grib

$HIMAN -d 5 -f stability_ec.json -s stat source_ec.grib --no-cuda

grib_compare result_ec.grib ./KINDEX-N_height_0_rll_161_177_0_001.grib

if [ $? -eq 0 ];then
  echo kindex/ec success!
else
  echo kindex/ec failed
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  rm -f *-N_height*001.grib

  $HIMAN -d 5 -f stability_ec.json -s stat source_ec.grib 

  grib_compare result_ec.grib ./KINDEX-N_height_0_rll_161_177_0_001.grib
  
  if [ $? -eq 0 ];then
    echo stability/ec success on GPU!
  else
    echo stability/ec failed on GPU
    exit 1
  fi
else
  echo "no cuda device found for cuda tests"
fi

