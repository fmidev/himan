#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f TPW-K*.grib

$HIMAN -d 5 -f tpot_ec.json source_ec.grib --no-cuda -s ec_nocuda

grib_compare ec_result.grib TPW-K_pressure_850_rll_161_177_0_003.grib

if [ $? -eq 0 ];then
  echo tpot/ec success on CPU!
else
  echo tpot/ec failed on CPU
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  rm -f TPW-K*.grib

  $HIMAN -d 5 -f tpot_ec.json source_ec.grib -s ec_cuda

  grib_compare -A 0.01 ec_result.grib ./TPW-K_pressure_850_rll_161_177_0_003.grib

  if [ $? -eq 0 ];then
    echo tpot/ec success on GPU!
  else
    echo tpot/ec failed on GPU
    exit 1
  fi
else
  echo "no cuda device found for cuda tests"
fi

