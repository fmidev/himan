#!/bin/sh
exit 0
set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f TPW-K*.grib

$HIMAN -d 5 -f tpot_ec.json -t grib ec_source.grib --no-cuda -s ec_nocuda

grib_compare ec_result_700.grib TPW-K_pressure_700_rll_161_177_0_003.grib

if [ $? -ne 0 ];then
  echo tpot/ec failed on CPU
  exit 1
fi

grib_compare ec_result_850.grib TPW-K_pressure_700_rll_161_177_0_003.grib

if [ $? -eq 0 ];then
  echo tpot/ec success on CPU!
else
  echo tpot/ec failed on CPU
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  rm -f TPW-K*.grib

  $HIMAN -d 5 -f tpot_ec.json -t grib source.grib -s ec_cuda

  grib_compare ec_result_700.grib ./TPW-K_pressure_850_rll_161_177_0_003.grib

  if [ $? -ne 0 ];then
    echo tpot/hl failed on GPU
    exit 1
  fi

  grib_compare ec_result_850.grib ./TPW-K_pressure_850_rll_161_177_0_003-CPU.grib

  if [ $? -eq 0 ];then
    echo tpot/ec success on GPU!
  else
    echo tpot/ec failed on GPU
    exit 1
  fi
else
  echo "no cuda device found for cuda tests"
fi

