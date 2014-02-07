#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f TP-K*.grib TPW-K*.grib

$HIMAN -d 5 -f tpot_hl.json -t grib source_hl.grib --no-cuda -s stat

grib_compare result_hl_theta_4.grib ./TP-K_pressure_850_rll_1030_816_0_004.grib

if [ $? -ne 0 ];then
  echo tpot/hl theta failed on CPU
  exit 1
fi

grib_compare result_hl_theta_5.grib ./TP-K_pressure_850_rll_1030_816_0_005.grib

if [ $? -eq 0 ];then
  echo tpot/hl theta success on CPU!
else
  echo tpot/hl theta failed on CPU
  exit 1
fi

echo "thetaw results are not complete yet"
exit 0

grib_compare result_hl_thetaw_4.grib ./TPW-K_pressure_850_rll_1030_816_0_004.grib

if [ $? -ne 0 ];then
  echo tpot/hl thetaw failed on CPU
  exit 1
fi

grib_compare result_hl_thetaw_5.grib ./TPW-K_pressure_850_rll_1030_816_0_005.grib

if [ $? -eq 0 ];then
  echo tpot/hl thetaw success on CPU!
else
  echo tpot/hl thetaw failed on CPU
  exit 1
fi

exit 0

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  mv ./TPW-K_pressure_850_rll_582_448_0_043.grib ./TPW-K_pressure_850_rll_582_448_0_043-CPU.grib
  mv ./TPW-K_pressure_700_rll_582_448_0_043.grib ./TPW-K_pressure_700_rll_582_448_0_043-CPU.grib

  $HIMAN -d 5 -f tpot_hl.json -t grib source.grib

  grib_compare hl_result_700.grib ./TPW-K_pressure_850_rll_582_448_0_043.grib

  if [ $? -ne 0 ];then
    echo tpot/hl failed on GPU
    exit 1
  fi

  grib_compare hl_result_850.grib ./TPW-K_pressure_850_rll_582_448_0_043-CPU.grib

  if [ $? -eq 0 ];then
    echo tpot/hl success on GPU!
  else
    echo tpot/hl failed on GPU
    exit 1
  fi
else
  echo "no cuda device found for cuda tests"
fi

