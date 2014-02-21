#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f TPE-K*.grib TP-K*.grib TPW-K*.grib

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

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then
  
  rm -f TP-K*.grib TPW-K*.grib

  $HIMAN -d 5 -f tpot_hl.json -s cuda source_hl.grib

  grib_compare -A 0.001 result_hl_theta_4.grib ./TP-K_pressure_850_rll_1030_816_0_004.grib
  
  if [ $? -ne 0 ];then
    echo tpot/hl failed on GPU
    exit 1
  fi

  grib_compare -A 0.001 result_hl_theta_5.grib ./TP-K_pressure_850_rll_1030_816_0_005.grib

  if [ $? -eq 0 ];then
    echo tpot/hl success on GPU!
  else
    echo tpot/hl failed on GPU
    exit 1
  fi

else
  echo "no cuda device found for cuda tests"
fi

# THETA W

$HIMAN -d 5 -f tpot_hl_thetaw.json -t grib source_hl.grib --no-cuda -s stat

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

# THETA E

$HIMAN -d 5 -f tpot_hl_thetae.json -t grib source_hl_thetae.grib --no-cuda -s stat

grib_compare result_hl_thetae.grib ./TPE-K_pressure_850_rll_1030_816_0_005.grib

if [ $? -ne 0 ];then
  echo tpot/hl thetae failed on CPU
  exit 1
fi

grib_compare result_hl_thetae.grib ./TPE-K_pressure_850_rll_1030_816_0_005.grib

if [ $? -eq 0 ];then
  echo tpot/hl thetae success on CPU!
else
  echo tpot/hl thetae failed on CPU
  exit 1
fi


