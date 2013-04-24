#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f DD-D*.grib FF-MS*.grib DF-MS*.grib

$HIMAN -d 5 -f windvector_hl_regular.json -t grib hl_regular_source.grib -s hl_reg_nocuda --no-cuda

grib_compare -A 0.01 hl_regular_result_FF.grib ./FF-MS_height_10_rll_1030_816_0_006.grib

if [ $? -eq 0 ];then
  echo windvector/hirlam regular grid wind speed success on CPU
else
  echo windvector/hirlam regular grid wind speed failed on CPU
  exit 1
fi

grib_compare -A 0.01 hl_regular_result_DD.grib ./DD-D_height_10_rll_1030_816_0_006.grib

if [ $? -eq 0 ];then
  echo windvector/hirlam regular grid wind direction success!
else
  echo windvector/hirlam regular grid wind direction failed
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  mv ./FF-MS_height_10_rll_1030_816_0_006.grib ./FF-MS_height_10_rll_1030_816_0_006-CPU.grib
  mv ./DD-D_height_10_rll_1030_816_0_006.grib ./DD-D_height_10_rll_1030_816_0_006-CPU.grib

  $HIMAN -d 5 -f windvector_hl_regular.json -t grib -s hl_reg_cuda hl_regular_source.grib

  grib_compare -b referenceValue -A 0.001 hl_regular_result_FF.grib ./FF-MS_height_10_rll_1030_816_0_006.grib

  if [ $? -eq 0 ];then
    echo windvector/hirlam regular grid wind speed success GPU
  else
    echo windvector/hirlam regular grid wind speed failed GPU
    exit 1
  fi

  grib_compare -A 1 hl_regular_result_DD.grib ./DD-D_height_10_rll_1030_816_0_006.grib

  if [ $? -eq 0 ];then
    echo windvector/hirlam regular grid wind direction success GPU
  else
    echo windvector/hirlam regular grid wind direction failed GPU
    exit 1
fi
fi

$HIMAN -d 5 -f windvector_hl_staggered.json -t grib hl_staggered_source.grib --no-cuda -s hl_stag_nocuda

grib_compare -A 0.01 hl_staggered_result_FF.grib ./FF-MS_hybrid_55_polster_290_225_0_025.grib

if [ $? -eq 0 ];then
  echo windvector/hirlam staggered grid wind speed success!
else
  echo windvector/hirlam staggered grid wind speed failed
#  exit 1
fi

grib_compare -A 0.01 hl_staggered_result_DD.grib ./DD-D_hybrid_55_polster_290_225_0_025.grib

if [ $? -eq 0 ];then
  echo windvector/hirlam staggered grid wind direction success!
else
  echo windvector/hirlam staggered grid wind direction failed
#  exit 1
fi
