#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f DD-D*.grib FF-MS*.grib 

$HIMAN -d 5 -f windvector_hl_unstagger.json -t grib hl_staggered_plugin_source.grib -s hl_reg_nocuda --no-cuda

grib_compare -A 0.01 hl_staggered_plugin_result_ff.grib ./FF-MS_hybrid_20_rll_1030_816_0_004.grib

if [ $? -eq 0 ];then
  echo windvector/hirlam staggered grid wind speed success on CPU
else
  echo windvector/hirlam staggered grid wind speed failed on CPU
  exit 1
fi

grib_compare -A 1 hl_staggered_plugin_result_dd.grib ./DD-D_hybrid_20_rll_1030_816_0_004.grib

if [ $? -eq 0 ];then
  echo windvector/hirlam staggered grid wind direction success!
else
  echo windvector/hirlam staggered grid wind direction failed
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  rm -f DD-D*.grib FF-MS*.grib 

  $HIMAN -d 5 -f windvector_hl_unstagger.json -t grib -s hl_reg_cuda hl_staggered_plugin_source.grib

  grib_compare -b referenceValue -A 0.001 hl_staggered_plugin_result_ff.grib ./FF-MS_hybrid_20_rll_1030_816_0_004.grib

  if [ $? -eq 0 ];then
    echo windvector/hirlam staggered grid wind speed success GPU
  else
    echo windvector/hirlam staggered grid wind speed failed GPU
    exit 1
  fi

  grib_compare -A 1 hl_staggered_plugin_result_dd.grib ./DD-D_hybrid_20_rll_1030_816_0_004.grib

  if [ $? -eq 0 ];then
    echo windvector/hirlam staggered grid wind direction success GPU
  else
    echo windvector/hirlam staggered grid wind direction failed GPU
    exit 1
  fi
fi

