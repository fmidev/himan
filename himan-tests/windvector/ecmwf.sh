#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f DD-D*.grib FF-MS*.grib DF-MS*.grib

$HIMAN -d 5 -f windvector_ec.json -t grib windvector_ec_ground_source.grib windvector_ec_hybrid_source.grib --no-cuda -s ec_nocuda

grib_compare windvector_ec_ground_FF_6_result.grib FF-MS_height_10_rll_161_177_0_006.grib

if [ $? -ne 0 ];then
  echo windvector/ec wind speed failed on CPU
  exit 1
fi

grib_compare -A 0.01 windvector_ec_ground_FF_9_result.grib FF-MS_height_10_rll_161_177_0_009.grib

if [ $? -eq 0 ];then
  echo windvector/ec wind speed success on CPU!
else
  echo windvector/ec wind speed failed on CPU
  exit 1
fi

grib_compare -A 0.01 windvector_ec_ground_DD_6_result.grib DD-D_height_10_rll_161_177_0_006.grib

if [ $? -ne 0 ];then
  echo windvector/ec wind direction failed on CPU
  exit 1
fi

grib_compare -A 0.01 windvector_ec_ground_DD_9_result.grib DD-D_height_10_rll_161_177_0_009.grib

if [ $? -eq 0 ];then
  echo windvector/ec wind direction success on CPU!
else
  echo windvector/ec wind direction failed on CPU
  exit 1
fi

echo "windvector EC on GPU .... SOON"
exit 0

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  rm -f DD-D*.grib FF-MS*.grib DF-MS*.grib

  $HIMAN -d 5 -f windvector_ec.json -t grib -s ec_cuda windvector_ec_ground_source.grib windvector_ec_hybrid_source.grib

  grib_compare -A 0.01 windvector_ec_ground_FF_6_result.grib FF-MS_height_10_rll_161_177_0_006.grib

  if [ $? -ne 0 ];then
    echo windvector/ec wind speed failed on GPU
    exit 1
  fi

  grib_compare -A 0.01 windvector_ec_ground_FF_9_result.grib FF-MS_height_10_rll_161_177_0_009.grib

  if [ $? -eq 0 ];then
    echo windvector/ec wind speed success on GPU!
  else
    echo windvector/ec wind speed failed on GPU
    exit 1
  fi

  grib_compare -A 0.01 windvector_ec_ground_DD_6_result.grib DD-D_height_10_rll_161_177_0_006.grib

  if [ $? -ne 0 ];then
    echo windvector/ec wind direction failed on GPU
    exit 1
  fi

  grib_compare -A 0.01 windvector_ec_ground_DD_9_result.grib DD-D_height_10_rll_161_177_0_009.grib

  if [ $? -eq 0 ];then
    echo windvector/ec wind direction success on GPU!
  else
    echo windvector/ec wind direction failed on GPU
    exit 1
  fi
fi
