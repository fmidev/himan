#!/bin/sh

echo "this test will work only when hybrid_pressure produces parameter P-PA, or when"
echo "vvms-plugin accepts parameter P-HPA"

exit 0

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f VV-MS*.grib P-PA*.grib

$HIMAN -d 5 -f vvms_hl.json -t grib -s vvms_hl_nocuda vvms_hl_source.grib vvms_hl_pa_source.grib --no-cuda

grib_compare vvms_hl_result_13.grib VV-MS_hybrid_65_rll_1030_816_0_013.grib

if [ $? -ne 0 ];then
  echo vvms/hl failed
  exit 1
fi

grib_compare vvms_hl_result_14.grib VV-MS_hybrid_65_rll_1030_816_0_014.grib

if [ $? -ne 0 ];then
  echo vvms/hl failed
  exit 1
fi

grib_compare vvms_hl_result_15.grib VV-MS_hybrid_65_rll_1030_816_0_015.grib

if [ $? -eq 0 ];then
  echo vvms/hl success!
else
  echo vvms/hl failed
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  mv vvms_hl.json.grib vvms_hl.json-CPU.grib

  $HIMAN -d 5 -f vvms_hl.json -t grib -s vvms_hl_cuda vvms_hl_source.grib vvms_hl_pa_source.grib

  grib_compare -b referenceValue -A 0.0001 vvms_hl_result_13.grib VV-MS_hybrid_65_rll_1030_816_0_013.grib

  if [ $? -ne 0 ];then
    echo vvms/ec failed on GPU
    exit 1
  fi

  grib_compare -b referenceValue -A 0.0001 vvms_hl_result_14.grib VV-MS_hybrid_65_rll_1030_816_0_014.grib

  if [ $? -ne 0 ];then
    echo vvms/ec failed on GPU
    exit 1
  fi

  grib_compare -b referenceValue -A 0.0001 vvms_hl_result_15.grib VV-MS_hybrid_65_rll_1030_816_0_015.grib

  if [ $? -eq 0 ];then
    echo vvms/ec success GPU
  else
    echo vvms/ec failed on GPU
    exit 1
  fi

else
  echo "no cuda device found for cuda tests"
fi
