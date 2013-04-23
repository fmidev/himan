#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f vvms_ec.json.grib vvms_ec.json-CPU.grib

$HIMAN -d 5 -f vvms_ec.json -t grib -s vvms_ec_nocuda ec_t.grib ec_vv.grib --no-cuda

grib_compare -A 0.0001 ec_result.grib vvms_ec.json.grib

if [ $? -eq 0 ];then
  echo vvms/ec success!
else
  echo vvms/ec failed
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  mv vvms_ec.json.grib vvms_ec.json-CPU.grib

  $HIMAN -d 5 -f vvms_ec.json -t grib -s vvms_ec_cuda ec_t.grib ec_vv.grib

  grib_compare -b referenceValue -A 0.0001 vvms_ec.json.grib vvms_ec.json-CPU.grib

  if [ $? -eq 0 ];then
    echo vvms/ec success on GPU!
  else
    echo vvms/ec failed on GPU
    exit 1
  fi
else
  echo "no cuda device found for cuda tests"
fi
