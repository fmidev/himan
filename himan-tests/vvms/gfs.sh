#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f VV-MS*.grib P-PA*.grib

$HIMAN -d 5 -f gfs.json -t grib -s vvms_gfs gfs_source.grib --no-cuda

grib_compare vvms_gfs_result.grib VV-MS_pressure_925_ll_720_361_0_192.grib

if [ $? -eq 0 ];then
  echo vvms/hl success!
else
  echo vvms/hl failed
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  mv vvms_hl.json.grib vvms_hl.json-CPU.grib

  $HIMAN -d 5 -f gfs.json -s vvms_gfs gfs_source.grib

  grib_compare -b referenceValue -A 0.0001 vvms_gfs_result.grib VV-MS_pressure_925_ll_720_361_0_192.grib

  if [ $? -eq 0 ];then
    echo vvms/hl success GPU
  else
    echo vvms/hl failed on GPU
    exit 1
  fi

else
  echo "no cuda device found for cuda tests"
fi
