#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f cloud_type_ecmwf.json.grib

$HIMAN -d 5 -f cloud_type_ecmwf.json -t grib --no-cuda -a ec/*

grib_compare cloud_type_ecmwf.json.grib result_ecmwf.grib 

if [ $? -eq 0 ];then
  echo relative cloud_type/ecmwf success on CPU!
else
  echo relative cloud_type/ecmwf failed on CPU
  exit 1
fi

