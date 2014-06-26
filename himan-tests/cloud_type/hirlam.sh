#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f cloud_type_hirlam.json.grib

$HIMAN -d 5 -f cloud_type_hirlam.json -t grib --no-cuda -a hirlam/*

grib_compare cloud_type_hirlam.json.grib result_hirlam.grib 

if [ $? -eq 0 ];then
  echo relative cloud_type/hirlam success on CPU!
else
  echo relative cloud_type/hirlam failed on CPU
  exit 1
fi

