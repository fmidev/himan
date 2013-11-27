#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f hybrid_height.json.grib

$HIMAN -d 5 -f hybrid_height.json -t grib t3.grib prev_t3.grib p3.grib prev_p3.grib prev_h3.grib

grib_compare result.grib hybrid_height.json.grib

if [ $? -eq 0 ];then
  echo hybrid_height/hirlam success!
else
  echo hybrid_height/hirlam failed
  exit 1
fi