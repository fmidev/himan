#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f hybrid_pressure_hir.json.grib

$HIMAN -d 5 -f hybrid_pressure_hir.json -t grib q2.grib q3.grib q4.grib p.grib

grib_compare -r -A 0.01 result.grib hybrid_pressure_hir.json.grib

if [ $? -eq 0 ];then
  echo hybrid_pressure/hirlam success!
else
  echo hybrid_pressure/hirlam failed
  exit 1
fi

