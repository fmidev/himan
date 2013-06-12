#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="/home/peramaki/workspace/himan-bin/build/release/himan"
fi

cd /home/peramaki/workspace/himan-bin/regression/hybrid_pressure

rm -f hybrid_pressure_hir.json.grib

$HIMAN -d 5 -f hybrid_pressure_hir.json -t grib q2.grib q3.grib q4.grib p.grib

grib_compare result.grib hybrid_pressure_hir.json.grib

if [ $? -eq 0 ];then
  echo hybrid_pressure/hirlam success!
  exit 0
else
  echo hybrid_pressure/hirlam failed
  exit 1
fi

