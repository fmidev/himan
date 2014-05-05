#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f weather_symbol_hir1h.json.grib

$HIMAN -d 5 -f weather_symbol_hir1h.json -t grib cldsym.grib hsade1.grib

grib_compare result.grib weather_symbol_hir1h.json.grib

if [ $? -eq 0 ];then
  echo weather_symbol/hirlam success!
else
  echo weather_symbol/hirlam failed
  exit 1
fi

