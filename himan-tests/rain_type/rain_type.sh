#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f rain_type_hir1h.json.grib

$HIMAN -d 5 -f rain_type_hir1h.json -t grib t.grib tmax.grib t850.grib n.grib kindex.grib z850.grib z1000.grib rr1mm_next.grib rr1mm_now.grib

grib_compare result.grib rain_type_hir1h.json.grib

if [ $? -eq 0 ];then
  echo rain_type/hirlam success!
else
  echo rain_type/hirlam failed
  exit 1
fi

