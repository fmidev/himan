#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f seaicing_hir1h.json.grib

$HIMAN -d 5 -f seaicing_hir1h.json -t grib t2m.grib t0m.grib ff10.grib

grib_compare result.grib seaicing_hir1h.json.grib

if [ $? -eq 0 ];then
  echo seaicing/hirlam success!
else
  echo seaicing/hirlam failed
  exit 1
fi

