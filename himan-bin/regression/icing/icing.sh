#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f icing_hir1h.json.grib

$HIMAN -d 5 -j 1 -f icing_hir1h.json -t grib t.grib vvms.grib cldwat.grib

grib_compare result.grib icing_hir1h.json.grib

if [ $? -eq 0 ];then
  echo icing/hirlam success!
else
  echo icing/hirlam failed
  exit 1
fi

