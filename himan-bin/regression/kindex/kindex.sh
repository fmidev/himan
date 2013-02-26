#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f himan*.grib

$HIMAN -d 5 -f kindex_hir1h.json -t grib t850.grib t700.grib t500.grib td850.grib td700.grib

grib_compare result.grib himan_KINDEX-N_201302260600.grib

if [ $? -eq 0 ];then
  echo kindex success!
else
  echo kindex failed
  exit 1
fi

