#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f seaicing_ec.json.grib

$HIMAN -d 5 -f seaicing_ec.json -t grib t2m_ec.grib t0m_ec.grib ff10_ec.grib

grib_compare result_ec.grib seaicing_ec.json.grib

if [ $? -eq 0 ];then
  echo seaicing/ec success!
else
  echo seaicing/ec failed
  exit 1
fi

