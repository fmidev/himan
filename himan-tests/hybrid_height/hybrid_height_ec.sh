#!/bin/sh

set -x

exit 0;

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f hybrid_height_ec.json.grib2

$HIMAN -d 5 -f hybrid_height_ec.json -t grib source_ec.grib

grib_compare result_ec.grib2 hybrid_height_ec.json.grib2

if [ $? -eq 0 ];then
  echo hybrid_height/ec success!
else
  echo hybrid_height/ec failed
  exit 1
fi
