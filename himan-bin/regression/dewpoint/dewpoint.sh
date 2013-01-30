#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f himan*.grib

$HIMAN -d 5 -f dewpoint_arome.json -t grib source.grib

grib_compare result.grib himan_TD-C_2013012806.grib

if [ $? -eq 0 ];then
  echo dewpoint success!
else
  echo dewpoint failed
  exit 1
fi

python ../test.py result.grib himan_TD-C_2013012806.grib

if [ $? -eq 0 ];then
  echo vvms success!
else
  echo vvms failed
  exit 1
fi

