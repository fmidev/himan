#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f himan*.grib

$HIMAN -d 5 -f vvms_ec1h.ini -t grib t.grib vv.grib

grib_compare result.grib himan_VV-MS_2013012100.grib

if [ $? -eq 0 ];then
  echo vvms success!
else
  echo vvms failed
  exit 1
fi

python test.py result.grib himan_VV-MS_2013012100.grib

if [ $? -eq 0 ];then
  echo vvms success!
else
  echo vvms failed
  exit 1
fi
