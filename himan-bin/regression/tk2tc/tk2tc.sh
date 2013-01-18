#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f himan*.grib

$HIMAN -d 5 -f tk2tc_ec1h.ini -t grib source.grib

grib_compare result.grib himan_2013011800.grib

if [ $? -eq 0 ];then
  echo tk2tc_ini success!
else
  echo tk2tc_ini failed
  exit 1
fi
