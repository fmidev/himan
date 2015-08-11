#!/bin/sh

set -ex

if [ $(/sbin/lsmod | egrep -c "^nvidia") -eq 0 ]; then
  echo "no cuda device found"
  exit 0
fi

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f cross_meridian*.grib

$HIMAN -d 5 -f cross_meridian.json --no-cuda -s stat source_latlon.grib
mv cross_meridian.json.grib newbase.grib

$HIMAN -d 5 -f cross_meridian.json -s stat source_latlon.grib
mv cross_meridian.json.grib cuda.grib

grib_compare -b referenceValue -A 1.5 newbase.grib cuda.grib

if [ $? -eq 0 ];then
  echo interpolation/cross_meridian success!
else
  echo interpolation/cross_meridian failed
  exit 1
fi

rm -f half*.grib

$HIMAN -d 5 -f half.json --no-cuda -s stat source_rotlatlon.grib
mv half.json.grib newbase.grib

$HIMAN -d 5 -f half.json -s stat source_rotlatlon.grib
mv half.json.grib cuda.grib

grib_compare -b referenceValue -A 0.5 newbase.grib cuda.grib

if [ $? -eq 0 ];then
  echo interpolation/half success!
else
  echo interpolation/half failed
  exit 1
fi

rm -f missing*.grib

$HIMAN -d 5 -f missing.json --no-cuda -s stat source_missing.grib
mv missing.json.grib newbase.grib

$HIMAN -d 5 -f missing.json -s stat source_missing.grib
mv missing.json.grib cuda.grib

grib_compare -A 1 newbase.grib cuda.grib

if [ $? -eq 0 ];then
  echo interpolation/missing success!
else
  echo interpolation/missing failed
  exit 1
fi

rm -f missing*.grib

$HIMAN -d 5 -f missing_bl.json --no-cuda -s stat source_missing.grib
mv missing_bl.json.grib newbase.grib

$HIMAN -d 5 -f missing_bl.json -s stat source_missing.grib
mv missing_bl.json.grib cuda.grib

grib_compare -A 1.5 newbase.grib cuda.grib

if [ $? -eq 0 ];then
  echo interpolation/missing_bl success!
else
  echo interpolation/missing_bl failed
  exit 1
fi
