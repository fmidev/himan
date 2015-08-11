#!/bin/sh

set -ex

if [ $(/sbin/lsmod | egrep -c "^nvidia") -eq 0 ]; then
  echo "no cuda device found"
  exit 0
fi

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f rotlatlon_*.grib

$HIMAN -d 5 -f rotlatlon_to_latlon.json --no-cuda -s stat source_rotlatlon.grib
mv rotlatlon_to_latlon.json.grib newbase.grib

$HIMAN -d 5 -f rotlatlon_to_latlon.json -t grib -s stat source_rotlatlon.grib
mv rotlatlon_to_latlon.json.grib cuda.grib

grib_compare -b referenceValue -A 0.5 newbase.grib cuda.grib

if [ $? -eq 0 ];then
  echo interpolation/rotlatlon_to_rotlatlon success!
else
  echo interpolation/rotlatlon_to_rotlatlon failed
  exit 1
fi

$HIMAN -d 5 -f rotlatlon_to_rotlatlon.json --no-cuda -s stat source_rotlatlon.grib
mv rotlatlon_to_rotlatlon.json.grib newbase.grib

$HIMAN -d 5 -f rotlatlon_to_rotlatlon.json -t grib -s stat source_rotlatlon.grib
mv rotlatlon_to_rotlatlon.json.grib cuda.grib

grib_compare -A 0.5 newbase.grib cuda.grib

if [ $? -eq 0 ];then
  echo interpolation/rotlatlon_to_rotrotlatlon success!
else
  echo interpolation/rotlatlon_to_rotrotlatlon failed
  exit 1
fi

$HIMAN -d 5 -f rotlatlon_to_polster.json --no-cuda -s stat source_rotlatlon.grib
mv rotlatlon_to_polster.json.grib newbase.grib

$HIMAN -d 5 -f rotlatlon_to_polster.json -t grib -s stat source_rotlatlon.grib
mv rotlatlon_to_polster.json.grib cuda.grib

grib_compare newbase.grib cuda.grib

if [ $? -eq 0 ];then
  echo interpolation/rotlatlon_to_polster success!
else
  echo interpolation/rotlatlon_to_polster failed
  exit 1
fi


