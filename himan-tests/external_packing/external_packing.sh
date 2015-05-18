#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f T-C*

$HIMAN -d 5 -j 1 -f external_packing_bzip2.json --no-cuda source.grib
$HIMAN -d 5 -j 1 -f external_packing_gzip.json --no-cuda T-C_height_0_rll_1030_816_0_015.grib.bz2
$HIMAN -d 5 -j 1 -f external_unpacking.json --no-cuda T-C_height_0_rll_1030_816_0_015.grib.gz

grib_compare T-C_height_0_rll_1030_816_0_015.grib source.grib 

if [ $? -eq 0 ];then
  echo external_packing success on CPU!
else
  echo external_packing failed on CPU
  exit 1
fi

rm -f T-C*
