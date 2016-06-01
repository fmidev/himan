#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f POT-PRCNT*.grib

$HIMAN -d 5 -j 1 -f pot_ec.json --no-cuda CAPE-JKG_GROUND_0_0_ll_3600_1801_0_001_1.grib RRR-KGM2_height_0_ll_3600_1801_0_001.grib RRR-KGM2_height_0_ll_3600_1801_0_002.grib

grib_compare POT-PRCNT_height_0_ll_3600_1801_0_001.grib result.grib 

if [ $? -eq 0 ];then
  echo pot/ecmwf success on CPU!
else
  echo pot/ecmwf failed on CPU
  exit 1
fi

rm -f POT-PRCNT*.grib
