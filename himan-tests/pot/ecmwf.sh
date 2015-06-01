#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f POT-PRCNT*.grib

$HIMAN -d 5 -j 1 -f pot_ec.json --no-cuda CAPE-JKG_GROUND_0_0_ll_2880_1441_0_015_1.grib RRR-KGM2_height_0_ll_2880_1441_0_015.grib

grib_compare POT-PRCNT_height_0_ll_2880_1441_0_015.grib result.grib 

if [ $? -eq 0 ];then
  echo pot/hirlam success on CPU!
else
  echo pot/hirlam failed on CPU
  exit 1
fi

rm -f POT-PRCNT*.grib
