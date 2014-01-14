#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f RH-PRCNT*.grib P-HPA*grib

$HIMAN -d 5 -f relative_humidity_harmonie.json -t grib --no-cuda source_harmonie.grib

grib_compare ./RH-PRCNT_hybrid_44_rll_290_594_0_1860.grib result_harmonie_1860.grib 

if [ $? -ne 0 ];then
  echo relative humidity/harmonie failed on CPU
  exit 1
fi

grib_compare ./RH-PRCNT_hybrid_44_rll_290_594_0_1875.grib result_harmonie_1875.grib 

if [ $? -ne 0 ];then
  echo relative humidity/harmonie failed on CPU
  exit 1
fi

grib_compare ./RH-PRCNT_hybrid_44_rll_290_594_0_1890.grib result_harmonie_1890.grib 

if [ $? -eq 0 ];then
  echo relative humidity/harmonie success on CPU!
else
  echo relative humidity/harmonie failed on CPU
  exit 1
fi
