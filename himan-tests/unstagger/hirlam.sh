#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f U-MS*.grib
rm -f V-MS*.grib

start_time=`date +%s`

$HIMAN -d 4 -f unstagger.json -t grib source_U.grib source_V.grib --no-cuda

end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.

grib_compare U-MS_hybrid_10_rll_1030_816_0_001.grib result_U.grib

if [ $? -eq 0 ];then
  echo unstagger/hirlam U-component success on CPU!
else
  echo unstagger/hirlam failed on CPU
  exit 1
fi

grib_compare V-MS_hybrid_10_rll_1030_816_0_001.grib result_V.grib

if [ $? -eq 0 ];then
  echo unstagger/hirlam V-component success on CPU!
else
    echo unstagger/hirlam failed on CPU
    exit 1
fi

