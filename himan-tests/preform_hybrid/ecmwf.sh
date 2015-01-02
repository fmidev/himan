#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/build/release/himan"
fi

rm -f PRECFORM*.grib

$HIMAN -d 5 -f preform_ec.json --no-cuda ecmwf_source.grib

grib_compare PRECFORM2-N_height_0_rll_529_461_0_003.grib ./ecmwf_result.grib

if [ $? -eq 0 ];then
  echo preform_hybrid/ecmwf success on CPU!
else
  echo preform_hybrid/ecmwf failed on CPU
  exit 1
fi

rm -f PRECFORM*.grib
