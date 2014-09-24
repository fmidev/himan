#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f HSADE*.grib

$HIMAN -d 5 -f ecmwf.json source_ec.grib

grib_compare result_ec.grib HSADE1-N_height_0_rll_161_177_0_005.grib

if [ $? -eq 0 ];then
  echo rain_type/ecmwf success!
else
  echo rain_type/ecmwf failed
  exit 1
fi

$HIMAN -d 5 -f ecmwf_6h.json source_ec_6h.grib

grib_compare result_ec_6h.grib HSADE1-N_height_0_ll_361_91_0_228.grib

if [ $? -eq 0 ];then
  echo rain_type/ecmwf success!
else
  echo rain_type/ecmwf failed
  exit 1
fi

