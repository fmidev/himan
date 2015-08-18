#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan"
fi

rm -f icing_ec.json.grib

$HIMAN -d 5 -j 1 -f icing_ec.json -t grib t_ec.grib vvms_ec.grib cldwat_ec.grib  --no-cuda

grib_compare result_ec.grib icing_ec.json.grib

if [ $? -eq 0 ];then
  echo icing/ec success!
else
  echo icing/ec failed
  exit 1
fi

