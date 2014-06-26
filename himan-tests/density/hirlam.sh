#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f RHO-KGM3*.grib

$HIMAN -d 5 -f density_hirlam.json -t grib --no-cuda hirlam_p_source.grib hirlam_t_source.grib

grib_compare RHO-KGM3_hybrid_65_rll_1030_816_0_006.grib result_hirlam.grib 

if [ $? -eq 0 ];then
  echo density/hirlam success on CPU!
else
  echo density/hirlam failed on CPU
  exit 1
fi

rm -f RHO-KGM3*.grib
