#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/build/release/himan"
fi

#rm -f RHO-KGM3*.grib ABS*.grib

$HIMAN -d 4 -f absolute_humidity_harmonie.json -t grib harmonie_p_source.grib harmonie_t_source.grib harmonie_rain_source.grib harmonie_snow_source.grib harmonie_graupel_source.grib

grib_compare ./RRRS-KGM2_hybrid_60_rll_290_594_0_360.grib result_solidpr_harmonie.grib
VAR_1=$?

if [ $VAR_1 -eq 0 ];then
  echo absolute_humidity/harmonie success on CPU!
else
  echo absolute_humidity/harmonie failed on CPU
  exit 1
fi

#rm -f RHO-KGM3*.grib RRR*.grib
