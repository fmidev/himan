#!/bin/sh

set -x

if [ -z "$HIMAN" ]; then
	export HIMAN="../../himan-bin/himan"
fi

rm -f R*.grib

$HIMAN -d 4 -f precipitation_rate_harmonie.json -t grib harmonie_p_source.grib harmonie_t_source.grib harmonie_rain_source.grib harmonie_snow_source.grib harmonie_graupel_source.grib --no-cuda

grib_compare ./RSI-KGM2_hybrid_60_rll_290_594_0_360.grib result_solidpr_harmonie.grib
VAR_1=$?
grib_compare ./RRI-KGM2_hybrid_60_rll_290_594_0_360.grib result_rain_harmonie.grib
VAR_2=$?

if [ $VAR_1 -eq 0 -a $VAR_2 -eq 0 ];then
  echo precipitation-rate/harmonie success on CPU!
else
  echo precipitation-rate/harmonie failed on CPU
  exit 1
fi

if [ $(/sbin/lsmod | egrep -c "^nvidia") -gt 0 ]; then

  $HIMAN -d 4 -f precipitation_rate_harmonie.json -t grib harmonie_p_source.grib harmonie_t_source.grib harmonie_rain_source.grib harmonie_snow_source.grib harmonie_graupel_source.grib

  grib_compare -A 0.01 ./RSI-KGM2_hybrid_60_rll_290_594_0_360.grib result_solidpr_harmonie.grib
  VAR_1=$?
  grib_compare -A 0.01 ./RRI-KGM2_hybrid_60_rll_290_594_0_360.grib result_rain_harmonie.grib
  VAR_2=$?

  if [ $VAR_1 -eq 0 -a $VAR_2 -eq 0 ];then
    echo precipitation-rate/harmonie success on GPU!
  else
    echo precipitation-rate/harmonie failed on GPU
    exit 1
  fi

fi
