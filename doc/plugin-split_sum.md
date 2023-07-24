# Summary

split_sum plugin is used to split cumulative parameters such as precipitation and radiation to rates and powers.

# Required source parameters

Precipitation

* RR-1-MM, RR-3-MM, RR-6-MM, RR-12-MM, RR-24-MM, RRR-KGM2: precipitation accumulation (RR-KGM2)
* RRRC-KGM2: convective precipitation accumulation (RRC-KGM2)
* RRRL-KGM2: large scale precipitation accumulation (RRL-KGM2)

Snow

* SN-3-MM, SN-6-MM, SN-24-MM, SN-120-MM, SNR-KGM2: snow precipitation accumulation (SNACC-KGM2)
* SNRC-KGM2: convective snow precipitation accumulation (SNC-KGM2)
* SNRL-KGM2: large scale snow precipitation accumulation (SNL-KGM2)

Other

* GRR-MMH: graupel precipitation accumulation (GR-KGM2)
* RRRS-KGM2: solid precipitation accumulation (RRS-KGM2)

Radiation

* RADGLO-WM2: global radiation accumulation (RADGLOA-JM2)
* RADGLOC-WM2: global radiation accumulation clear sky (RADGLOCA-JM2)
* RADLW-WM2: long wave radiation accumulation (RADGLOA-JM2)
* RTOPLW-WM2: net long wave radiation, top of atmosphere accumulation (RADGLOA-JM2)
* RNETLW-WM2: net long wave radiation accumulation (RADGLOA-JM2)
* RADSW-WM2: short wave radiation accumulation (RADGLOA-JM2)

General format

Because user might want arbitrary aggregation periods to be used, it is also possible to give all the necessary 
details in json-configuration. Details below.

# Output parameters

All parameters are optional and can be turned on via configuration file (see Per-plugin configuration options)

Precipitation

* RR-1-MM: one hour precipitation sum (mm)
* RR-3-MM: three hour precipitation sum (mm)
* RR-6-MM: six hour precipitation sum (mm)
* RR-12-MM: twelve hour precipitation sum (mm)
* RR-24-MM: 24 hour precipitation sum (mm)
* RRR-KGM2: precipitation rate (kg/m^2 or mm/h)
* RRRC-KGM2: convective precipitation rate (kg/m^2 or mm/h)
* RRRL-KGM2: large scale precipitation rate (kg/m^2 or mm/h)

Snow

* SN-3-MM: three hour snow accumulation (mm)
* SN-6-MM: six hour snow accumulation (mm)
* SN-24-MM: 24 hour snow accumulation (mm)
* SN-120-MM: 120 hour snow accumulation (mm)
* SNR-KGM2: snow precipitation rate (kg/m^2 or mm/h)
* SNRC-KGM2: convective snow precipitation rate (kg/m^2 or mm/h)
* SNRL-KGM2: large scale snow precipitation rate (kg/m^2 or mm/h)

Other

* GRR-MMH: graupel precipitation rate (kg/m^2 or mm/h)
* RRRS-KGM2: solid precipitation rate (snow + graupel + hail) (kg/m^2 or mm/h)
* RRS-3-MM: three hour solid precipitation accumulation (mm)
* RRS-24-MM: 24 hour solid precipitation accumulation (mm)

Radiation

* RADGLO-WM2: global radiation (W/m^2)
* RADGLOC-WM2: global radiation clear sky (W/m^2)
* RADLW-WM2: long wave radiation (W/m^2)
* RTOPLW-WM2: net long wave radiation, top of atmosphere (W/m^2)
* RNETLW-WM2: net long wave radiation (W/m^2)
* RADSW-WM2: short wave radiation (W/m^2)
* RNETSW-WM2: net short wave radiation (W/m^2)

# Method of calculation

Sums

    sum = value - previous_value

Rates

    rate = value - previous_value / period_length

Powers (radiation)

    power = (value - previous_value) / (period_length * 3600)

NB! All precipitation rates that are calculated for time periods of one hour or more, are
marked with aggregation type 'accumulation'. Aggregation type 'average' is only used when
time period is second, for radiations.

# Per-plugin configuration options

Calculate 1/3/6/12/24 hour precipitation from an accumulation:

    "rr1h" : true
    "rr3h" : true
    "rr6h : true
    "rr12h" : true
    "rr24h" : true

Calculate 3/6/24/120 hour snow accumulation:

    "sn3h" : true
    "sn6h" : true
    "sn24h" : true
    "sn120h" : true

Calculate precipitation rate, large scale precipitation rate or convective precipitation rate for total precipitation (first three) and snow accumulation (latter three).

    "rrr" : true
    "rrrl" : true
    "rrrc" : true
    "snr" : true
    "snrl" : true
    "snrc" : true

Calculate graupel precipitation rate, solid precipitation rate and solid precipitation accumulation.

    "grr" : true
    "rrrs" : true
    "rrs3h" : true
    "rrs24h" : true

Calculate power from radiation accumulation for global radiation, long wave radiation, short wave radiation, radiation at the top of atmosphere and net long wave radiation.

    "glob" :true
    "globc" : true
    "lw" : true
    "sw" : true
    "toplw" : true
    "netlw" : true
    "netsw" : true

General configuration

Here we define all the required components:
* source_param: name of the source parametee
* source_param_aggregation: aggregation type of the target parameter
* target_param: name of the source parametee
* target_param_aggregation: aggregation type of the target parameter
* target_param_aggregation_period: aggregation period length for the target parameter
* lower_limit: set hard coded lower limit, default: 0 (remove by setting value to "MISSING")
* is_rate: define if this parameter is a rate (values are divided by the length of the time period), default: false
* truncate_smaller_values: define if smaller values than this threshold are truncated to zero, default: "MISSING" (no truncation)
* scale: apply scaling, default: 1.0
* rate_resolution: if rates are calculated, define rate base time unit (second, minute, hour), default: hour

Example: produce the same thing as shortcut option "rrr" (for meps producer):

    "name" : "split_sum",
    "source_param" : "RR-KGM2",
    "source_param_aggregation" : "accumulation",
    "target_param" : "RRR-KGM2",
    "target_param_aggregation" : "accumulation",
    "target_param_aggregation_period" : "01:00:00",
    "lower_limit" : "0",
    "is_rate" : true
    "truncate_smaller_values" : "0.01" 
    "scale" : "1.0"
    "rate_resolution" : "hour"


Example: produce the same thing as shortcut option "glob":

    "name" : "split_sum",
    "source_param" : "RADGLOBA-JM2",
    "source_param_aggregation" : "accumulation",
    "target_param" : "RADGLO-WM2",
    "target_param_aggregation" : "average",
    "lower_limit" : "0",
    "is_rate" : true
    "truncate_smaller_values" : "MISSING" 
    "scale" : "1.0"
    "rate_resolution" : "second"


Example: produce net short wave radiation power from averages (powers) ie case icon:

    "name" : "split_sum",
    "source_param" : "RNETSW-WM2",
    "source_param_aggregation" : "average",
    "target_param" : "RNETSW-WM2",
    "target_param_aggregation" : "average",
    "lower_limit" : "MISSING",
    "is_rate" : false
    "truncate_smaller_values" : "MISSING" 
    "scale" : "1.0"
    "rate_resolution" : "hour"


