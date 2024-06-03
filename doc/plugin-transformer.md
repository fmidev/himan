# Summary

transformet plugin is used to perform simple parameter transformation.

Possible transformations are:

* scaling (adding scale and base)
* area interpolation
* land-sea mask apply
* level change
* forecast type change
* time interpolation
* vertical interpolation

Plugin is optimized for GPU use.

# Required source parameters

Any given input parameter.

# Output parameters

Output parameter defined in configuration.

# Method of calculation

Applying scale and base is simple value * scale + base.

Area interpolation can be done between any Himan-supported projections: latitude-longitude, rotated latitude-longitude, stereographic, lambert conformal conic, reduced gaussian grid, irregular grid (list of points)

Applying land-sea mask is done with value 0.5 by default, meaning that grid points with 50% or more sea are masked.

# Per-plugin configuration options

base: define base (default: 0)

    "base" : "10"

scale: define scale (default: 1)

    "scale" : "0.01"

maximum_value: define the maximum value for the data. Missing values are not changed. Minimum and maximum need to be defined together.

    "maximum_value" : "260"

minimum_value: define the minimum value for the data. Missing values are not changed. Minimum and maximum need to be defined together.

    "minimum_value" : "250"

target_param: define target parameter name 

    "target_param" : "T-C"

target_param_aggregation: define target parameter (time) aggregation type

    "target_param_aggregation" : "average"

target_param_aggregation_period: define target parameter aggregation period

    "target_param_aggregation_period" : "06:00"

target_param_processing_type: define target parameter processing type

    "target_param_processing_type" : "standard deviation"

source_param: define source parameter name

    "source_param" : "T-K"

source_param_aggregation: define source parameter (time) aggregation type

    "source_param_aggregation" : "average"

source_param_processing_type: define source parameter processing type

    "source_param_processing_type" : "standard deviation"

source_level_type: define source level type (default: target level type, ie. no level change is made)

    "source_level_type" : "height"

source_levels: define source level values (default: target level values)

    "target_level_values" : "100"

apply_landsea_mask: define is land-sea mask should be applied (default: false)

    "apply_landsea_mask" : true

landsea_mask_threshold: define threshold used in mask apply (default: 0.5)

    "landsea_mask_threshold" : "0.6"

interpolation: define interpolation method (default: whatever is defined in database)

    "interpolation" : "bilinear|nearest point"

target_forecast_type: define the target forecast type, if it differs from source type

    "target_forecast_type" : "cf|deterministic|analysis|pfNN"

rotation: specify which parameters (if any) should be rotated from projection north coordinate to earth-normal north (default: not done)

    "rotation" : "U-MS,V-MS"

time_interpolation: define if Himan should do time interpolation if data is not found for some leadtime (default: false). Himan will try to find neighboring data up to +/- 6 hours from current leadtime.

    "time_interpolation" : true

vertical_interpolation: define if Himan should do vertical interpolation, if data is not found for some level (default: false). Only supported for level types 'height' (unit=meters) and 'pressure' (unit=hectopascal). Both time_interpolation and vertical_interpolation cannot be defined at the same time.

    "vertical_interpolation" : true

change_missing_value_to: define if missing value should be changed to some normal floating point value (note: this is not the same as defining a different missing value: the resulting grib will have no missing values defined!)

    "change_missing_value_to" : 0

write_empty_grid: define if an empty grid (all values missing) should be written out or not. default: yes.

    "write_empty_grid" : false

precision: define the precision used writing output files, decimal places. default: use what is in database

    "precision" : 2

landscape_interpolation: use topography and land-sea mask to downscale gridded data

    "landscape_interpolation" : true

grib1_X: override grib1 parameter information that is normally fetched from database. Note: both keys need to exist.

    "grib1_table_number" : 0
    "grib1_parameter_number" : 0

grib2_X: override grib2 parameter information that is normally fetched from database. Note: all three keys need to exist.

    "grib2_discipline" : 0
    "grib2_parameter_category" : 0
    "grib2_parameter_number" : 0

univ_id: override querydata parameter number that is normally fetched from database

    "univ_id" : 4

extra_file_metadata: write any metadata to resulting files, overriding any existing metadata. Implementation and support depends on the filetype. Currently only grib is supported. Arguments are given in key-value pairs, comma separated. Data type of value can be string (:s), integer (:d) or float (:d). Default is integer.

    "extra_file_metadata" : "shapeOfTheEarth=6,latitudeOfFirstGridPointInDegrees:d=10.01"

named_ensemble: fetch ensemble configuration from database. Makes it easier to handle complicated ensembles like MEPS. This options is not supported with landscape/time/level interpolation, land sea masking or vector component rotation

    "named_ensemble" : "MEPS_SINGLE_ENSEMBLE"

source_forecast_period: specify a different leadtime for fetching data. Cannot be combined with 'time_interpolation'.

    "source_forecast_period" : "00:00:00"

read_previous_forecast_if_not_found: If data for some forecast is not found, try to read it from previous forecast (preserving data valid times). Does not apply to UV vector rotation. Default: false. Cannot be used together with time_interpolation.

    "read_previous_forecast_if_not_found" : true


