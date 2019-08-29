# Summary

transformet plugin is used to perform simple parameter transformation.

Possible transformations are:

* scaling (adding scale and base)
* area interpolation
* land-sea mask apply
* level change
* forecast type change
* time interpolation

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

target_param: define target parameter name 

    "target_param" : "T-C"

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

change_missing_value_to: define if missing value should be changed to some normal floating point value (note: this is not the same as defining a different missing value: the resulting grib will have no missing values defined!)

    "change_missing_value_to" : 0

write_empty_grid: define if an empty grid (all values missing) should be written out or not. default: yes.

    "write_empty_grid" : false

precision: define the precision used writing output files, decimal places. default: use what is in database

    "precision" : 2
