# Summary

transformet plugin is used to perform simple parameter transformation.

Possible transformations are:

* scaling (adding scale and base)
* area interpolation
* land-sea mask apply
* level change
* forecast type change

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
