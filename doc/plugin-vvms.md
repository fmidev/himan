# Summary

vvms plugin is used to calculate vertical velocity in m/s.

Plugin is optimized for GPU use.

# Required source parameters

* ver = vertical velocity (Pa/s)
* t = air temperature (K)
* p = pressure (Pa)

# Output parameters

* VV-MS or VV-MMS

Unit of resulting parameters is m/s or mm/s.

# Method of calculation

    w = -ver * 287 * t * (9.81 * p)

Positive values are upwards.

# Per-plugin configuration options

millimeters: define if output parameter unit should be mm/s instead of m/s

    "millimeters" : true