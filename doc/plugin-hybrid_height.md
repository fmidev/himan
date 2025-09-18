# Summary

hybrid_height plugin calculates the metric height of a given hybrid level by either using geopotential if available (fast), or using temperature and pressure (slow).

Plugin is optimized for GPU use.

# Required source parameters

* gp = geopotential in m2/s2

or 

* T = temperature in K
* p = pressure in hPa

# Output parameters

HL-M

Unit of resulting parameter is m.

# Method of calculation

If using geopotential, metric height is

    h = gp / g

When geopotential is not available hybrid height is found through vertical integration of the hypsometric equation (https://en.wikipedia.org/wiki/Hypsometric_equation)

    h = (R / g) * ((prevT - T) / 2) * log(prevp / p) + prevh

Where

* R is the specific gas constant for dry air
* g is acceleration of gravity

Note: virtual temperature is not used.

When solving the hypsometric equation, the calculation is started from the lowest level in a single threaded fashion.

# Per-plugin configuration options

    "method": "geopotential" | "hypsometric"

Defines which method to use for calculating hybrid height. Default is "geopotential".
