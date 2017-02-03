# Summary

hybrid_height plugin calculates the metric height of a given hybrid level by either using geopotential if available (fast), or using temperature and pressure (slow).

Plugin is optimized for GPU use.

# Required source parameters

* gp = geopotential in m2/s2

or 

* t = temperature in K
* p = pressure in hPa

# Output parameters

HL-M

Unit of resulting parameter is m.

# Method of calculation

If using geopotential, metric height is

    h = gp / g

If using iterative approach

    h = 2 * g * (prevt - t) * log(prevp / p) + prevh

In the iterative approach, the calculation is started from the lowest level in a single threaded fashion.

# Per-plugin configuration options

None