# Summary

hybrid_pressure calculates the pressure of a given hybrid level when vertical coordinates and surface pressure are known.

Plugin is optimized for GPU use.

# Required source parameters

* a,b = vertical coordinates
* ps = surface pressure in Pa

# Output parameters

P-HPA

Unit of resulting parameter is hectopascal.

# Method of calculation

Plugin calculates the exact values of the hybrid levels (vertical coordinates are often given in half-levels), and then calculates the pressure

    p = 0.01 * (a + p * b)

# Per-plugin configuration options

None