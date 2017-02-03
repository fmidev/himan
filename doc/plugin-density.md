# Summary

Density plugin calculates the density of (dry) air using ideal gas law.

# Required source parameters

* p = pressure (Pa)
* t = air temperature (K)
* R = gas constant for dry air = 287 J / kg*K

# Output parameters

RHO-KGM3

Unit of resulting parameter is kg/m^3.

# Method of calculation

    ρ = P / (R * t)

# Per-plugin configuration options

None