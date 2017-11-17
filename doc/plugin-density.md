# Summary

Density plugin calculates the density of (dry) air using the ideal gas law.

# Required source parameters

* p = pressure (Pa)
* T = air temperature (K)
* R = specific gas constant for dry air = 287 J / kg*K

# Output parameters

RHO-KGM3

Unit of resulting parameter is kg/m^3.

# Method of calculation

    ρ = P / (R * T)

# Per-plugin configuration options

None
