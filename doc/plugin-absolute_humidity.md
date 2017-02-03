# Summary

absolute_humidity -plugin calculates the absolute humidity of air.

# Required source parameters

* a = air density (kg/m^3)
* r = rain water mixing ratio (kg/kg)
* s = snow mixing ratio (kg/kg)
* g = graupel mixing ration (kg/kg)
* ρ = air density (kg/m^3)

# Output parameters

ABSH-KGM3.

Unit of resulting parameter is kg/m^3.

# Method of calculation

    a = ρ * (r + s + g)

# Per-plugin configuration options

None