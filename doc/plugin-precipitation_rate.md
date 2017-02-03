# Summary

precipitation_rate plugin calculates the approximated liquid and solid precipitation rate. The plugin is used to calculate precipitation rate in the atmosphere, it does not equal the precipitation rate at the surface of earth.

# Required source parameters

* ρ = air density (kg/m^3)
* r = rain mixing ratio (kg/kg)
* s = snow mixing ratio (kg/kg)
* g = graupel mixing ratio (kg/kg)

# Output parameters

RRI-KGM2 liquid precipitation rate
RSI-KGM2 snow/solid precipitation rate

Unit of resulting parameter is mm/h (or kg/m^2).

# Method of calculation

   rri = (ρ * max(r, 0) * 1000 / 0.072)^(1.0 / 0.880)
   rsi = (ρ * max(s + g, 0) * 1000.0 / 0.200)^(1.0 / 0.900)

# Per-plugin configuration options

None