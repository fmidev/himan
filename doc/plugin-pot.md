# Summary

pot plugin calculates the probability of thunderstorms.

# Required source parameters

* c = surface CAPE (J/kg)
* c1040 = cold CAPE (J/kg)
* rr = precipitation rate (mm/h)

# Output parameters

POT-PRCNT

Unit of resulting parameter is percent.

# Method of calculation

c1040 is the primary CAPE parameter, c is a fallback. The plugin combines the atmospheric stability with the precipitation forecast and performs spatial and temporal smoothing.

# Per-plugin configuration options

None