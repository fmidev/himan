# Summary

visibility plugin is used to calculate horizontal visibility.

# Required source parameters

* air temperature in the lowest 100m (K)
* humidity in the lowest 100m (%)
* total cloudiness in the lowest 300m (%)
* find speed at 10m (m/s)
* find speed at boundary layer height (m/s)
* precipitation sum (kg/m^2)
* precipitation form (code)
* cloud ceiling (feet)

# Output parameters

* VV2-M

Unit of resulting parameters is m.

# Method of calculation

Three possible visibility-altering weather scenarios are considered:

* precipitation (rain, snow)
* fog (from stratus)
* fog (from radiation)

From each a visibility value is computed and the worst one is selected as the final visibility.

# Per-plugin configuration options

None