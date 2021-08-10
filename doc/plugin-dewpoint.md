# Summary

Dewpoint plugin calculates dewpoint temperature in K. 

Formula used is taken from "The Relationship between Relative Humidity and the Dewpoint Temperature in Moist Air, A Simple Conversion and Applications" by Mark G. Lawrence (http://journals.ametsoc.org/doi/pdf/10.1175/BAMS-86-2-225).

Plugin has been optimized for GPU use.

# Required source parameters

* T = air temperature (K)
* r = relative humidity (%)
* g = gas constant ratio between water vapor and air, 461.5 J / K * kg
* h = latent heat for water vaporization, 2.5e6 J / kg

# Output parameters

TD-C

Unit of resulting parameter is Kelvin.

# Method of calculation

    td = T / (1 - T * ln(r * 0.01) * g / h)

Relative humidity values are capped to maximum 100%.

# Per-plugin configuration options

None
