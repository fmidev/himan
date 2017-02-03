# Summary

ncl plugin is used to calculate the height of zero degree or -20 degree level.

# Required source parameters

* t = air temperature (K)
* h = height (m)

# Output parameters

H0C-M or HM20C-M.

Unit of resulting parameter is m.

# Method of calculation

Search the atmosphere vertically starting from ground for the first height where temperature crosses zero or -20 degree boundary. Plugin takes account the effects of inversion in the lower atmosphere.

Note that similar functionality can be achieved with hitool function VerticalHeight(), although hitool does not consider the effects of inversion.

# Per-plugin configuration options

temp: select whether to search for zero degree or -20 degree pivot point

    "temp" : "0 | -20"