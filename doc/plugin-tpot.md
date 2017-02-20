# Summary

tpot plugin is used to calculate potential temperature, pseudoadiabatic potential temperature or equivalent potential temperature.

Plugin is optimized for GPU use.

# Required source parameters

* air temperature (K)
* air pressure (Pa)
* dew point temperature (K)

# Output parameters

TP-K (potential temperature), TPW-K (pseudo-adiabatic potential temperature), TPE-K (equivalent potential temperature)

Unit of resulting parameters is Kelvin.

# Method of calculation

Potential temperature: 

* Poisson's equation

Pseudo-adiabatic potential temperature: 

* Davies-Jones: An Efficient and Accurate Method for Computing the Wet-Bulb Temperature

Equivalent potential temperature:

* Bolton: The Computation of Equivalent Potential Temperature (1980)

# Per-plugin configuration options

theta: define whether to calculate potential temperature (default: true)

    "theta" : true

thetaw: define whether to calculate pseudo-adiabatic potential temperature (default: false)

    "thetaw" : true

thetae: define whether to calculate equivalent potential temperature (default: false)

    "thetae" : true