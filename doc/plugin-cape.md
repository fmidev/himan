# Summary

Cape plugin calculates three different CAPE (convective available potential energy) parameter variants with three different starting values. While doing this it also finds out the properties of LCL (Lifting Condensation Level), LFC (Level of Free Convectivity) and EL (Equilibrium Level) levels as well as CIN (Convective Inhibition) value.

The possible source data variations are:

* _surface_. Temperature and humidity are taken from close to surface (lowest hybrid level, ~10 meters usually)
* _500m mix_. Temperature and humidity are averages from 0-500m above ground. Averaging is done using potential temperature and mixing ratio.
* _most unstable_. Temperature and humidity are taken from the hybrid level that produces highest equivalent potential temperature. Search is started from lowest hybrid level and capped at 600hPa.

The parameters producer for each of these are:

* _cape_: regular Cape value, unit J/kg
* _cape 3km_: Cape value integration is capped at 3km, unit J/kg
* _cape -10..-40_: Cape integration is done only where environment temperature is between -10 and -40 centigrade, unit J/kg
* _cin_: Convective inhibition, unit J/kg, values are negative
* _lcl temperature_: temperature of lifting condensation level (~cloud base), Kelvins
* _lcl pressure_: pressure of lifting condensation level, hPa
* _lfc temperature_: temperature of level of free convection, Kelvins
* _lfc pressure_: pressure of level of free convection, hPa
* _el temperature_: temperature of equilibrium level, Kelvins
* _el pressure_: pressure of equilibrium level, hPa

This results then in altogether 3 * 10 = 30 separate output parameters.

Plugin is optimized for GPU use.

# Required source parameters

All parameters must be found from vertical levels starting from ground and up to ~100 Pa

* Air temperature, K
* Relative humidity, %
* Metric height of level

# Output parameters

CAPE-JKG
CAPE1040-JKG
CAPE3KM-JKG
CIN-JKG
LCL-K
LCL-HPA
LCL-M
LFC-K
LFC-HPA
LFC-M
EL-K
EL-HPA
EL-M

The results are written to a level corresponding the source value type, possible values are:

surface layer
height layer 0 to 500m
most unstable layer

# Method of calculation

Values are integrated while scanning through the 3D state of the atmosphere. The final CAPE results are averaged over the nearest four grid points.

# Per-plugin configuration options

source_data: define one or more source data variants.

    "source_data" : [ <value>, <value>, ... ]

Where value is one of:

* surface
* 500m mix
* most unstable