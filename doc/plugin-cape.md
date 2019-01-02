# Summary

Cape plugin calculates three different CAPE (convective available potential energy) parameter variants with three different starting values. While doing this it also finds out the state parameters of LCL (Lifting Condensation Level), LFC (Level of Free Convectivity), EL (Equilibrium Level) and LPL (Lifted Parcel Level) as well as CIN (Convective Inhibition) value.

The possible source data variations are:

* _surface_. Temperature and humidity are taken from close to surface (lowest hybrid level, ~10 meters usually)
* _500m mix_. Temperature and humidity are averages from 0-500m above ground. Averaging is done using potential temperature and mixing ratio.
* _most unstable_. Temperature and humidity are taken from the hybrid level that produces highest equivalent potential temperature. Search is started from lowest hybrid level and capped at 550hPa.

The parameters produced for each of these are:

* _cape_: regular Cape value, unit J/kg
* _cape 3km_: Cape value integration is capped at 3km, unit J/kg
* _cape -10..-40_: Cape integration is done only where environment temperature is between -10 and -40 centigrade, unit J/kg
* _cin_: Convective inhibition, unit J/kg, values are negative
* _lcl temperature_: temperature of lifting condensation level (~cloud base), Kelvins
* _lcl pressure_: pressure of lifting condensation level, hPa
* _lcl height_: height of lifting condensation level, m
* _lfc temperature_: temperature of level of free convection, K
* _lfc pressure_: pressure of level of free convection, hPa
* _lfc height_: height of level of free convection, m
* _el temperature (first)_: temperature of first equilibrium level found, K
* _el pressure (first)_: pressure of first equilibrium level found, hPa
* _el height (first)_: height of first equilibrium level found, m
* _el temperature (last)_: temperature of last equilibrium level found, K
* _el pressure (last)_: pressure of last equilibrium level found, hPa
* _el height (last)_: last of first equilibrium level found, m
* _lpl temperature_: temperature of lifted parcel level temperature (mu only), K
* _lpl pressure_: pressure of lifted parcel level temperature (mu only), hPa
* _lpl height_: height of lifted parcel level temperature (mu only), m

This results then in altogether 3 * 16 + 3 = 51 separate output parameters.

Plugin is optimized for GPU use.

# Required source parameters

All parameters must be found from vertical levels starting from lowest hybrid level (~10 meters) up to ~250 hPa.

* Air temperature, K
* Relative humidity, %
* Metric height of hybrid level
* Pressure of hybrid level

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
EL-LAST-K
EL-LAST-HPA
EL-LAST-M
LPL-K
LPL-HPA
LPL-M

The results are written to a level corresponding the source value type, possible values are:

* surface layer
* height layer 0 to 500m
* most unstable layer

# Method of calculation

Source data for surface-level based calculation is taken directly from the lowest hybrid level. For 500m mix level data, potential temperature and mixing ratio are sampled with 2 hPa interval (for the lowest 500 meters) and the mean value of these is converted to temperature and dewpoint.

For maximum theta e level, source data equivalent potential temperature is calculated for all levels below 550hPa. From this profile all local maximas below 650hPa are picked and sorted based on absolute value. Then three highest theta e locations are chose, temperature and dewpoint value are taken from their level and mucape is produced for all three starting positions. Whichever produces the highest mucape value is eventually chosen.

LCL values are calculated using Boltons approximations. For moist adiabatic lift of air parcel Wobus method is used. For CAPE and CIN integration, virtual temperature is used. Both, first and last, equilibrium levels are searched, although in most cases these two are equal.

The final CAPE an CIN results are averaged over the nearest grid points.

# Per-plugin configuration options

source_data: define one or more source data variants.

    "source_data" : [ <value>, <value>, ... ]

Where value is one of:

* surface
* 500m mix
* most unstable

virtual_temperature: define if virtual temperature correction should be used when air parcel is saturated. Default is `true`.

    "virtual_temperature" : true | false

