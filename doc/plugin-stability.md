# Summary

stability plugin calculates indices describing the state and stability of the atmosphere from pressure and model level data. 

The indices are 

* k-index
** http://glossary.ametsoc.org/wiki/Stability_index
* showalter index
** http://glossary.ametsoc.org/wiki/Stability_index
* lifted index
** http://glossary.ametsoc.org/wiki/Stability_index
** source data for the lifted parcel is an average from the lowest 500 meters
* cross totals index
** http://glossary.ametsoc.org/wiki/Stability_index
* vertical totals index
** http://glossary.ametsoc.org/wiki/Stability_index
* total totals index
** http://glossary.ametsoc.org/wiki/Stability_index
* bulk shear
** between 0 and 1km
** between 0 and 3km
** between 0 and 6km
** https://en.wikipedia.org/wiki/Wind_shear
* effective bulk shear
** http://www.spc.noaa.gov/exper/mesoanalysis/help/help_eshr.html
** Lifted parcel level is used as effective inflow base
** effective inflow top is found when going upwards from base up to 50% of EL height
* storm relative helicity
** between 0 and 1km
** between 0 and 3km
** http://www.spc.noaa.gov/exper/mesoanalysis/help/help_srh1.html
* equivalent potential temperature difference between 0 and 3km
* wind speed at 1.5km
* energy-helicity index 
** https://en.wikipedia.org/wiki/Hydrodynamical_helicity
* bulk richardson number
** http://glossary.ametsoc.org/wiki/Bulk_richardson_number

Plugin is optimized for GPU use.

# Required source parameters

* T = air temperature (K)
    * pressure levels 850, 700, 500
* TD = dew point temperature (K)
    * pressure levels 850, 700
* T_LIFT1 = temperature of an air parcel lifted from 850hPa to 500hPa
* T_LIFT2 = temperature of an air parcel lifted from surface to 500hPa, using temperature, dewpoint and pressure averaged from the lowest 500m
* U = wind u component
    * 0, 1 and 6 km above ground
* V = wind v component
    * 0, 1 and 6 km above ground

# Output parameters

Following parameters have index as unit:

* K-index: KINDEX-N
* Showalter index: SI-N
* Lifted index: LI-N
* Cross totals index: CTI-N
* Vertical totals index: VTI-N
* Total totals index: TTI-N

Following parameters are in knots:

* Bulk shear 0..1 km:  WSH-1-KT
* Bulk shear 0..6 km:  WSH-KT

# Method of calculation

    KINDEX-N = T_850 - T_500 + TD_850 - (T_700 - TD_700)
    SI-N = T_500 - T_LIFT1
    LI-N = T_500 - T_LIFT2
    CTI-N = TD_850 - T_500
    VTI-N = T_850 - T_500
    TTI-N = CTI-N - VTI-N

    U = U_1km - U_0km
    V = V_1km - V_0km
    WSH-1-KT = sqrt(U^2 + V^2) * 1.943844492

    U = U_6km - U_0km
    V = V_6km - V_0km
    WSH-KT = sqrt(U^2 + V^2) * 1.943844492

# Per-plugin configuration options

None
