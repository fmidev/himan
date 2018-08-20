# Summary

stability plugin calculates indices describing the state and stability of the atmosphere from pressure and model level data. 

stability calculations are actually split into two plugins, 'stability_simple' and 'stability'. 'stability_simple' consists of the first
four parameters in the list below, 'stability' holds the rest.

The indices are 

* k-index
  * http://glossary.ametsoc.org/wiki/Stability_index
* cross totals index
  * http://glossary.ametsoc.org/wiki/Stability_index
* vertical totals index
  * http://glossary.ametsoc.org/wiki/Stability_index
* total totals index
  * http://glossary.ametsoc.org/wiki/Stability_index
* showalter index
  * http://glossary.ametsoc.org/wiki/Stability_index
* lifted index
  * http://glossary.ametsoc.org/wiki/Stability_index
  * source data for the lifted parcel is an average from the lowest 500 meters
* bulk shear
  * between 0 and 1km
  * between 0 and 3km
  * between 0 and 6km
  * https://en.wikipedia.org/wiki/Wind_shear
* effective bulk shear
  * http://www.spc.noaa.gov/exper/mesoanalysis/help/help_eshr.html
  * Lifted parcel level is used as effective inflow base
  * effective inflow top is found when going upwards from base up to 50% of EL height
* cape shear
  * http://apps.ecmwf.int/codes/grib/param-db?id=228044
  * ebs is used as wind shear parameter
  * most unstable cape is the cape parameter
* storm relative helicity
  * between 0 and 1km
  * between 0 and 3km
  * http://www.spc.noaa.gov/exper/mesoanalysis/help/help_srh1.html
* equivalent potential temperature difference between 0 and 3km
* wind speed at 1.5km
* energy-helicity index 
  * https://en.wikipedia.org/wiki/Hydrodynamical_helicity
* bulk richardson number
  * http://glossary.ametsoc.org/wiki/Bulk_richardson_number
* mean mixing ratio in the lowest 500m

Plugin is optimized for GPU use.

# Required source parameters

# Output parameters

Parameters from stability_simple plugin:

* K-index: KINDEX-N
* Cross totals index: CTI-N
* Vertical totals index: VTI-N
* Total totals index: TTI-N

Parameters from stability plugin:

Level (height, 0):

* Effective bulk shear: EWSH-MS
* Lifted index: LI-N
* Showalter index: SI-N
* Cape shear: CAPES-JKG

Level (height, 1500):

* Wind speed: FF-MS

Level (heightlayer, 0, 500):

* Mean mixing ratio: Q-KGKG

Level (heightlayer, 0, 1000):

* Bulk shear: BS-MS
* Storm relative helicity: HLCY-M2S2
* Energy helicity index: EHI-N

Level (heightlayer, 0, 3000):

* Bulk shear: BS-MS
* Storm relative helicity: HLCY-M2S2
* Equivalent potential temperature difference: TPE-K

Level (heightlayer, 0, 6000):

* Bulk shear: BS-MS
* Bulk richardson number: BRN-N

# Method of calculation

# Per-plugin configuration options

None
