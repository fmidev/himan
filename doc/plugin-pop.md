# Summary

pop plugin calculates the probability of precipitation (any amount).

# Required source parameters

precipitation rate from following models

* ECMWF HRES (weight: 2)
* ECMWF HRES previous forecast (1)
* Hirlam (1)
* Harmonie (1)
* GFS (1)
* ECMWF ENS median (0.25)
* EMCWF ENS third quartile (0.25)
* ECMWF ENS probability of 6h precipition sum >= 1mm 
* ECMWF ENS probability of 6h precipition sum >= 0.1mm
* FMI Multi-Model Ensemble PEPS probability of 1h precipition sum >= 0.2mm (1)

Result data is averaged using a stencile with sizes ranging from 25 to 81 grid points. 

# Output parameters

POP-PRCNT

Unit of resulting parameter is %.

# Method of calculation

 The mathematical definition of Probability of Precipitation is defined as: PoP = C * A
 
 C = the confidence that precipitation will occur somewhere in the forecast area

 A = the percentage of the area that will receive measurable precipitation, if it occurs at all

# Per-plugin configuration options

None
