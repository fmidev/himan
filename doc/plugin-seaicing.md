# Summary

seaicing plugin calculates an index describing the risk of sea spray icing (the accumulation of ice on sea vessel hulls). Plugin has two slightly different formulas, one meant for the Baltic sea and the other for the oceans where sea saltiness is larger than in the Baltic sea.

# Required source parameters

* T = air temperature at 2m (K)
* TG = ground temperature (K)
* FF = wind speed at 10m (m/s)
* S = saltiness index

# Output parameters

ICING-N (Baltic sea) or SSICING-N (oceans).

Unit of resulting parameter is an index value ranging from 0 to 4.

0: No risk for icing
1: Minor risk for icing, icing rate is less than 0.7 cm/h
2: Moderate risk for icing, icing rate between 0.7 and 2 cm/h
3: Major risk for icing, icing rate between 2 and 4 cm/h
4: Extreme risk for icing, icing rate more than 4 cm/h

# Method of calculation

Index by Antonios Niros: Vessel icing forecast and services: further development and perspectives.

Pseudocode

    icing = FF * (-S - T) / (1 + 0.3 * (TG + S))

    if icing <= 0:
        icing = 0
    elif icing < 22.4:
        icing = 1
    elif icing < 53.3:
        icing = 2
    elif icing < 83:
        icing = 3
    else:
        icing = 4

# Per-plugin configuration options

global: Define whether to calculate for Baltic sea or oceans (default: false = Baltic sea)

    "global" : "true"