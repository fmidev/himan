# Summary

relative_humidity plugin calculates relative humidity of air (doh).

Plugin is optimized for GPU use.

# Required source parameters

Either

* T = air temperature (K)
* P = air pressure (Pa)
* Q = specific humidity (kg/kg)
* E = water vapor saturated pressure (Pa)
* Dimensionless ratio of the specific gas constant of dry air to the specific gas constant for water vapor = 0.622

Or

* T = air temperature (K)
* TD = dewpoint temperature (K)

# Output parameters

RH-PRCNT

Unit of resulting parameter is %.

# Method of calculation

Method 1

    T = T - 273.15
    E = (T > -5) ? (6.107 * 10^(7.5 * T / (237 + T))) : (6.107 * 10^(9.5 * T / (265.5 + T)))
    RH = 100 * (P * Q) / (0.622 * E) * (P - E) / (P - (Q*P) / 0.622)

Method 2

    T = T - 273.15
    TD = TD - 273.15
    RH = 100 * (e^(1.8+17.27*(TD / (TD + 237.3)))) / (e^(1.8 + 17.27 * (T / (T + 237.3))))

# Per-plugin configuration options

None