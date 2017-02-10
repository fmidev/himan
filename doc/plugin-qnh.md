# Summary

qnh plugin calculates air pressure reduced to mean sea level according to ICAO standard atmosphere.

# Required source parameters

* geopotential at surface (m^2/s^2)
* pressure at surface (Pa)
* sea level standard atmospheric pressure p0 = 1013.25 hPa
* sea level standard temperature T0 = 288.15 K
* Earth-surface gravitational acceleration g = 9.80665 m/s2.
* temperature lapse rate L = 0.0065 K/m
* universal gas constant R = 8.31447 J/(mol K)
* molar mass of dry air M = 0.0289644 kg/mol

# Output parameters

QNH-HPA

Unit of resulting parameter is hPa.

# Method of calculation

ICAO ISA:

    p(h) = p0 * (1-L*h/T0)^(g*M/R/L)
    => h(p) = [T0-T0*(p/p0)^(R*L/g/M)] / L

QFE -> QNH:
  1. calculate ICAO ISA altitude z corresponding to pressure at station (QFE) [m]
  2. calculate MSL (at station) = z - topo (topography = height of aerodrome) [m]
  3. calculate p at level MSL in ISA = QNH


# Per-plugin configuration options

None