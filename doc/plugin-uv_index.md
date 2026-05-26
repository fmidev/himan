# Summary

uv_index plugin calculates clear-sky ultraviolet (UV) index forecasts and
total-ozone anomalies. It is a port of the legacy FMI `uvennuste` /
`otsoniano` tool, fitted into the himan framework.

# Required source parameters

UV modes (`uvi`, `uvimax`, `uv`):

* total column ozone (TOZONE-KGM2)
* snow water equivalent (SD-TM2)
* surface geopotential (Z-M2S2), fetched at analysis time (step 0)

Ozone-anomaly mode (`o3anom`):

* total column ozone (TOZONE-KGM2)

In addition the UV modes require these static files (paths given in the
plugin's JSON options block, see below):

* clear-sky UV irradiance look-up table (`disort_24012003.dat`)
* aerosol optical depth and single-scattering albedo climatology
  (four grib files: `atau_summer`, `atau_winter`, `assa_summer`,
  `assa_winter`)

The ozone-anomaly mode requires:

* total-ozone climatology grib (`O3clim.grib`, 5 messages keyed by level
  0..4 holding the Fourier coefficients B0..B4 of the TOMS climatology)

# Output parameters

| mode             | output param(s)         | description                                              |
|------------------|-------------------------|----------------------------------------------------------|
| `uvimax`         | UVIMAX-N                | clear-sky daily-max UV index (solar-noon SZA)            |
| `uvi`            | UVI-N                   | clear-sky instantaneous UV index (SZA at valid time)     |
| `uv`             | UVIMAX-N and UVI-N      | both produced in one pass (shared input fetches)         |
| `o3anom`         | O3ANOM-PRCNT            | 100 · (forecast − climatology) / climatology             |

# Method of calculation

UV modes feed a 7-element input vector `(albedo, ozone, atau, assa, ctau,
sza, alt)` to a precomputed DISORT clear-sky UV irradiance look-up table
and apply Earth-Sun distance correction and a fixed scaling factor of 40
to produce the UV index. The cloud optical depth `ctau` is hard-coded to
0; cloud effects are layered on by a downstream plugin. Surface albedo is
0.4 if the snow water-equivalent is above a threshold, 0.03 otherwise.
The solar zenith angle is taken at local solar noon for `uvimax` and at
the forecast valid time for `uvi`. The table is interpolated by
tensor-product Lagrange interpolation with degree at most three per axis.

The ozone-anomaly mode bilinear-interpolates the five Fourier
coefficients of the TOMS climatology onto the target grid, reconstructs
the climatological total ozone at the forecast Julian day, and writes
the percent deviation of the forecast field from that climatology.

The plugin is bit-for-bit faithful to the legacy `uvennuste` /
`otsoniano` reference outputs, including the linear-extrapolation edge
behaviour of `cieInterp.for` and the legacy unit scalings.

# Per-plugin configuration options

`mode` selects which output(s) the plugin produces. Default is `uv`.

    "mode": "uvi"        # UVI-N only
    "mode": "uvimax"     # UVIMAX-N only
    "mode": "uv"         # both UVIMAX-N and UVI-N in one pass
    "mode": "o3anom"     # O3ANOM-PRCNT

`disort_table` (required for UV modes): path to the DISORT lookup table.

    "disort_table": "disort_24012003.dat"

`atau_summer`, `atau_winter`, `assa_summer`, `assa_winter` (required for
UV modes): paths to the four seasonal aerosol climatology grib files.
The plugin picks summer (months 4..9) or winter by the forecast valid
month.

    "atau_summer": "atau_summer.grib"
    "atau_winter": "atau_winter.grib"
    "assa_summer": "assa_summer.grib"
    "assa_winter": "assa_winter.grib"

`o3_climatology` (required for `o3anom` mode): path to the 5-message
TOMS total-ozone climatology grib (levels 0..4 carry the Fourier
coefficients B0..B4).

    "o3_climatology": "O3clim.grib"
