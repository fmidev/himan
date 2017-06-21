# Summary

Monin_obukhov -plugin calculates the inverse value of monin-obukhov length.

# Required source parameters

* T = air temperature (K)
* Tv = virtual air temperature (K)
* ρ = air density (kg/m^3)
* _u*_ = friction velocity (m/s)
* shf = surface sensible heat flux (W/m^2)
* lhf = surface latent heat flux (W/m^2)
* p = pressure (Pa)
* g = acceleration of gravity (m/s^2)
* κ = von Kármán constant (0.41)
* cp = specific heat capacity of dry air (J/K)
* Qv = surface virtual temperature flux (K*m/s)
* Lw = specific latent heat of condensation of water (J/kg)

# Output parameters

MOL-M

Unit of resulting parameter is 1/m.

# Method of calculation

    Tc = T - 273.15
    cp = 1.0056e3 + 0.017766 * (Tc + 4.0501e-4 * Tc^2 - 1.017e-6 * Tc^3 +
         1.4715e-8 * Tc^4 - 7.4022e-11 * Tc^5 + 1.2521e-13 * Tc^6
    Lw = 2500800.0 - 2360.0 * Tc + 1.6 * Tc^2 - 0.06 * Tc^3
    Qv = (shf + 0.61 * cp * T / Lw * lhf) / (ρ * cp)
    1/L = -(κ * g * Qv) / ((u*)^3 * Tv)

# Per-plugin configuration options

None
