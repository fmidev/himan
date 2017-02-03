# Summary

Monin_obukhov -plugin calculates the inverse value of monin-obukhov length.

# Required source parameters

* t = air temperature (K)
* ρ = air density (kg/m^3)
* _u*_ = friction velocity (m/s)
* s = surface sensible heat flux (J/m^2)
* l = surface latent heat flux (J/m^2)
* p = pressure (Pa)
* g = gravitational constant
* κ = von Karman constant
* cp = specific heat capacity (J/K)
* Q = surface heat flux

# Output parameters

MOL-M

Unit of resulting parameter is m.

# Method of calculation

    tc = t - 273.15
    Q = s + 0.07 * l
    cp = 1.0056e3 + 0.017766 * (tc + 4.0501e-4 * tc^2 - 1.017e-6 * tc^3 +
         1.4715e-8 * tc^4 - 7.4022e-11 * tc^5 + 1.2521e-13 * tc^6;
    1/L = -(κ * g * Q) / (ρ * cp * (u*)^3 * t)

# Per-plugin configuration options

None