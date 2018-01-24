# Summary

Icing -plugin calculates icing, the formation of water/ice on the surfaces of an aircraft. The result is an index ranging from 0 to 15.

# Required source parameters

* T = temperature (K)
* w = vertical velocity (m/s or mm/s)
* c = cloud water (kg/kg)

# Output parameters

ICING-N

Unit of resulting parameter is an index ranging from 0 (no icing) to 15 (very heavy icing).

# Method of calculation

Pseudocode

    // vcor: vertical velocity correction factor
 
    if w < 0: vcor = -1
    else if w >= 0 and w <= 50: vcor = 0
    else if w >= 50 and w <= 100: vcor = 1
    else if w >= 100 and w <= 200: vcor = 2
    else if w >= 200 and w <= 300: vcor = 3
    else if w >= 300 and w <= 1000: vcor = 4
    else vcor = 5
 
    // tcor: temperature correction factor
 
    if T <= 0 and T > -1: Tcor = -2
    else if T <= -1 and T > -2: tcor = -1
    else if T <= -2 and T > -3: tcor = 0
    else if T <= -3 and T > -12: tcor = 1
    else if T <= -12 and T > -15: tcor = 0
    else if T <= -15 and T > -18: tcor = -1
    else if T < -18: Tcor = -2
    else: Tcor = 0
 
    if c <= 0 or T > 0: icing = 0
    else: icing = round(log(cl) + 6) + vcor + Tcor
 
    icing = min(15, max(0, icing))

# Per-plugin configuration options

None
