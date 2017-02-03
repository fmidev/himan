# Summary

Icing -plugin calculates icing, the formation of water/ice on the surfaces of an aircraft. The result is an index ranging from 0 to 15.

# Required source parameters

* t = temperature (K)
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
 
    if t <= 0 and t > -1: tcor = -2
    else if t <= -1 and t > -2: tcor = -1
    else if t <= -2 and t > -3: tcor = 0
    else if t <= -3 and t > -12: tcor = 1
    else if t <= -12 and t > -15: tcor = 0
    else if t <= -15 and t > -18: tcor = -1
    else if t < -18: tcor = -2
    else: tcor = 0
 
    if c <= 0 or t > 0: icing = 0
    else: icing = round(log(cl) + 6) + vcor + tcor
 
    icing = min(15, max(0, icing))

# Per-plugin configuration options

None