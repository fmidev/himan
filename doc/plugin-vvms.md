# Summary

vvms plugin is used to calculate vertical velocity in m/s.

Plugin is optimized for GPU use.

# Required source parameters

* ver = vertical velocity (Pa/s)
* T = air temperature (K)
* p = pressure (Pa)

# Output parameters

* VV-MS or VV-MMS

Unit of resulting parameters is m/s or mm/s.

# Method of calculation

Using the hydrostatic equation the geometric vertical velocity is calculated as follows:

```
# hydrostatic equation
dp/dz = -g * roo

# vertical velocity in pressure coordinates
ver = dp/dt

# vertical velocity in geometric coordinates
w = dz/dt

# substitute variables to hydrostatic equation
dp/dt = -g * roo * dz/dt
w = -ver / (g * roo)
```

Where:

```
roo = air density
g = 9.81 m/s^2
```

Equation of state for ideal gases:

```
p = roo * R * T
-->
roo = p / (R * T)
```

Where:

```
p = pressure (Pa)
R = specific gas constant of dry air = 287 (J/kg/K)
T = air temperature (K)
```

Substitute air density into the equation for geometric vertical velocity:

```
w = -ver * R * T / (g * p)
```

Positive values are upwards.

# Per-plugin configuration options

millimeters: define if output parameter unit should be mm/s instead of m/s

    "millimeters" : true

reverse: define if pressure-coordinate variable is produced from geometric vertical velocity

    "reverse" : true
