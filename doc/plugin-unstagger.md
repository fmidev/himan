# Summary

unstagger plugin is used to unstagger an Arakawa type C grid to match type A.

Plugin is optimized for GPU use.

# Required source parameters

* wind u vector (m/s)
* wind v vector (m/s)

# Output parameters

* U-MS
* V-MS

Unit of resulting parameters is m/s.

# Method of calculation

Data is unstaggered half a grid length to match Arakawa A. This is much faster that doing bilinear/nearest point interpolation.

# Per-plugin configuration options

None