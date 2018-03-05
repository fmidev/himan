# Summary

tropopause-plugin scans the atmosphere vertically to find the height (flight level) of the tropopause.


# Required source parameters

* HL-M
* T-K
* P-HPA

# Output parameters

* TROPO-FL

# Method of calculation

1. Starting from flight level 140 upwards to 530 find lowest level where the lapse-rate drops below 2K/km. 
2. Check that the average lapse rate between this level and all higher levels within 2 km does not exceed 2K/km.

# Per-plugin configuration options

None
