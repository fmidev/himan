# Summary

turbulence plugin is used to calculate two [clear air turbulence](https://en.wikipedia.org/wiki/Clear-air_turbulence) (CAT) indices which describe turbulence in the free atmosphere.

# Required source parameters

* S = scaling factor based on wind speed [non-dimensional]
* VWS = Vertical Wind Shear [1/s]
* DEF = deformation of the horizontal wind [1/s]
* CVG = convergence of the horizontal wind [1/s] 
* DIV = divergence of the horizontal wind [1/s]


# Output parameters

TI-S2 and TI2-S2

Unit of resulting parameters is 1/s.

# Method of calculation

    TI1-S2 = S x VWS × DEF

    TI2-S2 = S x VWS × (DEF + CVG) = S × VWS × (DEF - DIV)

where

    VWS = |ΔV/Δz| = sqrt [ (Δu/Δz)2 + (Δv/Δz)2 ]
    DEF = sqrt [ (Δu/Δx - Δv/Δy)2 + (Δv/Δx + Δu/Δy)2 ] = sqrt (DST^2 + DSH^2)
    CVG = - Δu/Δx - Δv/Δy = - DIV
    V = horizontal wind 
    u = u component of wind   
    v = v component of wind
    Δz = layer thickness
    Δx = Δy = distance between grid points
    DST = stretching deformation [s-1]
    DSH = shearing deformation [s-1]

Notes from algorithm author:

> Especially when calculated in the model resolution, the original Ellrod indices TI and TI2 tend to give (greatly) inflated values in situations where the wind speed is quite low, but vertical wind shear and/or deformation  (+convergence) are strong. This leads to difficulties in interpreting the index values and probably over-forecasting  of CAT. Thus a modified version of the index is presented here by adding a scaling factor based on wind speed into the original index. This should lead to a more consistent index which better describes CAT potential in different  situations, possibly even as a quasi-probability value (higher index values consistently indicate increased moderate  or greater turbulence risk).

# Per-plugin configuration options

None