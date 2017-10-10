# Summary

auto_taf -plugin scans the atmosphere for cloud layers and categorizes them to be used for automatically generated TAF-messages.

# Required source parameters

* N-0TO1
* H-M
* CL-2-FT
* CBTCU-FT
* LCL-HPA

# Output parameters

* FEW1-FT
* SCT1-FT
* BKN1-FT
* OVC1-FT
* CB-FT
* CB-PRCNT

# Method of calculation

1. First it looks from the TCu_Cb_param if there are convective clouds found and base of convective clouds is initially set to height of LCL500.
2. Then it looks all model levels from the surface level upwards and the sequential layers with cloudiness over 5% are defined as a cloud layer.
3. If the maximum cloudiness of a cloud layer is over 90%, the layer is OVC-cloud, if it is over 62% it is BKN-cloud, if it is over 37% it is SCT-cloud and otherwise it is a FEW-cloud.
4. Pointless cloud layers will be deleted e.g.

* only 1 layer to one significant low cloud height class (<200ft, 200-400ft, 500-900ft, 1000-1400ft), lowest BKN/OVC, lowest SCT or finally lowest FEW
* over 1500ft if difference of bases after rounding is <=500ft, only lowest OVC, lowest BKN, lowest SCT or finally lowest FEW is chosen

5. After that if there is convective cloud, its cloud fraction and base height are defined.

* The nearest layer to the initial base of convective clouds is chosen if it is close enough
* Otherwise, if the nearest layer is the lowest layer, a layer is left and another convective layer is created with base value of the initial base of convective clouds and cloud fraction of the nearest layer
* If the nearest layer is on height of over 5000ft, then it will be dropped to initial base of convective clouds and it is the convective cloud

6. Height of lowest FEW, lowest SCT, lowest BKN, lowest OVC and height and cloud fraction of convective cloud is returned.

# Per-plugin configuration options

None
