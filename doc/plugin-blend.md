# Summary

Blend plugin calculates a weighted blend of different forecast models using precalculated bias correction fields and weights calculated with method based on mean absolute error (MAE).

For calculating the bias correction and MAE fields LAPS analysis is used as the truth field. This means that all the data needs to be converted to a common grid, currently we employ LAPS' grid. The transformation is done with the transformer plugin as a separate pass as the data comes in. Correct operation of the actual blending needs a days worth of BC and MAE fields.

# Configuration

* param = radon parameter name, for example "T-K"
* mode = "blend" | "mae" | "bias"

if mode = "mae" or mode = "bias":

	"producer" : "MOS" | "ECG" | "HL2" | "MEPS" | "GFS"
	"analysis_hour" : 0 .. 23
	"hours" : number of hours to process in total (fetching historical data)


For example:
	"plugins" : [ { "name" : "blend", "param" : "T-K", "mode" : "bias", "producer" : "HL2", "hours" : 54, "analysis_hour" : "0" } ]

Or:

	"plugins" : [ { "name" : "blend", "param" : "T-K", "mode" : "blend" } ]

# Required source parameters

Bias correction phase (with mode set to "bias") needs LAPS data and raw model data. MAE calculation (mode set to "mae") needs LAPS data, bias data, and raw model data. The actual blending operation (mode set to "blend") requires bias, mae, and current raw model data.


# Output parameters

A weighted blend is calculated from the current model data using precalculated bias correction fields and MAE fields as weights.
