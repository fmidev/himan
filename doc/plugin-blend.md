# Summary

Blend plugin calculates a weighted blend of different forecasts specified in its configuration.

# Required source parameters

The parameter that is specified in the configuration file is required from all source producers at
the specified levels.

# Output parameters

A weighted blend is calculated from all of the specified source producers.

The plugin outputs an ensemble consisting of the source forecasts as perturbed members and the blend as the control forecast.

# Method of calculation

```CF_Output = f[0] * w[1] + f[1] * w[1] + ... + f[n] * w[n]```

# Per-plugin configuration options

param: Specifies the parameter to be blended
weights_file: File containing whitespace separated list of weights for blending. The number of weights should match the number of input forecasts.
options: Specifies a list of producer specifications of the form: 
```{ "producer" : PROD, "geom" : GEOM, "forecast_type" : FTYPE, "leveltype" : LTYPE, "level" : L }```

Full plugin configuration example:
```
"plugins" : [ { "name" : "blend", "param" : "T-K", "weights_file" : "weights_file",
		"options" : [
			{ "producer" : "HL2", "geom" : "RCR068", "forecast_type" : "deterministic",
			  "leveltype" : "height", "level" : 0 },
			{ "producer" : "MEPS", "geom" : "MEPSNOFMI2500", "forecast_type" : "cf",
			  "leveltype" : "height", "level" : 0 } ]
		} ]
```

