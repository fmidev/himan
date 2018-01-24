# Summary

Blend plugin calculates a weighted blend of different forecasts specified in its configuration.

# Required source parameters

The parameter that is specified in the configuration file is required from all source producers at
the specified levels.

# Output parameters

A weighted blend is calculated from all of the specified source producers.

The plugin outputs an ensemble consisting of the source forecasts as perturbed members and the blend as the control forecast.

# Method of calculation

```CF_Output = f[0] * w[0] + f[1] * w[1] + ... + f[n] * w[n]```

# Per-plugin configuration options

param: Specifies the parameter to be blended

options: Specifies a list of producer specifications of the form: 
```{ "producer" : PROD, "geom" : GEOM, "forecast_type" : FTYPE }```

Full plugin configuration example:
```
"plugins" : [ { "name" : "blend", "param" : "T-K",
		"options" : [
			{ "producer" : "HL2", "geom" : "RCR068", "forecast_type" : "deterministic" },
			{ "producer" : "MEPS", "geom" : "MEPSNOFMI2500", "forecast_type" : "cf" } ]
		} ]
```

