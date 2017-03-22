# Summary

radon plugin is one of the infrastructure plugins. It is responsible for accessing the radon database for both reading and writing.

Database password is given as an environment variable.

# Configuration options

Himan executable has command line switch ```-R``` which can be used to force database to radon, default value is to two databases (Oracle and Postgres based) but this functionality is deprecated and will be removed in the near future. At the same time switch ```-R``` will be removed.