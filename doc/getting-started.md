# Getting started

Himan can be either built from source, or installed using pre-built rpm packages (https://download.fmi.fi/himan). Latter is recommended for a quick start. See also [using Docker images](#Using_Docker_images).

In operative environments Himan relies heavily on a database that's providing all data and metadata. This database schema will be open sourced later this year. In the meanwhile, Himan can be tested using a "no database" mode.

Example: running seaicing plugin for Hirlam data. Seaicing plugin calculates an index that describes the amount of ice that is built up on a ship's superstructure. Files for this example are located at example/seaicing.

## Define a json configuration

```
{
        "bbox" : "5,45,30,65",
        "scanning_mode" : "+x+y",
        "ni" : "150",
        "nj" : "150",
        "source_producer" : "999999",
        "target_producer" : "999999",
        "hours" : "3",
        "origintime" : "2017-04-05 00:00:00",
        "file_write" : "multiple",

        "processqueue" : [
        {
                "leveltype" : "height",
                "levels" : "0",
                "plugins" : [ { "name" : "seaicing" } ]
        }
        ]
}
```

Breaking the configuration into pieces:

```
        "bbox" : "5,45,30,65",
        "scanning_mode" : "+x+y",
        "ni" : "150",
        "nj" : "150",
```

The target area for the calculation has lower-left longitude 5 degrees and latitude 10 degrees. The corresponding top-right coordinates are 30,65. Scanning mode for this grid is +x+y, which means that reading starts from the bottom left corner and goes right and up. The grid for this area has size 150x150 grid points. Source area does not need to be defined: Himan will determine it from the source file and will interpolate all data to target area and grid. Himan supports a few of the most common projections used in the meteorological domain. In operational environments it might be prudent (and faster!) to produce the data in the same grid as the source data.


```
        "source_producer" : "999999",
        "target_producer" : "999999",
```

Both source and target producer id's are 99999. This is corresponds to missing value: because the example is run using no-database mode, Himan is not able to fetch the real producer ids.

```
        "hours" : "3",
        "origintime" : "2017-04-05 00:00:00",
```

The analysis time (or origin time) for the forecast is 5th of April, 2017 00z. The calculation is done for forecast hour (leadtime) 3, ie. valid time is 2017-04-05 03:00:00.

```
        "file_write" : "multiple",

```

Output data is written so that each field is written to a separate file. In the context of this example this does not really matter because the calculation outputs only one field. The default output file type is Grib.

```

        "processqueue" : [
        {
                "leveltype" : "height",
                "levels" : "0",
                "plugins" : [ { "name" : "seaicing" } ]
        }
        ]
```

The processqueue (list of plugins that are executed) consists only one plugin: seaicing. The resulting data is written to leveltype height/0.

## Run Himan

```
$ himan -f seaicing.json --no-database --param-file param-file.txt seaicing.grib

************************************************
* By the Power of Grayskull, I Have the Power! *
************************************************

Info::himan Found 46 plugins
Debug::himan Processqueue size: 1
Info::himan Calculating seaicing
Info::compiled_plugin: Thread 1 starting
Info::seaicingThread #1 Calculating time 201704050300 level height/0
Debug::fetcher Start full auxiliary files read
Debug::grib Read file 'seaicing.grib' (36 MB/s)
Debug::fetcher Auxiliary files read finished, cache size is now 3
Info::seaicingThread #1 [CPU] Missing values: 0/22500
Info::grib Wrote file './ICING-N_height_0_ll_150_150_0_003.grib' (1 MB/s)

```

## Check contents

```
$ grib_histogram ICING-N_height_0_ll_150_150_0_003.grib

min=0 max=3 size=22500
 0:0.3 0.3:0.6 0.6:0.9 0.9:1.2 1.2:1.5 1.5:1.8 1.8:2.1 2.1:2.4 2.4:2.7 2.7:3
 21606 0 0 883 0 0 10 0 0 1
```

Note that seaicing plugin does not separate land points from sea points: the index is mostly zero due to warm conditions on the baltic sea.

<a name="Using_Docker_images"></a>

# Using Docker images

For a really quick start, the example case can be run using a Dockerfile. Note that as Himan is not an interactive program and not a daemon, docker is generally not very suitable for running Himan, but for quick testing it works ok.


```
$ docker build -t himan_test .
$ docker run -v /tmp:/tmp himan_test
$ grib_histogram /tmp/ICING-N_height_0_ll_150_150_0_003.grib
min=0 max=3 size=22500
 0:0.3 0.3:0.6 0.6:0.9 0.9:1.2 1.2:1.5 1.5:1.8 1.8:2.1 2.1:2.4 2.4:2.7 2.7:3
 21606 0 0 883 0 0 10 0 0 1
```