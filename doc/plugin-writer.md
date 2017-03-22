# Summary

writer plugin is one of the infrastructure plugins. It is responsible for writing any resulting fields (grids) to files. Writer will also update cache and database if needed. Writer plugin does not directly write the files but will call other plugins to do that.

# Configuration options

file_write: control how resulting grids should be written.

    "file_write" : "multiple | single | database | cache only"

options are:

* single: all fields are written in a single file
* multiple: each field is written to an individual file in current work directory
* database: each field is written to an individual file to a separate directory, database metadata is updated
* cache only: fields are only written to cache