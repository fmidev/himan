# Summary

Cache plugin is one of the infrastructure plugins. It stores all read and written fields to memory for further use. It has the ability to evict data with an LRU algorithm when the cache has grown over a specified limit. The default behavior is to store everything without eviction. When plugins are requesting data, cache is the first place they will check.

The fields are stored as shared pointers, and each field will have a label associated which will be matched to incoming data requests.

The cache has the ability to store data in different data types, for example float or double. If data is found from cache in a different data type than what was requested, plugin will do a conversion *unless* strict mode has been enabled. In the latter case cache plugin will return nil and other data sources must be searched.

# Configuration options

Cache can be turned off with configuration file option `use_cache`. Default value is `true`.
