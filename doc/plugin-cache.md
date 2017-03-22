# Summary

cache plugin is one of the infrastructure plugins. It stores all read and written fields to memory for further use. It has the ability to evict data with an LRU algorithm when the cache has grown over a specified limit. The default behavior is to store everything without eviction. When plugins are requesting data, cache is the first place they will check. Cache is immutable.

The fields are stored as shared_ptr's, and each field will have a label associated which will be matched to incoming cache requests. When a field is returned from cache, it is copied to the caller so that cache contents are immutable.

# Configuration options

cache can be turned off with configuration file option `use_cache`. Default value is true.