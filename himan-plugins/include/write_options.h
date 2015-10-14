/** 
 * @file	write_options.h
 * @autho	partio
 *
 * @date	September 7, 2015, 8:03 AM
 */

#ifndef WRITE_OPTIONS_H
#define	WRITE_OPTIONS_H

#include <plugin_configuration.h>

namespace himan {
namespace plugin {

struct write_options {
	std::shared_ptr<const plugin_configuration> configuration;
	bool use_bitmap;
	HPPackingType packing_type; // for grib, ie simple_packing, jpeg_packing
	
	write_options()
		: use_bitmap(true)
		, packing_type(kSimplePacking)
	{}
};

}
}

#endif	/* WRITE_OPTIONS_H */

