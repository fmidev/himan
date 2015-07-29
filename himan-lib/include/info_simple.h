/**
 * @file info_simple.h
 * @author partio
 *
 * @date January 27, 2014, 7:22 AM
 */

#ifndef INFO_SIMPLE_H
#define	INFO_SIMPLE_H

#ifdef HAVE_CUDA

#include "simple_packed.h"

namespace himan
{

/**
 * @brief Dumbed-down version of info-class
 *
 * Cuda has two big drawbacks (at least until and including version 5.5):
 *
 * 1) It doesn't support C++11
 * 2) The memory management is, how to say it, anal.
 *
 * Because of reason 1) we have to make a dumbed-down version of info class
 * since that class is using C++11 features. Because of reason 2) we have to
 * manage some of the memory ourselves, both at the host AND at the device.
 * Also, the driver does not know how to distinguish host and device memory
 * which means this responsibility is left to the programmer.
 *
 * So the MO when calculating on Cuda is:
 *
 * 1) Make an info_simple struct for each info class instance, for both source
 *    and target data.
 * 2) If source data is packed, assign the pre-allocated unpacked data pointer to
 *    struct and register it as page-locked.
 * 3) In .cu, allocate device memory, unpack source data if needed and transfer
 *    unpacked data to host. If source data is already unpacked, transfer that
 *    to device.
 * 4) Start kernel and pass 'options' struct BY VALUE. This means of course that
 *    the info_simple pointers in that struct cannot be used as the point to
 *    host memory. This is intentional.
 * 5) After calculation transfer results back to host (to the page-locked memory).
 *    Release device memory. Since the pointer to unpacked values came from the
 *    info class vector, no memcpy inside host memory is necessary.
 * 6) Unregister page-locked memory.
 */
	
struct info_simple
{

	size_t size_x;
	size_t size_y;

	double first_lat;
	double first_lon;

	// rotated latlon
	double south_pole_lat;
	double south_pole_lon;

	// stereographic projection
	double orientation;
	
	double di;
	double dj;

	bool j_scans_positive;
	bool is_page_locked;

	double *values;
	simple_packed* packed_values;

	std::string param;

	size_t missing;

	HPProjectionType projection;
	
	info_simple()
		: size_x(0)
		, size_y(0)
		, j_scans_positive(true)
		, is_page_locked(false)
		, values(0)
		, packed_values(0)
		, missing(0)
		, projection(kUnknownProjection)
	{}

};

}

#endif // /* HAVE_CUDA */

#endif	/* INFO_SIMPLE_H */

