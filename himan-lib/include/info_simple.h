/**
 * @file info_simple.h
 *
 */

#ifndef INFO_SIMPLE_H
#define INFO_SIMPLE_H

#ifdef HAVE_CUDA

#include "himan_common.h"

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

class simple_packed;

struct info_simple
{
	// grid size
	size_t size_x;
	size_t size_y;

	// first point of grid
	double first_lat;
	double first_lon;

	// only for rotated latlon
	double south_pole_lat;
	double south_pole_lon;

	// only stereographic projection and lambert
	double orientation;

	// only lambert
	double latin1;
	double latin2;

	// distance between two grid points, degrees or meters (stereographic))
	double di;
	double dj;

	bool j_scans_positive;

	double* values;
	simple_packed* packed_values;

	HPGridType projection;
	HPInterpolationMethod interpolation;

	// true if area extends over zero meridian (important information in interpolation)
	bool wraps_globally;

	info_simple()
	    : size_x(0),
	      size_y(0),
	      first_lat(MissingDouble()),
	      first_lon(MissingDouble()),
	      south_pole_lat(MissingDouble()),
	      south_pole_lon(MissingDouble()),
	      orientation(MissingDouble()),
	      latin1(MissingDouble()),
	      latin2(MissingDouble()),
	      di(MissingDouble()),
	      dj(MissingDouble()),
	      j_scans_positive(true),
	      values(0),
	      packed_values(0),
	      projection(kUnknownGridType),
	      interpolation(kBiLinear),
	      wraps_globally(false)
	{
	}
};
}

#endif  // /* HAVE_CUDA */

#endif /* INFO_SIMPLE_H */
