/**
 * @file info_simple.h
 * @author partio
 *
 * @date January 27, 2014, 7:22 AM
 */

#ifndef INFO_SIMPLE_H
#define	INFO_SIMPLE_H

#include "simple_packed.h"
#include "cuda_helper.h"

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
 * 2) If source data is packed, allocate page-locked memory where the unpacked
 *    data can be transferred.
 * 3) In .cu, allocate device memory, unpack source data if needed and transfer
 *    unpacked data to host. If source data is already ubpacked, transfer that
 *    to device.
 * 4) Start kernel and pass 'options' struct BY VALUE. This means of course that
 *    the info_simple pointers in that struct cannot be used as the point to
 *    host memory. This is intentional.
 * 5) After calculation transfer results back to host (to the page-locked memory).
 *    Release device memory.
 * 6) Copy results to info-class memory and free page-locked memory.
 *
 * Step 6 might seem a bit odd but it is necessary for two reasons:
 *
 * 1) We could transfer the data directly from device to info-class but that would
 *    mean that the info class should use page-locked memory (or be prepared for
 *    very slow memory copies).
 * 2) Page locked memory is scarce: it is possible that if we use page-locked
 *    memory for all himan-operations we might run out of it; simple copy of
 *    data from one memory location to another is instead quite fast and can
 *    is easily optimized by the compiler.
 *
 *
 */
	
struct info_simple
{

	size_t size_x;
	size_t size_y;

	double first_lat;
	double first_lon;

	double south_pole_lat;
	double south_pole_lon;

	double di;
	double dj;

	bool j_scans_positive;
	bool is_page_locked;

	double *values;
	simple_packed* packed_values;

	std::string param;

	info_simple()
		: size_x(0)
		, size_y(0)
		, j_scans_positive(true)
		, is_page_locked(false)
		, values(0)
		, packed_values(0)
	{}

	bool create()
	{

		if (size_x == 0 || size_y == 0)
		{
			return false;
		}
		
		if (!values)
		{
			is_page_locked = true;
			CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**> (&values), size_x * size_y * sizeof(double)));
		}

		return true;
	}

	void free_values()
	{
		// We don't destroy packed_values since that's the responsibility of
		// the real info class

		// Also if 'values' was not allocated from page-locked memory, do nothing
		
		if (is_page_locked && values)
		{
			CUDA_CHECK(cudaFreeHost(values));
		}
	}

};

/*struct info_simple_list
{
	std::vector<info_simple*> list;

	info_simple_list()
	{}

	info_simple_list (const info_simple_list& other)
	{
		list.resize(other.list.size());

		for (size_t i = 0; i < list.size(); i++)
		{
			// shallow copy
			list[i] = new info_simple(*other.list[i]);
		}
	}
};*/

}

#endif	/* INFO_SIMPLE_H */

