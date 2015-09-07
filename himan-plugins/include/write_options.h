/** 
 * @file	write_options.h
 * @autho	partio
 *
 * @date	September 7, 2015, 8:03 AM
 */

#ifndef WRITE_OPTIONS_H
#define	WRITE_OPTIONS_H

namespace himan {
namespace plugin {

struct write_options {
	bool use_bitmap;
	
	write_options()
		: use_bitmap(true)
	{}
};

}
}

#endif	/* WRITE_OPTIONS_H */

