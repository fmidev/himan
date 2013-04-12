/* 
 * File:   simple_packing_options.h
 * Author: partio
 *
 * Created on April 11, 2013, 2:21 PM
 */

#ifndef SIMPLE_PACKING_OPTIONS__H
#define	SIMPLE_PACKING_OPTIONS__H

struct simple_packing_options
{
	long bitsPerValue;
	double binaryScaleFactor;
	double decimalScaleFactor;
	double referenceValue;
	size_t N;
};

#endif	/* SIMPLE_PACKING_OPTIONS__H */

