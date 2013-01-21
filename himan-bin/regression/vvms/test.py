import traceback
import sys
import gribapi

from gribapi import *

HIL_PP=sys.argv[1]
HIMAN=sys.argv[2]

HIMAN_VALUE_SCALE = 1
MAX_DIFF = 0.01

def example():
    hilfile = open(HIL_PP)
    himfile = open(HIMAN)
    
    hilgrid = grib_new_from_file(hilfile)
    himgrid = grib_new_from_file(himfile)

    hilvalues = grib_get_values(hilgrid)
    himvalues = grib_get_values(himgrid)

    if len(hilvalues) != len(himvalues):
    	print "ERROR: grib data lengths are ntot the same"
	sys.exit(1)
	
    max_found_diff = 0
    grid_point_of_max_diff = 0
        
    for i in xrange(len(hilvalues)):
    	hilvalue = hilvalues[i]
	himvalue = himvalues[i]
	
	himvalue = HIMAN_VALUE_SCALE * himvalue
    
        diff = abs(himvalue-hilvalue)
	
	if diff > MAX_DIFF:
	    print "Max diff exceeded: " + str(MAX_DIFF)
	    print "Grid point: " + str(i) + " hilvalue: " + str(hilvalue) + ", himvalue: " + str(himvalue) + " diff: " + str(diff)
	    sys.exit(1)
	    
        if diff > max_found_diff:
	     max_found_diff = diff
	     grid_point_of_max_diff = i
	     
    print "Grib data is equal enough (largest diff allowed: " + str(MAX_DIFF) + ", largest diff found: " + str(max_found_diff) + ", gridpoint " + str(grid_point_of_max_diff) + ")"
    
    grib_release(hilgrid)
    grib_release(himgrid)
    
    hilfile.close()
    himfile.close()

def main():
    try:
        example()
    except GribInternalError,err:
        if VERBOSE:
            traceback.print_exc(file=sys.stderr)
        else:
            print >>sys.stderr,err.msg

        return 1

if __name__ == "__main__":
    sys.exit(main())
