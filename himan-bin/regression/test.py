import traceback
import sys
import gribapi
import decimal

from decimal import *
from gribapi import *

HIL_PP=sys.argv[1]
HIMAN=sys.argv[2]

HIMAN_VALUE_SCALE = 1
MAX_DIFF = 0.01

def read():
    
    hilfile = open(HIL_PP)
    himfile = open(HIMAN)
    
    msgno = 0
    max_found_diff = 0
    grid_point_of_max_diff = 0
    msg_of_max_diff = 0

    e = abs(Decimal(str(MAX_DIFF)).as_tuple().exponent)
	
    while True:   

        hilgrid = grib_new_from_file(hilfile)
        himgrid = grib_new_from_file(himfile)

        if hilgrid is None or himgrid is None:
            break

        print "Reading msg " + str(msgno)

        hilvalues = grib_get_values(hilgrid)
        himvalues = grib_get_values(himgrid)

        if len(hilvalues) != len(himvalues):
    	    print "ERROR: grib data lengths are not the same"
	    sys.exit(1)

        for i in xrange(len(hilvalues)):
            hilvalue = hilvalues[i]
	    himvalue = himvalues[i]

            himvalue = HIMAN_VALUE_SCALE * himvalue
    
            diff = abs(himvalue-hilvalue)
	    diff = round(diff, e)

            if diff > MAX_DIFF:
                print "Max diff exceeded: " + str(MAX_DIFF)
	        print "Grid point: " + str(i) + " hilvalue: " + str(hilvalue) + ", himvalue: " + str(himvalue) + " diff: " + str(diff)
	        sys.exit(1)
	    
            if diff > max_found_diff:
	        max_found_diff = diff
	        grid_point_of_max_diff = i
		msg_of_max_diff = msgno
		
        msgno = msgno+1
	
        grib_release(hilgrid)
        grib_release(himgrid)     
        
    print "Grib data is equal enough (largest diff allowed: " + str(MAX_DIFF) + ", largest diff found: " + str(max_found_diff) + ", gridpoint " + str(grid_point_of_max_diff) + ", message " + str(msg_of_max_diff) + ")"
    
          
    hilfile.close()
    himfile.close()

def main():
    try:
        read()
    except GribInternalError,err:
        print >>sys.stderr,err.msg

    return 1

if __name__ == "__main__":
    sys.exit(main())
