#!/usr/bin/env python

import traceback
import sys
import gribapi
import decimal

from decimal import *
from gribapi import *

if len(sys.argv) < 3:
	print "usage: test.py firstfile secondfile"
	sys.exit(1)

FIRST_FILE=sys.argv[1]
SECOND_FILE=sys.argv[2]

SECOND_VALUE_SCALE = 1
SECOND_VALUE_BASE = 0

MAX_DIFF = 1
ALLOWED_ERRORS=50

def read():
	
	firstfile = open(FIRST_FILE)
	secondfile = open(SECOND_FILE)

	print "First file: " + FIRST_FILE
	print "Second file: " + SECOND_FILE
	
	msgno = 0
	max_found_diff = 0
	grid_point_of_max_diff = 0
	msg_of_max_diff = 0

	e = abs(Decimal(str(MAX_DIFF)).as_tuple().exponent)
	
	while True:   

		firstgrid = grib_new_from_file(firstfile)
		secondgrid = grib_new_from_file(secondfile)

		if firstgrid is None or secondgrid is None:
			break

		print "Reading msg " + str(msgno)

		firstvalues = grib_get_values(firstgrid)
		secondvalues = grib_get_values(secondgrid)

		if len(firstvalues) != len(secondvalues):
			print "ERROR: grib data lengths are not the same"
			sys.exit(1)

		errors=0

		for i in xrange(len(firstvalues)):
			firstvalue = firstvalues[i]
			secondvalue = secondvalues[i]

			secondvalue = SECOND_VALUE_BASE + SECOND_VALUE_SCALE * secondvalue
	
			diff = abs(secondvalue-firstvalue)
			diff = round(diff, e)

			if diff > MAX_DIFF:
				print "Max diff (" + str(MAX_DIFF) + ") exceeded at grid point: " + str(i) + " (#" + str(i+1) + ") first value: " + str(firstvalue) + ", secondvalue: " + str(secondvalue) + " diff: " + str(diff)
				if ALLOWED_ERRORS > 0:
					errors = errors+1
					if errors > ALLOWED_ERRORS:
						print "Too many errors, exiting"
						sys.exit(1)
				else:
					sys.exit(1)

			if diff > max_found_diff:
				max_found_diff = diff
				grid_point_of_max_diff = i
				msg_of_max_diff = msgno
		
		msgno = msgno+1
	
		grib_release(firstgrid)
		grib_release(secondgrid)	 
		
		print "Grib data is equal enough (largest diff allowed: " + str(MAX_DIFF) + ", largest diff found: " + str(max_found_diff) + ", gridpoint " + str(grid_point_of_max_diff) + ", message " + str(msg_of_max_diff) + " allowed errors " + str(errors) + "/" + str(ALLOWED_ERRORS) + ")"
	
	firstfile.close()
	secondfile.close()

def main():
	try:
		read()
	except GribInternalError,err:
		print >>sys.stderr,err.msg

	return 0

if __name__ == "__main__":
	sys.exit(main())
