#!/usr/bin/env python
#
# Print grib values to stdout
#

import traceback
import sys
import gribapi
import decimal

from decimal import *
from gribapi import *

def read(FILE,POINT):
	
	gribfile = open(FILE)

	print "File: " + FILE
	
	msgno = 0
	
	while True:   

		grib = grib_new_from_file(gribfile)

		if grib is None:
			break

		print "Reading msg " + str(msgno)

		values = grib_get_values(grib)

		if POINT != None:
			print str(int(POINT)) + " " + str(values[int(POINT)])
		else:
			for i in xrange(len(values)):
				value = values[i]

				print str(i) + " " + str(value)

		msgno = msgno+1
	
		grib_release(grib)
		
	gribfile.close()

def main():
	if len(sys.argv) < 2 or len(sys.argv) > 3:
		print "usage: read.py file [ point_number_to_test ]"
		sys.exit(1)

	try:

		FILE=""
		POINT=None

		if len(sys.argv) == 2:
			FILE=sys.argv[1]
		elif len(sys.argv) == 3:
			POINT=sys.argv[2]
			FILE=sys.argv[1]

		read(FILE,POINT)
	except GribInternalError,err:
		print >>sys.stderr,err.msg

	return 0

if __name__ == "__main__":
	sys.exit(main())
