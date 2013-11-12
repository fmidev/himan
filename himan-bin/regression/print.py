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
	count = grib_count_in_file(gribfile)
	
	while True:   

		grib = grib_new_from_file(gribfile)

		if grib is None:
			break

		if count > 1:
			print "Reading msg " + str(msgno)

		values = grib_get_values(grib)

		if POINT != None:
			print str(POINT) + " " + str(values[int(POINT)])
		else:
			for i in xrange(len(values)):
				value = values[i]

				print str(i) + " " + str(value)

		msgno = msgno+1
	
		grib_release(grib)
		
	gribfile.close()

def main():
	if len(sys.argv) < 2 :
		print "usage: read.py file1 [ file2 file3 ... ] [ point_number_to_test ]"
		sys.exit(1)

	try:

		arglen=len(sys.argv)

		POINT=None

		try:
			POINT = int(sys.argv[arglen-1])
		except ValueError:
			POINT = None
			pass

		i=1

		reduction=0

		if POINT != None:
			reduction=1
			
		for i in xrange(1, arglen-reduction):
			file=sys.argv[i]

			read(file,POINT)

	except GribInternalError,err:
		print >>sys.stderr,err.msg

	return 0

if __name__ == "__main__":
	sys.exit(main())
