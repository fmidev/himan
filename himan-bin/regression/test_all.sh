#!/bin/sh

export HIMAN=$(pwd)/../build/debug/himan
LOGDIR=/tmp

for d in $(find . -maxdepth 1 -type d ! -name ".*" -print); do
	echo -- testing $d
	cd $d
	for s in $(ls *.sh); do
		LOGFILE=$LOGDIR/$s.log

		echo running script $s, log at \'$LOGFILE\'
		sh $s > $LOGFILE 2>&1

		if [ $? -ne 0 ]; then
			echo test failed
		else
			echo test success
		fi
	done
	cd ..
		
done
