#!/bin/sh

export HIMAN=$(pwd)/../build/debug/himan

for d in $(find . -maxdepth 1 -type d ! -name ".*" -print); do
	echo -- testing $d
	cd $d
	for s in $(ls *.sh); do
		echo running script $s
		sh $s
	done
	cd ..
		
done
