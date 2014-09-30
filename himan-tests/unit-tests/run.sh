set -x

if [ -z "$HIMAN_LIBRARY_PATH" ]; then
	export LD_LIBRARY_PATH=/usr/lib64/himan-plugins:$LD_LIBRARY_PATH
else
	export LD_LIBRARY_PATH=$HIMAN_LIBRARY_PATH:$LD_LIBRARY_PATH
fi

if [ -z "$1" ]; then
	path="build/debug"
else
	path=$1
fi

if [ $? -eq 0 ]; then
	for i in $(find $path -maxdepth 1 -type f -executable); do 
		$i --build_info --log_level=test_suite --show_progress
	
		if [ $? -ne 0 ]; then
			exit $?
		fi
	done
fi
