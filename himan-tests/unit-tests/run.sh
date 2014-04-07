set -x

export LD_LIBRARY_PATH=$HOME/workspace/himan-plugins/build/debug:$LD_LIBRARY_PATH

path=$1

if [ $? -eq 0 ]; then
	for i in $(find $path -maxdepth 1 -type f -executable); do 
		$i --build_info --log_level=test_suite --show_progress
	
		if [ $? -ne 0 ]; then
			exit $?
		fi
	done
fi
