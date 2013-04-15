#!/bin/sh

export HIMAN=$(pwd)/../build/debug/himan
LOGDIR=/tmp

txtund=$(tput sgr 0 1)    # Underline
txtbld=$(tput bold)       # Bold
txtred=$(tput setaf 1)    # Red
txtgrn=$(tput setaf 2)    # Green
txtylw=$(tput setaf 3)    # Yellow
txtblu=$(tput setaf 4)    # Blue
txtpur=$(tput setaf 5)    # Purple
txtcyn=$(tput setaf 6)    # Cyan
txtwht=$(tput setaf 7)    # White
txtrst=$(tput sgr0)       # Text reset

for d in $(find . -maxdepth 1 -type d ! -name ".*" -print); do
	cd $d
	for s in $(ls *.sh); do
		LOGFILE=$LOGDIR/$s.log

		printf "%-10s %-20s %-45s " \
			$(basename $d) \
			$s \
			" (log: $LOGFILE)" 


		RESULT=""
		
		sh $s > $LOGFILE 2>&1
		
		ret=$?
		
		if [ $ret -ne 0 ]; then
		        RESULT="[${txtred}FAILED${txtrst}]"
		else
			RESULT="[${txtgrn}SUCCESS${txtgrn}]"
		fi
		
	        printf "%-15s\n" $RESULT
		
		if [ $ret -ne 0 ]; then
			exit 1
		fi
	done
	cd ..
		
done
