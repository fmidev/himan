#! /bin/sh
# file: src/shunit2_test_himan.sh

testEquality()
{
  bash /home/peramaki/workspace/himan-bin/regression/hybrid_pressure/hybrid_pressure.sh
  testbool=$?
  assertEquals $testbool 0
}

# load shunit2
. ../src/shunit2