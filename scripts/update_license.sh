#!/bin/bash

licenseheaders -t ./license_template -vv -cy -d ../src
licenseheaders -t ./license_template -vv -cy -d ../cmake --additional-extensions cmake=.cmake,.cmake.in -x "*/Catch.cmake" "*/CatchAddTests.cmake" "*/ParseAndAddCatchTests.cmake"
licenseheaders -t ./license_template -vv -cy -d ../tests
licenseheaders -t ./license_template -vv -cy -d ../examples
licenseheaders -t ./license_template -vv -cy -f ../CMakeLists.txt
