#!/usr/bin/env bash

#python ./compare_observed.py ../../../nwmv3_oe_install/test/tmp/nwm_ana_arch_test_stofs/outputs  \
#    076,9751364,9751639,9755371,9759938,9751381,9752235,9759110,9751401,9752695,9759394,9761115 \
#    ../../../nwmv3_oe_install/test/tmp/observed_water_level ./

python ./compare_observed.py ../../../nwmv3_oe_install/test/tmp/hi_nwm_ana_arch_test_stofs_demo/outputs  \
    1611400,1612480,1617760,1612340,1615680,1612401,1617433 \
    ../../../nwmv3_oe_install/test/tmp/observed_water_level ./ hawaii
