#!/bin/bash

###############################################################################
#  File name: bsub_timeslice.bash                                             #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@noaa.gov)                          #
#                                                                             #
#  Initial version date:                                                      #
#                                                                             #
#  Last modification date:  7/12/2017                                         #
#                                                                             #
#  Description: Make calls to the time slice making program and submit the    #
#               job to the dev_shared queue                                   #
#                                                                             #
###############################################################################

if [ ! -e "$HOME/usgs_download/time_slices" ]; then mkdir -p $HOME/usgs_download/time_slices; fi

if [ ! -e "$HOME/usgs_download/LOGS" ]; then mkdir -p $HOME/usgs_download/LOGS; fi

cd $HOME/usgs_download/LOGS

if [ -e "time_slices.bsub" ]; then rm -f "time_slices.bsub";  fi

cat > time_slices.bsub << EOF
#!/bin/bash
#BSUB -J time_slice_for_da    # Job name
#BSUB -o time_slice_for_da.out                     # output filename
#BSUB -e time_slice_for_da.err                     # error filename
#BSUB -L /bin/sh                     # shell script
#BSUB -q "dev_shared"                       # queue
#BSUB -W 00:30                       # wallclock time - timing require to complete run
#BSUB -P "OHD-T2O"                   # Project name
#BSUB  -R 'span[ptile=10]'
#BSUB -M 2048                where ## is memory in MB

source $HOME/.bashrc

find $HOME/usgs_download/time_slices -mtime +5 -exec rm {} \;

$HOME/usgs_download/analysis/make_time_slice_from_usgs_waterml.py -i $1 -o $HOME/usgs_download/time_slices > $HOME/usgs_download/LOGS/time_slices.log
EOF

bsub  < time_slices.bsub
