# Overview
This directory contains python scripts to compare the simulated water levels, that is the SCHISM outputs, with observed values.

# Prerequisite
1) SCHISM output directory that contains the output files. The filename of the SCHISM output has the pattern of out2d_*.nc where * is a number such as 1, 2, 3, .... See the README.md in the 'calib' directory about running SCHISM simulations using respective or archived NWM v3 data.
2) Downloaded observed water level json files from NOAA. See the README.md file in the 'Observed_Waterlevel_Scripts' directory about downloading observed data for a given location and time period.


# Script description
compare_observed.py : This is the main driver script to create comparison plots and observed and simulated time series files. 
NOAAObservedWL.py : This script manages an observed water level json for a station.
SCHISMOutput.py   : This script manages a SCHISM output file 
plot_water_level.gp : This is a sample Gnuplot script to create interactive figures using the output time series from the main driver script, compare_observed.py.


# Script Usage
The compare_observed.py takes 4 arguments.

usage: compare_observed.py [-h] schism_output_dir station_list observed_wl_dir output_dir

positional arguments:
  schism_output_dir  schism output directory
  station_list       list of the observed station IDs
  observed_wl_dir    Observed water level files
  output_dir         output directory for the comparison figures

optional arguments:
  -h, --help         show this help message and exit

The first argument, schism_output_dir, is the output directory of the SCHISM simulation. Normally, this directory is named 'outputs' under the SCHISM working directory.
The second argument, station_list, contains a list of observed station IDS separated by a comma, ','. 
The third argument, observed_wl_dir, is the directory where the observed water level json files are downloaded.
The forth argument, output_dir, specifies the directory for the output files.  There are three types of output files, *.png, *_obv.txt, and *_sim.txt. The *.png files are the image files for each station that compare the simulated values with observed values in the same plot. The *_obv.txt and *_sim.txt files are text files. Each contains a time series of time-value pairs for one station. The *_obv.txt has the observed time series, while the *_sim.txt has the simulated time series.

#### Examples ####

cd ngen-forcing/coastal/compare_observed
python ./compare_observed.py ../../../nwm_ana_arch_test_stofs/outputs  \
    9751364,9751639,9755371,9759938,9751381,9752235,9759110,9751401,9752695,9759394,9761115 \
    ../../../observed_water_level ./

#### Output ####
Check the output files, 
ls *.png
ls *_obv.txt
ls *_sim.txt

#### Interactive plot with Gnuplot ####

After the output files has been created by running the ./compare_observed.py script, the Gnuplot script, plot_water_level.gp, can be used to create interactive plot in a desktop environment. It assumes that gnuplot v5.2 or higher has been installed on the system. It uses the output files (the *_obv.txt and *_sim.txt files) from ./compare_observed.py as inputs. First, start the Gnuplot program by issuing the 'gnuplot' command at the prompt.

gnuplot

Then in the gnuplot environment, use the 'call' command to execute the gnuplot script. The gnuplot script needs two arguments, the first is the observed time series file, and the second argument is the simulated time series file, for example,

gnuplot> call "plot_water_level.gp" "./1612340_obv.txt" "./1612340_sim.txt"

Note: the quotation mark "" around the script and the filenames are necessary.


#### Other visualization programs ####
Because the observed and simulated time series for each station and corresponding node have been written as text files, users can use other programs at their preference to visualize the calibration results.

