import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
from os.path import join

### Import all NextGen Met Modules #####
from NextGen_AORC_forcings_module import NextGen_Forcings_AORC #NextGen_AORC_forcings_module.py
from NextGen_GFS_forcings_module import NextGen_Forcings_GFS #NextGen_GFS_forcings_module.py
from NextGen_HRRR_forcings_module import NextGen_Forcings_HRRR #NextGen_HRRR_forcings_module.py
from NextGen_CFS_forcings_module import NextGen_Forcings_CFS #NextGen_CFS_forcings_module.py


###### NextGen lumped forcings driver #######
def NextGen_lumped_forcings_driver(output_root, start_time, end_time, met_dataset, hyfabfile, met_dataset_pathway=None, weights_file=None, netcdf=True, csv=False, bias_calibration = False, downscaling = False, CONUS=False, AnA=False, num_processes=1):

    # provide checks on user input arguements to ensure they will work with script
    if(os.path.isdir(output_root) == False):
        raise TypeError("Output root pathway is not a directory. Module is exiting.")
       
    if(os.path.isfile(hyfabfile) == False):
        raise TypeError("The specified hydrofabric file pathway is not an actual file. Module is exiting.")

    if(weights_file != None and os.path.isfile(weights_file) == False):
        raise TypeError("The specified ExactExtract coverage fraction weights file pathway is not an actual file. Module is exiting.")

    if(isinstance(netcdf,bool) == False or isinstance(csv,bool) == False or isinstance(CONUS,bool) == False or isinstance(AnA,bool) == False or isinstance(bias_calibration,bool) == False or isinstance(downscaling,bool) == False):
        raise TypeError("The variables 'netcdf, csv, weights, CONUS, bias_calibration, downscaling, AnA' all must be a boolean flag (True, False). Incorrect assignment has occured. Module is exiting")
        
    if(netcdf == False and csv == False):
        raise TypeError("User did not specify either netcdf or csv file production for NextGen AORC forcings. One of the options must be True. Module is exiting")
        
    if(isinstance(num_processes,int) == False):
        raise TypeError("User must specify a valid integer for the number of processes to run for the python module. Module is exiting")
        
    if(met_dataset.lower() != "aorc" and met_dataset.lower() != "gfs" and met_dataset.lower() != "hrrr" and met_dataset.lower() != "cfs"):
        raise TypeError("User must specify a valid meteorological dataset that can be currently used to calculate lumped forcings (AORC, GFS, CFS, or HRRR). Module is exiting")
        
    if(met_dataset_pathway == None or os.path.isdir(met_dataset_pathway) == False):
        ### Since AORC dataset doesn't have a pathway, we are checking to see if user specified AORC
        if(met_dataset.lower() != "aorc"):
            raise TypeError("Since user specified a forecast dataset(GFS/CFS/HRRR), they must specify a valid pathway to a forecast cycle. Module is exiting")
        else:
            print("Warning, you did not specify any AORC dataset pathway in the lumped forcings driver. You can only download AORC data from the ERRDAP cluster assuming you're on a National Water Center (NWC) server while executing this script. If you're not on the NWC server in this case, then this data extraction module will fail.")

    if(AnA == True and met_dataset.lower() != "hrrr"):
        raise TypeError("The Analysis and Assimilation (AnA) confiugration is only valid for the HRRR forecast module. Module is exiting")
 
    # If meterology dataset is a forecast dataset, then the assumption
    # is that we will create the NextGen forcing file for the entire 
    # forecast cycle, otherwise we need to check the validity of the
    # user start and end time for pulling AORC data
    if(met_dataset.lower() == "aorc"):
        # Check to see if pandas datetime accepts AORC start and end time
        # format, otherwise, throw error and quit module
        try:
            test = pd.period_range(start=start_time,end=end_time,freq='H')
            if(test.year.min() < 1997 or test.year.max() > 2021):
                raise ValueError("The NWC ERRDAP server only holds AORC data from 1997-02-01 to 2021-08-31. The user must specify a start and end date between this range. Module is exiting")
               
        except:
            raise TypeError("Start and/or End time syntax was in an incorrect format. The accepted format is 'YYYY-MM-DD HH:00:00'. Please redo the format for these variables. Module is exiting")

        # Flag to see if user selected CONUS options with AORC data and csv output
        # and if so, raise execption to indicate that memory storage will be an 
        # issue here for saving more than 800,000 csv files at once
        if(CONUS == True and csv == True):
            raise TypeError("The user has selected as having a NextGen CONUS geopackage file and requesting csv catchment outputs. We have negated this option becuase of memory issues with saving 800,000 + csv files to a given directory. Please just select netcdf option for the NextGen CONUS geopackage file. Module is exiting")
            

    if(met_dataset.lower() == "aorc"):
        NextGen_Forcings_AORC(output_root, met_dataset_pathway, start_time, end_time, netcdf, csv, CONUS, hyfabfile, weights_file, num_processes)
    elif(met_dataset.lower() == "gfs"):
        NextGen_Forcings_GFS(output_root, met_dataset_pathway, netcdf, csv, hyfabfile, weights_file, bias_calibration, downscaling, num_processes)
    elif(met_dataset.lower() == "cfs"):
        NextGen_Forcings_CFS(output_root, met_dataset_pathway, netcdf, csv, hyfabfile, weights_file, bias_calibration, downscaling, num_processes)
    elif(met_dataset.lower() == "hrrr"):
        NextGen_Forcings_HRRR(output_root, met_dataset_pathway, start_time, AnA, netcdf, csv, hyfabfile, weights_file, bias_calibration, downscaling, num_processes)

