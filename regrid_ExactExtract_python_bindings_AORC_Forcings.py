import numpy as np
import geopandas as gpd
import netCDF4 as nc4
import os
import glob
from pathlib import Path
from os.path import join
import argparse
import pandas as pd
from  multiprocessing import Process, Lock, Queue
import multiprocessing
import time
import datetime
import re
# load python C++ binds from ExactExtract module library
from exactextract import GDALDatasetWrapper, GDALRasterWrapper, Operation, MapWriter, FeatureSequentialProcessor, GDALWriter
# must import gdal to properly read and partiton rasters
# from AORC netcdf files
from osgeo import gdal
# import gc library to collect garabge
# and save RAM for multithreading processes
import gc

def get_date_time(path):
    """
    Extract the date-time from the file path
    """
    path = Path(path)
    name = path.stem
    date_time = name.split('.')[0]
    date_time = date_time.split('_')[1]  #this index may depend on the naming format of the forcing data
    date_time = re.sub('\D','',date_time)
    return date_time


def create_ngen_netcdf(aorc_ncfile):
    """
    Create NextGen netcdf file with specified format
    """

    # get datetime of forcing file to append
    # to ExactExtract csv output file
    start_time = get_date_time(aorc_ncfile)

    # first read AORC metadata in to save to NextGen forcing file
    ds = nc4.Dataset(aorc_ncfile)

    #create output netcdf file name
    output_path = join(forcing, "NextGen_forcing_"+start_time+".nc")

    #make the data set
    filename = output_path
    filename_out = output_path

     
    # write data to netcdf files
    filename_out = output_path
    ncfile_out = nc4.Dataset(filename_out, 'w', format='NETCDF4')

    #add the dimensions
    time_dim = ncfile_out.createDimension('time', None)
    catchment_id_dim = ncfile_out.createDimension('catchment-id', num_catchments)
    string_dim =ncfile_out.createDimension('str_dim', 1)

    # create variables
    cat_id_out = ncfile_out.createVariable('ids', 'str', ('catchment-id'), fill_value="None")
    time_out = ncfile_out.createVariable('Time', 'double', ('catchment-id','time',), fill_value=-99999)
    APCP_surface_out = ncfile_out.createVariable('RAINRATE', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    TMP_2maboveground_out = ncfile_out.createVariable('T2D', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    SPFH_2maboveground_out = ncfile_out.createVariable('Q2D', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    UGRD_10maboveground_out = ncfile_out.createVariable('U2D', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    VGRD_10maboveground_out = ncfile_out.createVariable('V2D', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    PRES_surface_out = ncfile_out.createVariable('PSFC', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    DSWRF_surface_out = ncfile_out.createVariable('SWDOWN', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    DLWRF_surface_out = ncfile_out.createVariable('LWDOWN', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)

    #set output netcdf file atributes
    varout_dict = {'time':time_out,
                   'APCP_surface':APCP_surface_out, 'DLWRF_surface':DLWRF_surface_out, 'DSWRF_surface':DSWRF_surface_out,
                   'PRES_surface':PRES_surface_out, 'SPFH_2maboveground':SPFH_2maboveground_out, 'TMP_2maboveground':TMP_2maboveground_out,
                   'UGRD_10maboveground':UGRD_10maboveground_out, 'VGRD_10maboveground':VGRD_10maboveground_out}


    #copy all attributes from input netcdf file
    for name, variable in ds.variables.items():
        if name == 'latitude' or name == 'longitude':
            pass
        else:
            varout_name = varout_dict[name]
            for attrname in variable.ncattrs():
                if name == "time" and attrname == "units":
                    #slight hack here to be compatible with current NetCDFPerFeatureDataProvider
                    # ... change it instead?
                    #see also https://www.unidata.ucar.edu/software/netcdf/time/recs.html
                    setattr(varout_name, "units", "seconds")
                    setattr(varout_name, "epoch_start", "01/01/1970 00:00:00")
                elif attrname != '_FillValue':
                    setattr(varout_name, attrname, getattr(variable, attrname))

    #drop the scale_factor and add_offset from the output netcdf forcing file attributes
    for key, varout_name in varout_dict.items():
        if key != 'time':
            try:
                del varout_name.scale_factor
            except: 
                print("No scale factor in forcing files. No keys to tweak for output netcdf")
            try:
                del varout_name.add_offset
            except:
                print("No add offset in forcing files. No keys to tweak for output netcdf")


    #####################################################################

    #set attributes for additional variables
    setattr(cat_id_out, 'description', 'catchment_id')
   
    cat_id_out[:] = cat_ids[:]
    time_out[:,:] = time[:,:]
    APCP_surface_out[:,:] = APCP_surface[:,:]
    DLWRF_surface_out[:,:] = DLWRF_surface[:,:]
    DSWRF_surface_out[:,:] = DSWRF_surface[:,:]
    PRES_surface_out[:,:] = PRES_surface[:,:]
    SPFH_2maboveground_out[:,:] = SPFH_2maboveground[:,:]
    TMP_2maboveground_out[:,:] = TMP_2maboveground[:,:]
    UGRD_10maboveground_out[:,:] = UGRD_10maboveground[:,:]
    VGRD_10maboveground_out[:,:] = VGRD_10maboveground[:,:]

    # Now close NextGen netcdf file
    # and AORC file
    ncfile_out.close()
    ds.close()

def python_ExactExtract(aorc_file):

    # load AORC netcdf file into gdal dataframe to
    # partition out meterological variables into rasters
    aorc = gdal.Open(aorc_file)

    # Get gdal sub-datasets, which will seperate each AORC
    # variable into their own raster wrapper
    nc_rasters = aorc.GetSubDatasets()

    # Initalize pandas dataframe to save results to csv file
    csv_results = pd.DataFrame([])

    # get datetime of forcing file to append
    # to ExactExtract csv output file
    date_time = get_date_time(aorc_file)

    # get seconds since AORC reference date for time array in
    # pandas dataframe
    time = np.zeros(num_catchments)
    time[:] = (pd.Timestamp(datetime.datetime.strptime(date_time,'%Y%m%d%H')) - ref_date).total_seconds()

    # We are only saving csv files currentlya as diagnostics for
    # evaluating ExactExtract python module performace
    NextGen_csv = join(exactextract_files,'NextGen_forcings_'+str(date_time)+'.csv')

    # loop over each meteorological variable and call 
    # ExactExtract to regrid raster to lumped sum for
    # a given NextGen catchment
    for i in np.arange(len(AORC_met_vars)):

        # Define gdal writer to only return ExactExtract
        # regrid results as a python dict
        writer = MapWriter()

        # Get variable name in netcdf file
        variable = nc_rasters[i][0].split(":")[-1]
        # Get the gdal netcdf syntax for netcdf variable
        # Example syntax: 'NETCDF:"AORC-OWP_2012050100z.nc4":APCP_surface'
        nc_dataset_name = nc_rasters[i][0]

        # For each AORC met variable, we must redefine the 
        # hydrofabric raster dataset to regrid forcings
        # based on user operation below
        dsw = GDALDatasetWrapper(hyfabfile)

        # Define raster wrapper for AORC meteorological variable
        # and specify nc_file attribute to be True. Otherwise,
        # this function will expect a .tif file
        rsw = GDALRasterWrapper(nc_dataset_name,nc_file=True) 

        # Define operation to use for raster
        op = Operation.from_descriptor('mean('+variable+')', raster=rsw)

        # Process the data and write results to writer instance
        processor = FeatureSequentialProcessor(dsw, writer, [op])
        processor.process()

        # convert dict results to pandas dataframe 
        results = pd.DataFrame(writer.output.items(),columns=['cat-id',variable])

        # find indices where scale factor is for AORC variable
        idx = np.where(variable==AORC_met_vars)[0][0]

        # save AORC results to pandas dataframe and account for scale factor
        # and offset of variable (if any). Set flag to just append 'cat-id'
        # and time data to just first variable in the loop to pandas dataframe
        if(i == 0):
            csv_results['cat-id'] = results['cat-id'].values
            csv_results['time'] = time
            csv_results[variable] = np.stack(results[variable].values,axis=0).flatten()*scale_factor[idx] + add_offset[idx]
        else:
            csv_results[variable] = np.stack(results[variable].values,axis=0).flatten()*scale_factor[idx] + add_offset[idx]

    # Flush changes to disk
    writer = None

    # Save pandas dataframe of AORC ExactExtract
    # regridded data to csv file for diagnostics
    csv_results.to_csv(NextGen_csv,index=False)

    # return csv regridded results to save
    # to each thread in process_sublist
    return csv_results


def process_sublist(data : dict, lock: Lock, EE_results, num: int):
    # Get number of files in each thread to loop through
    num_files = len(data["forcing_files"])    

    # Initalize pandas dataframe to save the 
    # regridded AORC ExactExtract results from
    # each AORC file we loop through
    EE_df_final = pd.DataFrame()

    for i in range(num_files):
        # extract forcing file and file index
        aorc_file = data["forcing_files"][i]
        # Call python ExactExtract routine to directly extract 
        # AORC regridded results and save to pandas dataframe
        # for each thread
        print(aorc_file)
        EE_df = python_ExactExtract(aorc_file)
        # concatenate the regridded data to threads final dataframe
        EE_df_final = pd.concat([EE_df_final,EE_df])

    # collect the garbage within the thread before sending results back to main
    # thread to maximize our RAM as much as possible
    gc.collect()

    # Put regridded results into thread queue to return to main thread
    EE_results.put(EE_df_final)

if __name__ == '__main__':
   
    #example: python code_name -i /home/jason.ducker/esmf_forcing_files_test/ExactExtract_sugar_creek -o /home/jason.ducker/esmf_forcing_files_test/ExactExtract_sugar_creek -a /apd_common/test/test_data/aorc_netcdf/AORC/2015 -f forcing_files/ -e_csv csv_files/ -c /home/jason.ducker/hydrofabric/catchment_data.geojson

    #parse the input and output root directory
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="input_root", type=str, required=True, help="The input directory with csv files")
    parser.add_argument("-o", dest="output_root", type=str, required=True, help="The output file path")
    parser.add_argument("-a", dest="aorc_netcdf", type=str, required=True, help="The input aorc netcdf files directory")
    parser.add_argument("-f", dest="forcing", type=str, required=True, help="The output forcing files sub_dir")
    parser.add_argument("-e_csv", dest="ExactExtract_csv_files", type=str, required=True, help="The output sub_dir for ExactExtract csv files created from each AORC file")
    parser.add_argument("-c", dest="catchment_source", type=str, required=True, help="The hydrofabric catchment file or data ")
    parser.add_argument("-j", dest="num_processes", type=int, required=False, default=96, help="The number of processes to run in parallel")
    args = parser.parse_args()

    #retrieve parsed values
    input_root = args.input_root
    output_root = args.output_root
    aorc_netcdf = args.aorc_netcdf
    forcing = join(output_root,args.forcing)
    exactextract_files = join(output_root,args.ExactExtract_csv_files)
    num_processes = args.num_processes
    hyfabfile = args.catchment_source

    #generate catchment geometry from hydrofabric
    cat_df_full = gpd.read_file(hyfabfile)
    h = [i for i in cat_df_full.id]
    n_cats = len(h)
    num_catchments = n_cats
    print("number of catchments = {}".format(n_cats))


    # Data paths for either sugar_creek (AOR_Charlotte) or 
    # Huc01 (AORC-OWP) files on Linux clustet
    #datafile_path = join(aorc_netcdf, "AORC_Charlotte_*.nc4")
    datafile_path = join(aorc_netcdf, "AORC-OWP_*.nc4")
    #get list of files
    datafiles = glob.glob(datafile_path)
    print("number of forcing files = {}".format(len(datafiles)))
    #process data with time ordered
    datafiles.sort()

    #prepare for processing
    num_forcing_files = len(datafiles)

    # AORC reference time to use
    ref_date = pd.Timestamp("1970-01-01 00:00:00")


    # Just take the first AORC file to get variable
    # metadata for creating NextGen netcdf files
    aorc_ncfile = datafiles[0]

    # Extract variable names from AORC netcdf data
    nc_file = nc4.Dataset(aorc_ncfile)
    # Get variable list from AORC file
    nc_vars = list(nc_file.variables.keys())
    # Get indices corresponding to Meteorological data
    indices = [nc_vars.index(i) for i in nc_vars if '_' in i]
    # Make array with variable names to use for ExactExtract module
    AORC_met_vars = np.array(nc_vars)[indices]


    # get scale_factor and offset keys if available
    # (AORC-OWP files for HUC01 scenario has this metadata)
    add_offset = np.zeros([len(AORC_met_vars)])
    scale_factor = np.zeros([len(AORC_met_vars)])
    i = 0
    for key in AORC_met_vars:
        try:
            scale_factor[i] = nc_file.variables[key].scale_factor
        except AttributeError as e:
            scale_factor[i] = 1.0
        try:
            add_offset[i] = nc_file.variables[key].add_offset
        except AttributeError as e:
            add_offset[i] = 0.0
        i += 1


    # Close netcdf file
    nc_file.close()



    #generate the data objects for child processes
    file_groups = np.array_split(np.array(datafiles), num_processes)

    process_data = []
    process_list = []
    lock = Lock()

    # Initalize thread storage to return to main program
    EE_results = Queue()

    for i in range(num_processes):
        # fill the dictionary with aorc forcing files
        data = {}
        data["forcing_files"] = file_groups[i]
      
        #append to the list
        process_data.append(data)

        p = Process(target=process_sublist, args=(data, lock, EE_results, i))

        process_list.append(p)

    #start all processes
    for p in process_list:
        p.start()

    # Before we terminate threads, aggregate thread 
    #regridded results together and save to main thread
    final_df = pd.DataFrame()
    for i in range(num_processes):
        result = EE_results.get()
        final_df = pd.concat([final_df,result])
        del(result)

    #wait for termination
    for p in process_list:
        p.join()

    # Collect garbage from main program to save RAM
    gc.collect()

    
    # Prepare AORC NextGen global arrays for ExactExtract regrid module
    # and to save data to NextGen netcdf file format
    cat_ids = np.array(h, dtype="S16")
    time = np.zeros((num_catchments,num_forcing_files))
    APCP_surface = np.zeros((num_catchments,num_forcing_files))
    DLWRF_surface = np.zeros((num_catchments,num_forcing_files))
    DSWRF_surface = np.zeros((num_catchments,num_forcing_files))
    PRES_surface = np.zeros((num_catchments,num_forcing_files))
    SPFH_2maboveground = np.zeros((num_catchments,num_forcing_files))
    TMP_2maboveground = np.zeros((num_catchments,num_forcing_files))
    UGRD_10maboveground = np.zeros((num_catchments,num_forcing_files))
    VGRD_10maboveground = np.zeros((num_catchments,num_forcing_files))

    # Get unique timestamps to loop through the regridded data
    # and extract data for global arrays to save to NextGen 
    # netcdf or return data for BMI model
    timestamps = final_df['time'].unique()

    for i in range(num_forcing_files):
        # find and slice dataframe for timestamp loop
        idx_time = final_df['time'] == timestamps[i]
        gfs_regridded_data = final_df.loc[idx_time,:]
        # only assign cat ids data once within loop
        if(i==0):
            cat_ids[:] = gfs_regridded_data['cat-id'].values
        # fill global arrays with slicaed dataframe data for timestamp
        time[:,i] = gfs_regridded_data['time'].values
        APCP_surface[:,i] = gfs_regridded_data['APCP_surface'].values
        PRES_surface[:,i] = gfs_regridded_data['PRES_surface'].values
        DLWRF_surface[:,i] = gfs_regridded_data['DLWRF_surface'].values
        DSWRF_surface[:,i] = gfs_regridded_data['DSWRF_surface'].values
        SPFH_2maboveground[:,i] = gfs_regridded_data['SPFH_2maboveground'].values
        TMP_2maboveground[:,i] = gfs_regridded_data['TMP_2maboveground'].values
        UGRD_10maboveground[:,i] = gfs_regridded_data['UGRD_10maboveground'].values
        VGRD_10maboveground[:,i] = gfs_regridded_data['VGRD_10maboveground'].values

    # remove final data to save RAM
    del(final_df)

    #generate single NextGen netcdf file from generated ExactExtract weighted csv files
    create_ngen_netcdf(aorc_ncfile)

