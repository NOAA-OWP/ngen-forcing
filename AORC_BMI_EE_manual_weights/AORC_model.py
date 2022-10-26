# load python C++ binds from ExactExtract module library
from exactextract import GDALDatasetWrapper, GDALRasterWrapper, Operation, MapWriter, FeatureSequentialProcessor, GDALWriter
# must import gdal to properly read and partiton rasters
# from AORC netcdf files
from osgeo import gdal
import pandas as pd
import numpy as np
import os
import wget

class ngen_AORC_model():
    # TODO: refactor the bmi_model.py file and this to have this type maintain its own state.
    #def __init__(self):
    #    super(ngen_model, self).__init__()
    #    #self._model = model

    def run(self, model: dict, dt: int, date, base_url, aorc_beg, aorc_end, aorc_new_end, AORC_met_vars, scale_factor, add_offset, hyfabfile, EE_weights, EE_data, AORC_data):
        """
        Run this model into the future.

        Run this model into the future, updating the state stored in the provided model dict appropriately.

        Note that the model assumes the current values set for input variables are appropriately for the time
        duration of this update (i.e., ``dt``) and do not need to be interpolated any here.

        Parameters
        ----------
        model: dict
            The model state data structure.
        dt: int
            The number of seconds into the future to advance the model.

        Returns
        -------
    
        """

        # Get current datetime stamp
        current_time = date + pd.TimedeltaIndex(np.array([model['current_model_time']],dtype=float),'s')

        # Create AORC url link and filename from ERRDAP server based on current timestamp
        # and flag for file extenstion difference based on year of timestamp
        if(current_time.year[0] > 2019):
            AORC_url_link = base_url + current_time.strftime("%Y%m")[0] +"/" + aorc_beg + current_time.strftime("%Y%m%d%H")[0] + aorc_new_end
            AORC_file = aorc_beg + current_time.strftime("%Y%m%d%H")[0] + aorc_new_end
        else:
            AORC_url_link = base_url + current_time.strftime("%Y%m")[0] +"/" + aorc_beg + current_time.strftime("%Y%m%d%H")[0] + aorc_end
            AORC_file = aorc_beg + current_time.strftime("%Y%m%d%H")[0] + aorc_end
     
        # Download the AORC forcing file for current timestamp using wget 
        filename = wget.download(AORC_url_link,bar=None)

        # load AORC netcdf file into gdal dataframe to
        # partition out meterological variables into rasters
        AORC_ncfile = gdal.Open(AORC_file)

        # Get gdal sub-datasets, which will seperate each AORC
        # variable into their own raster wrapper
        nc_rasters = AORC_ncfile.GetSubDatasets()

        # Copy initalized EE weights dataframe and
        # allocated data array from main thread
        EE_results = EE_weights.copy()
        EE_data_sum = EE_data.copy()
        # loop over each meteorological variable and get data
        # from each gdal raster to manually regrid
        for i in np.arange(len(AORC_met_vars)):
            # Get variable name in netcdf file
            variable = nc_rasters[i][0].split(":")[-1]
            # Get the gdal netcdf syntax for netcdf variable
            # Example syntax: 'NETCDF:"AORC-OWP_2012050100z.nc4":APCP_surface'
            nc_dataset_name = nc_rasters[i][0]
            # Grab data from each netcdf raster to thread AORC data array
            AORC_data[i,:,:] = gdal.Open(nc_dataset_name).ReadAsArray()
        # Now loop through EE weights and raster indices and
        # calculate coverage fraction summation
        for row in zip(EE_results.index, EE_results['row'], EE_results['col'], EE_results['coverage_fraction']):
            # Loop over each AORC met variable and calculate
            # coverage fraction summation (value * coverage fraction)
            for var in np.arange(len(AORC_met_vars)):
                EE_data_sum[var,row[0]] += (AORC_data[var,row[1],row[2]]*row[3])
        # Once summation is finished for all met variables
        # then we groupby the catchment ids and  calculate
        # coverage fraction weighted mean (summation/coverage fraction total)
        # over each met variable
        var_loop = 0
        for var in AORC_met_vars:
            EE_results[var] = EE_data_sum[var_loop,:]
            var_loop += 1
        EE_results = EE_results.groupby('id').sum()
        EE_results['ids'] = np.array(EE_results.index.values)
        met_loop = 0
        for var in AORC_met_vars:
            EE_results[var] = (EE_results[var]/EE_results['coverage_fraction'])*scale_factor[met_loop] + add_offset[met_loop]
            met_loop += 1
        # Now sort the dataframe columnds and drop the groupby
        # index before we return the dataframe to thread
        AORC_df = EE_results[['ids','APCP_surface','DLWRF_surface','DSWRF_surface','PRES_surface','SPFH_2maboveground','TMP_2maboveground','UGRD_10maboveground','VGRD_10maboveground']]
        AORC_df = AORC_df.reset_index(drop=True)

        # Now remove AORC netcdf file since its no longer needed
        os.remove(AORC_file)
      
        # Send regridded AORC data back to model output
        for var in AORC_df.columns:
            model[var] = AORC_df[var].values
           
        # Update the model time for next NextGen model iteration
        model['current_model_time'] = model['current_model_time'] + dt
        
        #return model
