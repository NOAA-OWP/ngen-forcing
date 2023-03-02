import netCDF4 as nc
import pandas as pd
import numpy as np
import os
import wget

class ngen_AORC_model():
    # TODO: refactor the bmi_model.py file and this to have this type maintain its own state.
    #def __init__(self):
    #    super(ngen_model, self).__init__()
    #    #self._model = model

    def run(self, model: dict, dt: int, date, base_url, aorc_beg, aorc_end, aorc_new_end, AORC_met_vars, scale_factor, add_offset, missing_value):
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

        # load AORC netcdf file
        AORC_ncfile = nc.Dataset(AORC_file)

        # Send regridded AORC data back to model output
        i = 0
        for var in AORC_met_vars:           
            met_data = np.where(AORC_ncfile.variables[var][:].data == missing_value[i], np.nan, AORC_ncfile.variables[var][:].data) 
            print(met_data)
            model[var] = met_data *scale_factor[i] + add_offset[i]
            i += 1

        # Update the model time for next NextGen model iteration
        model['current_model_time'] = model['current_model_time'] + dt

        # Now remove AORC netcdf file since its no longer needed
        os.remove(AORC_file)
      
        #return model
