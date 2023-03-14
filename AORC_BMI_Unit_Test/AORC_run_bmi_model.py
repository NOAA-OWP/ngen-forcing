from pathlib import Path

import numpy as np

# This is the BMI LSTM that we will be running
from AORC_bmi_model import AORC_bmi_model


def execute():
    # creating an instance of a model
    print('creating an instance of the  AORC_BMI_MODEL model object')
    model = AORC_bmi_model()

    # Initializing the BMI
    print('Initializing the AORC Forcings BMI')
    current_dir = Path(__file__).parent.resolve()
    model.initialize(bmi_cfg_file_name=str(current_dir.joinpath('config.yml')))

    # Now loop through the time steps, update the AORC Forcings model, and set output values
    print('Now loop through the time steps, run the AORC forcings model (update), and set output values')
    print('\n')
    print('model time', 'RAINRATE', 'T2D', 'Q2D', 'U2D', 'V2D', 'PSFC', 'SWDOWN', 'LWDOWN')
    for x in range(10):

        #########################################
        # UPDATE THE MODEL WITH THE NEW INPUTS ##
        model.update()     ######################
        #########################################

        # PRINT THE MODEL RESULTS FOR THIS TIME STEP#################################################
        #print('{:.2f}, {s}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(model.get_current_time(),
        #                                                         model.get_value_ptr('ids'),
        #                                                         model.get_value_ptr('RAINRATE'),
        #                                                         model.get_value_ptr('T2D'),
        #                                                         model.get_value_ptr('Q2D'),
        #                                                         model.get_value_ptr('U2D'),
        #                                                         model.get_value_ptr('V2D'),
        #                                                         model.get_value_ptr('PSFC'),
        #                                                         model.get_value_ptr('SWDOWN'),
        #                                                         model.get_value_ptr('LWDOWN')))

        print(model.get_current_time(),np.nanmean(model.get_value_ptr('APCP_surface')[:]),np.nanmean(model.get_value_ptr('TMP_2maboveground')[:]),np.nanmean(model.get_value_ptr('SPFH_2maboveground')[:]),np.nanmean(model.get_value_ptr('UGRD_10maboveground')[:]),np.nanmean(model.get_value_ptr('VGRD_10maboveground')[:]),np.nanmean(model.get_value_ptr('PRES_surface')[:]),np.nanmean(model.get_value_ptr('DSWRF_surface')[:]),np.nanmean(model.get_value_ptr('DLWRF_surface')[:]))


    # Finalizing the BMI
    print('Finalizing the AORC Forcings BMI')
    model.finalize()


if __name__ == '__main__':
    execute()
