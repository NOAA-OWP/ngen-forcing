from pathlib import Path

import numpy as np

# This is the BMI LSTM that we will be running
from bmi_model import NWMv3_Forcing_Engine_BMI_model


def execute():
    # creating an instance of a model
    print('creating an instance of an BMI_MODEL model object')
    model = NWMv3_Forcing_Engine_BMI_model()

    # Initializing the BMI
    print('Initializing the BMI')
    current_dir = Path(__file__).parent.resolve()
    model.initialize(bmi_cfg_file_name=str(current_dir.joinpath('config.yml')))

    # Now loop through the inputs, set the forcing values, and update the model
    print('Now loop through the inputs, updating the model, and extracting forcing data')
    print('\n')
    if(model._grid_type == "gridded"):
        # Initialize numpy arrays for get value
        U2D = np.zeros(model._varsize,dtype=float)
        V2D = np.zeros(model._varsize,dtype=float)
        LWDOWN = np.zeros(model._varsize,dtype=float)
        SWDOWN = np.zeros(model._varsize,dtype=float)
        T2D = np.zeros(model._varsize,dtype=float)
        Q2D = np.zeros(model._varsize,dtype=float)
        PSFC = np.zeros(model._varsize,dtype=float)
        RAINRATE = np.zeros(model._varsize,dtype=float)
    elif(model._grid_type == "hydrofabric"):
        # Initialize numpy arrays for get value
        CAT_IDS = np.zeros(model._varsize,dtype=int)
        U2D = np.zeros(model._varsize,dtype=float)
        V2D = np.zeros(model._varsize,dtype=float)
        LWDOWN = np.zeros(model._varsize,dtype=float)
        SWDOWN = np.zeros(model._varsize,dtype=float)
        T2D = np.zeros(model._varsize,dtype=float)
        Q2D = np.zeros(model._varsize,dtype=float)
        PSFC = np.zeros(model._varsize,dtype=float)
        RAINRATE = np.zeros(model._varsize,dtype=float)
    else:
        # Initialize numpy arrays for get value
        U2D_NODE = np.zeros(model._varsize,dtype=float)
        V2D_NODE = np.zeros(model._varsize,dtype=float)
        LWDOWN_NODE = np.zeros(model._varsize,dtype=float)
        SWDOWN_NODE = np.zeros(model._varsize,dtype=float)
        T2D_NODE = np.zeros(model._varsize,dtype=float)
        Q2D_NODE = np.zeros(model._varsize,dtype=float)
        PSFC_NODE = np.zeros(model._varsize,dtype=float)
        RAINRATE_NODE = np.zeros(model._varsize,dtype=float)

        U2D_ELEMENT = np.zeros(model._varsize_elem,dtype=float)
        V2D_ELEMENT = np.zeros(model._varsize_elem,dtype=float)
        LWDOWN_ELEMENT = np.zeros(model._varsize_elem,dtype=float)
        SWDOWN_ELEMENT = np.zeros(model._varsize_elem,dtype=float)
        T2D_ELEMENT = np.zeros(model._varsize_elem,dtype=float)
        Q2D_ELEMENT = np.zeros(model._varsize_elem,dtype=float)
        PSFC_ELEMENT = np.zeros(model._varsize_elem,dtype=float)
        RAINRATE_ELEMENT = np.zeros(model._varsize_elem,dtype=float)

    for x in range(18):

        #########################################
        # UPDATE THE MODEL AND GET REGRIDDED FORCINGS #
        model.update()     ######################
        #########################################

        # PRINT THE MODEL RESULTS FOR THIS TIME STEP#################################################

        ### Due to numpy version, we must initialize arrays
        ### to get values and directly call the array as an
        ### input arguement
        if(model._grid_type == "gridded"):
            U2D = model.get_value('U2D_ELEMENT',U2D)
            V2D = model.get_value('V2D_ELEMENT',V2D)
            T2D = model.get_value('T2D_ELEMENT',T2D)
            Q2D = model.get_value('Q2D_ELEMENT',Q2D)
            SWDOWN = model.get_value('SWDOWN_ELEMENT',SWDOWN)
            LWDOWN = model.get_value('LWDOWN_ELEMENT',LWDOWN)
            PSFC = model.get_value('PSFC_ELEMENT',PSFC)
            RAINRATE = model.get_value('RAINRATE_ELEMENT',RAINRATE)
            print('model time', 'U2D_ELEMENT max', 'V2D_ELEMENT max', 'LWDOWN_ELEMENT max','SWDOWN_ELEMENT max','T2D_ELEMENT max','Q2D_ELEMENT max','PSFC_ELEMENT max','RAINRATE_ELEMENT max')
            print(model.get_current_time(), U2D.max(), V2D.max(), LWDOWN.max(), SWDOWN.max(), T2D.max(), Q2D.max(), PSFC.max(), RAINRATE.max())
            print('model time', 'U2D_ELEMENT min', 'V2D_ELEMENT min', 'LWDOWN_ELEMENT min','SWDOWN_ELEMENT min','T2D_ELEMENT min','Q2D_ELEMENT min','PSFC_ELEMENT min','RAINRATE_ELEMENT min')
            print(model.get_current_time(), U2D.min(), V2D.min(), LWDOWN.min(), SWDOWN.min(), T2D.min(), Q2D.min(), PSFC.min(), RAINRATE.min())
        elif(model._grid_type == "hydrofabric"):
            CAT_IDS = model.get_value('CAT-ID',CAT_IDS)
            U2D = model.get_value('U2D_ELEMENT',U2D)
            V2D = model.get_value('V2D_ELEMENT',V2D)
            T2D = model.get_value('T2D_ELEMENT',T2D)
            Q2D = model.get_value('Q2D_ELEMENT',Q2D)
            SWDOWN = model.get_value('SWDOWN_ELEMENT',SWDOWN)
            LWDOWN = model.get_value('LWDOWN_ELEMENT',LWDOWN)
            PSFC = model.get_value('PSFC_ELEMENT',PSFC)
            RAINRATE = model.get_value('RAINRATE_ELEMENT',RAINRATE)
            print('model time', 'CAT_ID_max','U2D_ELEMENT max', 'V2D_ELEMENT max', 'LWDOWN_ELEMENT max','SWDOWN_ELEMENT max','T2D_ELEMENT max','Q2D_ELEMENT max','PSFC_ELEMENT max','RAINRATE_ELEMENT max')
            print(model.get_current_time(), CAT_IDS.max(), U2D.max(), V2D.max(), LWDOWN.max(), SWDOWN.max(), T2D.max(), Q2D.max(), PSFC.max(), RAINRATE.max())
            print('model time', 'CAT_ID_min','U2D_ELEMENT min', 'V2D_ELEMENT min', 'LWDOWN_ELEMENT min','SWDOWN_ELEMENT min','T2D_ELEMENT min','Q2D_ELEMENT min','PSFC_ELEMENT min','RAINRATE_ELEMENT min')
            print(model.get_current_time(), CAT_IDS.min(), U2D.min(), V2D.min(), LWDOWN.min(), SWDOWN.min(), T2D.min(), Q2D.min(), PSFC.min(), RAINRATE.min())
        else:
            U2D_NODE = model.get_value('U2D_NODE',U2D_NODE)
            V2D_NODE = model.get_value('V2D_NODE',V2D_NODE)
            T2D_NODE = model.get_value('T2D_NODE',T2D_NODE)
            Q2D_NODE = model.get_value('Q2D_NODE',Q2D_NODE)
            SWDOWN_NODE = model.get_value('SWDOWN_NODE',SWDOWN_NODE)
            LWDOWN_NODE = model.get_value('LWDOWN_NODE',LWDOWN_NODE)
            PSFC_NODE = model.get_value('PSFC_NODE',PSFC_NODE)
            RAINRATE_NODE = model.get_value('RAINRATE_NODE',RAINRATE_NODE)

            U2D_ELEMENT = model.get_value('U2D_ELEMENT',U2D_ELEMENT)
            V2D_ELEMENT = model.get_value('V2D_ELEMENT',V2D_ELEMENT)
            T2D_ELEMENT = model.get_value('T2D_ELEMENT',T2D_ELEMENT)
            Q2D_ELEMENT = model.get_value('Q2D_ELEMENT',Q2D_ELEMENT)
            SWDOWN_ELEMENT = model.get_value('SWDOWN_ELEMENT',SWDOWN_ELEMENT)
            LWDOWN_ELEMENT = model.get_value('LWDOWN_ELEMENT',LWDOWN_ELEMENT)
            PSFC_ELEMENT = model.get_value('PSFC_ELEMENT',PSFC_ELEMENT)
            RAINRATE_ELEMENT = model.get_value('RAINRATE_ELEMENT',RAINRATE_ELEMENT)
            print('model time', 'U2D_NODE max', 'V2D_NODE max', 'LWDOWN_NODE max','SWDOWN_NODE max','T2D_NODE max','Q2D_NODE max','PSFC_NODE max','RAINRATE_NODE max', 'U2D_ELEMENT max', 'V2D_ELEMENT max', 'LWDOWN_ELEMENT max','SWDOWN_ELEMENT max','T2D_ELEMENT max','Q2D_ELEMENT max','PSFC_ELEMENT max','RAINRATE_ELEMENT max')
            print(model.get_current_time(), U2D_NODE.max(), V2D_NODE.max(), LWDOWN_NODE.max(), SWDOWN_NODE.max(), T2D_NODE.max(), Q2D_NODE.max(), PSFC_NODE.max(), RAINRATE_NODE.max(), U2D_ELEMENT.max(), V2D_ELEMENT.max(), LWDOWN_ELEMENT.max(), SWDOWN_ELEMENT.max(), T2D_ELEMENT.max(), Q2D_ELEMENT.max(), PSFC_ELEMENT.max(), RAINRATE_ELEMENT.max())
            print('model time', 'U2D_NODE min', 'V2D_NODE min', 'LWDOWN_NODE min','SWDOWN_NODE min','T2D_NODE min','Q2D_NODE min','PSFC_NODE min','RAINRATE_NODE min', 'U2D_ELEMENT min', 'V2D_ELEMENT min', 'LWDOWN_ELEMENT min','SWDOWN_ELEMENT min','T2D_ELEMENT min','Q2D_ELEMENT min','PSFC_ELEMENT min','RAINRATE_ELEMENT min')
            print(model.get_current_time(), U2D_NODE.min(), V2D_NODE.min(), LWDOWN_NODE.min(), SWDOWN_NODE.min(), T2D_NODE.min(), Q2D_NODE.min(), PSFC_NODE.min(), RAINRATE_NODE.min(), U2D_ELEMENT.min(), V2D_ELEMENT.min(), LWDOWN_ELEMENT.min(), SWDOWN_ELEMENT.min(), T2D_ELEMENT.min(), Q2D_ELEMENT.min(), PSFC_ELEMENT.min(), RAINRATE_ELEMENT.min())

    # Finalizing the BMI
    print('Finalizing the BMI')
    model.finalize()


if __name__ == '__main__':
    execute()
