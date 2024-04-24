"""
Layering module for implementing various layering schemes in the WRF-Hydro forcing engine.
Future functionality may include blenidng, etc.
"""
import numpy as np
from . import err_handler

def layer_final_forcings(OutputObj,input_forcings,ConfigOptions,MpiConfig):
    """
    Function to perform basic layering of input forcings as they are processed. The logic
    works as following:
    1.) As the parent calling program loops through the forcings for each layer
        for this timestep, forcings are placed onto the output grid by shear brute
        replacement. However, this only occurs where valid data exists.
        Supplemental precipitation will be layered in separately.
    :param OutputObj:
    :param input_forcings:
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """
    # Loop through the 8 forcing products to layer in:
    # 0.) U-Wind (m/s)
    # 1.) V-Wind (m/s)
    # 2.) Surface incoming longwave radiation flux (W/m^2)
    # 3.) Precipitation rate (mm/s)
    # 4.) 2-meter temperature (K)
    # 5.) 2-meter specific humidity (kg/kg)
    # 6.) Surface pressure (Pa)
    # 7.) Surface incoming shortwave radiation flux (W/m^2)

    for force_idx in range(0,8):
        if force_idx in input_forcings.input_map_output:
            if(ConfigOptions.grid_type == "gridded"):
                outLayerCurrent = OutputObj.output_local[force_idx,:,:]
                layerIn = input_forcings.final_forcings[force_idx,:,:]
                if(input_forcings.productName == 'ERA5' and 12 in ConfigOptions.input_forcings or 21 in ConfigOptions.input_forcings):
                    outLayerCurrent[np.where(input_forcings.regridded_mask_AORC==0)] = layerIn[np.where(input_forcings.regridded_mask_AORC==0)]
                    OutputObj.output_local[force_idx, :, :] = outLayerCurrent
                else:
                    indSet = np.where(layerIn != ConfigOptions.globalNdv)
                    outLayerCurrent[indSet] = layerIn[indSet]
                    OutputObj.output_local[force_idx, :, :] = outLayerCurrent

                # Reset for next iteration and memory efficiency.
                indSet = None
            elif(ConfigOptions.grid_type == "unstructured"):
                outLayerCurrent = OutputObj.output_local[force_idx,:]
                layerIn = input_forcings.final_forcings[force_idx,:]
                if(input_forcings.productName == 'ERA5' and 12 in ConfigOptions.input_forcings or 21 in ConfigOptions.input_forcings):
                    outLayerCurrent[np.where(input_forcings.regridded_mask_AORC==0)] = layerIn[np.where(input_forcings.regridded_mask_AORC==0)]
                    OutputObj.output_local[force_idx, :] = outLayerCurrent
                else:
                    indSet = np.where(layerIn != ConfigOptions.globalNdv)
                    outLayerCurrent[indSet] = layerIn[indSet]
                    OutputObj.output_local[force_idx, :] = outLayerCurrent

                outLayerCurrent_elem = OutputObj.output_local_elem[force_idx,:]
                layerIn_elem = input_forcings.final_forcings_elem[force_idx,:]
                if(input_forcings.productName == 'ERA5' and 12 in ConfigOptions.input_forcings or 21 in ConfigOptions.input_forcings):
                    outLayerCurrent_elem[np.where(input_forcings.regridded_mask_elem_AORC==0)] = layerIn_elem[np.where(input_forcings.regridded_mask_elem_AORC==0)]
                    OutputObj.output_local_elem[force_idx, :] = outLayerCurrent_elem
                else:
                    indSet_elem = np.where(layerIn_elem != ConfigOptions.globalNdv)
                    outLayerCurrent_elem[indSet_elem] = layerIn_elem[indSet_elem]
                    OutputObj.output_local_elem[force_idx, :] = outLayerCurrent_elem

                # Reset for next iteration and memory efficiency.
                indSet = None
                indSet_elem = None
            elif(ConfigOptions.grid_type == "hydrofabric"):
                outLayerCurrent = OutputObj.output_local[force_idx,:]
                layerIn = input_forcings.final_forcings[force_idx,:]
                if(input_forcings.productName == 'ERA5' and 12 in ConfigOptions.input_forcings or 21 in ConfigOptions.input_forcings):
                    outLayerCurrent[np.where(input_forcings.regridded_mask_AORC==0)] = layerIn[np.where(input_forcings.regridded_mask_AORC==0)]
                    OutputObj.output_local[force_idx, :] = outLayerCurrent
                else:
                    indSet = np.where(layerIn != ConfigOptions.globalNdv)
                    outLayerCurrent[indSet] = layerIn[indSet]
                    OutputObj.output_local[force_idx, :] = outLayerCurrent
                # Reset for next iteration and memory efficiency.
                indSet = None

    # MpiConfig.comm.barrier()


def layer_supplemental_forcing(OutputObj, supplemental_precip, ConfigOptions, MpiConfig):
    """
    Function to layer in supplemental precipitation where we have valid values. Any pixel
    cells that contain missing values will not be layered in, and background input forcings
    will be used instead.
    :param OutputObj:
    :param supplemental_precip:
    :param ConfigOptions:
    :param MpiConfig:
    :return:
    """

    if(ConfigOptions.grid_type == "gridded"):
        indSet = np.where(supplemental_precip.final_supp_precip != ConfigOptions.globalNdv)
        layerIn = supplemental_precip.final_supp_precip
        layerOut = OutputObj.output_local[supplemental_precip.output_var_idx, :, :]
        #TODO: review test layering for ExtAnA calculation to replace FE QPE with MPE RAINRATE
        #If this isn't sufficient, replace QPE with MPE here:
        #if supplemental_precip.keyValue == 11:
        #    ConfigOptions.statusMsg = "Performing ExtAnA calculation"
        #    err_handler.log_msg(ConfigOptions, MpiConfig)
        if len(indSet[0]) != 0:
            layerOut[indSet] = layerIn[indSet]
        else:
            # We have all missing data for the supplemental precip for this step.
            layerOut = layerOut
        # TODO: test that even does anything...?s
        OutputObj.output_local[supplemental_precip.output_var_idx, :, :] = layerOut
    elif(ConfigOptions.grid_type == "unstructured"):
        indSet = np.where(supplemental_precip.final_supp_precip != ConfigOptions.globalNdv)
        layerIn = supplemental_precip.final_supp_precip
        layerOut = OutputObj.output_local[supplemental_precip.output_var_idx, :]
  
        if len(indSet[0]) != 0:
            layerOut[indSet] = layerIn[indSet]
        else:
            # We have all missing data for the supplemental precip for this step.
            layerOut = layerOut
        # TODO: test that even does anything...?s
        OutputObj.output_local[supplemental_precip.output_var_idx, :] = layerOut

        indSet_elem = np.where(supplemental_precip.final_supp_precip_elem != ConfigOptions.globalNdv)
        layerIn_elem = supplemental_precip.final_supp_precip_elem
        layerOut_elem = OutputObj.output_local_elem[supplemental_precip.output_var_idx, :]

        if len(indSet_elem[0]) != 0:
            layerOut_elem[indSet_elem] = layerIn_elem[indSet_elem]
        else:
            # We have all missing data for the supplemental precip for this step.
            layerOut_elem = layerOut_elem
        # TODO: test that even does anything...?s
        OutputObj.output_local_elem[supplemental_precip.output_var_idx, :] = layerOut_elem
    elif(ConfigOptions.grid_type == "hydrofabric"):
        indSet = np.where(supplemental_precip.final_supp_precip != ConfigOptions.globalNdv)
        layerIn = supplemental_precip.final_supp_precip
        layerOut = OutputObj.output_local[supplemental_precip.output_var_idx, :]

        if len(indSet[0]) != 0:
            layerOut[indSet] = layerIn[indSet]
        else:
            # We have all missing data for the supplemental precip for this step.
            layerOut = layerOut
        # TODO: test that even does anything...?s
        OutputObj.output_local[supplemental_precip.output_var_idx, :] = layerOut

