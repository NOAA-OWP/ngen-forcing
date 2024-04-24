# Testing BMI Python modules.
 - ./NextGen_Forcings_Engine/model.py: This file is the "NextGen Forcings Engine Driver" it takes inputs and gives an output
 - ./NextGen_Forcings_Engine/bmi_model.py: This is the Basic Model Interface that talks with the model. This is the NextGen Forcings Engine BMI class.
 - run_bmi_model.py: This is a file that mimics the framework, in the sense that it initializes the model with the BMI function. Then it runs the model with the BMI Update function, etc.
 - run_bmi_unit_test.py: This is a file that runs each BMI unit test to make sure that the BMI is complete and functioning as expected.
 - config.yml: This is a configuration file that the BMI reads to set inital_time (initial value of current_model_time) and all the required variables needed to drive the NextGen Forcings Engine.
 - environment.yml: Environment file with the required Python libraries needed to run the model with BMI and the NextGen Forcings Engine. Create the environment with this command: `conda env create -f environment.yml`, then activate it with `conda activate bmi_test`
 - slurm.job: Example Python submission script for a slurm job manager that is utilized to execute a MPI Python script (NextGen Forcings Engine) on a supercomputer.
 - ./NextGen_Forcings_Engine/core/: The sub-directory containing all the support Python modules required to execute the NextGen Forcings Engine driver within model.py script.
 - ./BMI_NextGen_Configs/: A sub-directory containing all the current config.yml BMI file setupts that will allow a a user to driver the NextGen BMI Forcings Engine for a gridded domain, a coastal-model unstructured mesh domain, and a NextGen hydrofabric (VPU 06) unstructured mesh domain. 
 - ./NextGen_Domains/: A sub-directory that contains a sample domain geogrid file for a gridded domain, a coastal-model unstructured mesh domain, and a NextGen hydrofabric (VPU 06) unstructured mesh domain that can all be utitlized currently to regrid forcings for a Medium Range forecast simulation driver by GFS forcings.
 - ./NWM_Params/: An empty sub-directory that will eventually contain supporting climatology files that will support implementing various bias calibration and downscaling technqius that are only associated with the original WRF-Hydro domain (aka NOAH-OWP Modular).
 - ./Unit_Test_Output/: An sub-directory that is essentially the current scratch directory setup within each of the config.yml files to temporary place I/O commands and the log file of the progress of the NextGen Forcings Engine

# About
This is an implementation of a Python-based model that fulfills the Python language BMI interface and can be used in the Framework. It is intended to serve as a control for testing purposes, freeing the framework from dependency on any real-world model in order to test BMI related functionality.

# Implementation Details

## Test the complete BMI functionality
`python run_bmi_unit_test.py`

## Run the model
`python run_bmi_model.py`

## Sample output
'model time', 'U2D_ELEMENT', 'V2D_ELEMENT', 'LWDOWN_ELEMENT','SWDOWN_ELEMENT','T2D_ELEMENT','Q2D_ELEMENT','PSFC_ELEMENT','RAINRATE_ELEMENT' 
3600 -0.34978889998563434 1.9576997889275944 353.8881030867496 635.9115803301148 299.8541540030717 0.01014856288865672 101253.80092868667 0.0
7200 -0.55664622087895 1.7230866051531413 351.70806444209256 446.07219911115914 299.7995647600879 0.01005461030680388 101259.55711493774 0.0
10800 -0.2516353004284137 1.4639377367413153 347.3662126755073 221.91463406286712 299.4248073664293 0.01016205978429921 101303.53098399537 0.0
