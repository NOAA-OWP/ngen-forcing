# Testing AORC BMI Python modules.
 - AORC_model.py: This file is the " AORC model ExactExtract Manual Regridding Method", which will directly download AORC data for user-specified start time and yields regridded meteorological forcings as output. This AORC model will directly use coverage fractions and raster indices from a user specifed ExactExtract "weights" file to manually calculate aerial weighted averages for AORC forcings data based on a given NextGen hydrofabric. 
 - AORC_bmi_model.py: This is the Basic Model Interface that talks with the model. We've integrated the AORC forcing initalization phase to ingest AORC ExactExtract weights and load up the NextGen hydrofabric and a AORC forcing file to initalize BMI arrays. 
 - AORC_run_bmi_model.py: This is a file that mimics the framework, in the sense that it initializes the model with the BMI function. Then it runs the model with the BMI Update function, etc.
 - AORC_run_bmi_unit_test.py: This is a file that runs each BMI unit test to make sure that the BMI is complete and functioning as expected.
 - config.yml: This is a configuration file that the BMI reads to set inital_time (initial value of current_model_time) and time_step_seconds (time_step_size), a user specified timestamp string indication the start time of the AORC BMI model (start_time), the NextGen hydrofabric file (hyfabfile), and the user specified ExactExtract weights file stating coverage fraction weights for the NextGen hydrofabric (EE_weights). 
 - environment.yml: Environment file with the required Python libraries needed to run the model with BMI. Create the environment with this command: `conda env create -f environment.yml`, then activate it with `conda activate bmi_test`

# About
This is an implementation of a Python-based model that fulfills the Python language BMI interface and can be used in the Framework. It is intended to serve as a control for testing purposes, freeing the framework from dependency on any real-world model in order to test BMI related functionality.

# Implementation Details

## Test the complete BMI functionality
`python run_bmi_unit_test.py`

## Run the model
`python run_bmi_model.py`

## Sample output
model time ids RAINRATE T2D Q2D U2D V2D PSFC SWDOWN LWDOWN
3600 cat-10088 0.0 283.9231261997945 0.007404321230037181 3.2132686504685797 2.893705546416704 98701.07845861379 0.0 342.29985908513993
7200 cat-10088 0.0 283.6979080297457 0.007397903615430698 3.718639911253726 3.0937055493969363 98651.07845861379 0.0 342.9028921828677
10800 cat-10088 0.0 283.4251115637865 0.0073251325976620586 4.200163196069272 3.298307641534474 98601.131810596 0.0 343.49047860100234
14400 cat-10088 0.0 283.1899549316553 0.007297903617956911 4.689946044253713 3.5016319681762806 98551.07845861379 0.0 332.44299921301393
18000 cat-10088 0.0 282.97822978238713 0.007197903620483123 4.37430360498967 3.8385290178442695 98535.8141166359 0.0 332.99168173391695
21600 cat-10088 0.0 282.7251115533557 0.007097903623009335 4.0713375645731285 4.181766995253243 98529.80286349979 0.0 333.50639059797936
25200 cat-10088 0.0 282.4859092709742 0.0070008696613626156 3.748494163852598 4.504770822794534 98515.37607509154 0.0 340.02091884882384
28800 cat-10088 0.0 282.4268919690291 0.006979694171876986 4.113720513336804 4.4719649857095405 98542.09466076619 0.0 340.33649679687164
32400 cat-10088 0.0 282.3559123356471 0.006928942563344763 4.471337570533593 4.423092421484163 98569.94048845569 0.0 340.5343598713857
36000 cat-10088 0.0 282.25888278626934 0.006888448868227177 4.838621319736439 4.390739532822253 98594.68597277963 0.0 328.7507544564197
