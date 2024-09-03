import configparser
import datetime
import json
import os

import numpy as np

from . import time_handling
from . import err_handler

FORCE_COUNT = 27

class ConfigOptions:
    """
    Configuration abstract class for configuration options read in from the file
    specified by the user.
    """

    def __init__(self, config):
        """
        Initialize the configuration class to empty None attributes
        param config: The user-specified path to the configuration file.
        """
        self.bmi_time = None
        self.current_time = None
        self.bmi_time_index = 0
        self.input_forcings = None
        self.precip_only_flag = False
        self.supp_precip_forcings = None
        self.input_force_dirs = None
        self.input_force_types = None
        self.supp_precip_dirs = None
        self.supp_precip_file_types = None
        self.supp_precip_param_dir = None
        self.input_force_mandatory = None
        self.supp_precip_mandatory = None
        self.supp_pcp_max_hours = None
        self.number_inputs = None
        self.number_supp_pcp = None
        self.number_custom_inputs = 0
        self.output_freq = None
        self.sub_output_hour = None
        self.sub_output_freq = None
        self.scratch_dir = None
        self.useCompression = 0
        self.useFloats = 0
        self.num_output_steps = None
        self.num_supp_output_steps = None
        self.actual_output_steps = None
        self.realtime_flag = None
        self.refcst_flag = None
        self.ana_flag = None
        self.b_date_proc = None
        self.e_date_proc = None
        self.first_fcst_cycle = None
        self.current_fcst_cycle = None
        self.current_output_step = None
        self.cycle_length_minutes = None
        self.prev_output_date = None
        self.current_output_date = None
        self.look_back = None
        self.future_time = None
        self.fcst_freq = None
        self.nFcsts = None
        self.fcst_shift = None
        self.fcst_input_horizons = None
        self.fcst_input_offsets = None
        self.process_window = None
        self.spatial_meta = None
        self.grid_type = None
        self.grid_meta = None
        self.ExactExtract = None
        self.lat_var = None
        self.lon_var = None
        self.hgt_var = None
        self.cosalpha_var = None
        self.sinalpha_var = None
        self.slope_var = None
        self.slope_azimuth_var = None
        self.slope_var_elem = None
        self.slope_azimuth_var_elem = None
        self.nodecoords_var = None
        self.elemcoords_var = None
        self.elemconn_var = None
        self.numelemconn_var = None
        self.element_id_var = None
        self.hgt_elem_var = None
        self.ignored_border_widths = None
        self.regrid_opt = None
        self.weightsDir = None
        self.regrid_opt_supp_pcp = None
        self.config_path = config
        self.errMsg = None
        self.statusMsg = None
        self.logFile = None
        self.logHandle = None
        self.dScaleParamDirs = None
        self.paramFlagArray = None
        self.forceTemoralInterp = None
        self.suppTemporalInterp = None
        self.t2dDownscaleOpt = None
        self.swDownscaleOpt = None
        self.psfcDownscaleOpt = None
        self.precipDownscaleOpt = None
        self.q2dDownscaleOpt = None
        self.t2BiasCorrectOpt = None
        self.psfcBiasCorrectOpt = None
        self.q2BiasCorrectOpt = None
        self.windBiasCorrect = None
        self.swBiasCorrectOpt = None
        self.lwBiasCorrectOpt = None
        self.precipBiasCorrectOpt = None
        self.runCfsNldasBiasCorrect = False
        self.cfsv2EnsMember = None
        self.customSuppPcpFreq = None
        self.customFcstFreq = None
        self.rqiMethod = None
        self.rqiThresh = 1.0
        self.globalNdv = -9999.0
        self.d_program_init = datetime.datetime.utcnow()
        self.errFlag = 0
        self.nwmVersion = None
        self.nwmConfig = None
        self.include_lqfrac = False
        self.forcing_output = None
        self.aws = None
        self.aws_obj = None
        self.aorc_source = "s3://noaa-nws-aorc-v1-1-1km"
        self.aorc_year_url = "{source}/{year}.zarr"
        self.nwm_source = "s3://noaa-nwm-retrospective-3-0-pds"
        self.nwm_url = None
        self.nwm_domain = None
        self.aws_time = None
        self.nwm_geogrid = None

    def read_config(self, cfg):
        """
        Read in options from the configuration file and check that proper options
        were provided.
        """
        # Read in the base input forcing options as an array of values to map.
        try:
            self.supp_precip_forcings = cfg['SuppPcp']
        except KeyError:
            err_handler.err_out_screen('Unable to locate SuppPcp under SuppForcing section in configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate SuppPcp under SuppForcing section in configuration file.')
        except json.decoder.JSONDecodeError:
            err_handler.err_out_screen('Improper SuppPcp option specified in configuration file')

        self.number_supp_pcp = len(self.supp_precip_forcings)

        if self.number_supp_pcp == 1:
            if int(self.supp_precip_forcings[0]) == 14:
                self.precip_only_flag = True

        if self.precip_only_flag == False:
            # Read in the base input forcing options as an array of values to map.
            try:
                self.input_forcings = cfg['InputForcings']
            except KeyError:
                err_handler.err_out_screen('Unable to locate InputForcings under Input section in configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate InputForcings under Input section in configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper InputForcings option specified in configuration file')
            if len(self.input_forcings) == 0:
                err_handler.err_out_screen('Please choose at least one InputForcings dataset to process')
            self.number_inputs = len(self.input_forcings)

            # Check to make sure forcing options make sense
            for forceOpt in self.input_forcings:
                if forceOpt < 0 or forceOpt > FORCE_COUNT:
                    err_handler.err_out_screen(f'Please specify InputForcings values between 1 and {FORCE_COUNT}.')

                # Keep tabs on how many custom input forcings we have.
                if forceOpt == 10:
                    self.number_custom_inputs = self.number_custom_inputs + 1

                # Flag to force mandatory configuration option to specify the NWM geogrid file if user requests
                # NWM forcing files to be regridded to a given domain configuration
                if forceOpt == 27:
                    try:
                        self.nwm_geogrid = cfg['NWM_Geogrid']
                    except KeyError:
                        err_handler.err_out_screen('Unable to locate NWM Geogrid file required for the NWM forcings module. Need to specify the pathway to the NWM geo_em_DOMAIN.nc file to the NWM_Geogrid configuration input option within the configuration file.')
                    except configparser.NoOptionError:
                        err_handler.err_out_screen('Unable to locate NWM Geogrid file required for the NWM forcings module. Need to specify the pathway to the NWM geo_em_DOMAIN.nc file to the NWM_Geogrid configuration input option within the configuration file.')
                    except json.decoder.JSONDecodeError:
                        err_handler.err_out_screen('Improper NWM Geogrid file option specified in configuration file')
                    if(self.nwm_geogrid.split('/')[-1].split('_')[-1].split('.')[0] == 'CONUS'):
                        self.nwm_domain = 'CONUS'
                        self.nwm_url = "{source}/{domain}/zarr/forcing/{var}.zarr"
                    elif(self.nwm_geogrid.split('/')[-1].split('_')[-1].split('.')[0] == 'HI'):
                        self.nwm_domain = 'Hawaii'
                        self.nwm_url = "{source}/{domain}/zarr/forcing.zarr"
                    elif(self.nwm_geogrid.split('/')[-1].split('_')[-1].split('.')[0] == 'PRVI'):
                        self.nwm_domain = 'PR'
                        self.nwm_url = "{source}/{domain}/zarr/forcing.zarr"
                    elif(self.nwm_geogrid.split('/')[-1].split('_')[-1].split('.')[0] == 'AK'):
                        self.nwm_domain = 'Alaska'
                        self.nwm_url = "{source}/{domain}/zarr/forcing.zarr"

            # Read in the input forcings types (GRIB[1|2], NETCDF)
            try:
                #self.input_force_types = config.get('Input', 'InputForcingTypes').strip("[]").split(',')
                #self.input_force_types = [ftype.strip() for ftype in self.input_force_types]
                self.input_force_types = cfg['InputForcingTypes']
                if self.input_force_types == ['']:
                    self.input_force_types = []
            except KeyError:
                err_handler.err_out_screen('Unable to locate InputForcingTypes in Input section '
                                           'in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate InputForcingTypes in Input section '
                                           'in the configuration file.')
            if len(self.input_force_types) != self.number_inputs:
                err_handler.err_out_screen('Number of InputForcingTypes must match the number '
                                           'of InputForcings in the configuration file.')
            for fileType in self.input_force_types:
                if fileType not in ['GRIB1', 'GRIB2', 'NETCDF', 'NETCDF4','NWM']:
                    err_handler.err_out_screen('Invalid forcing file type "{}" specified. '
                                               'Only GRIB1, GRIB2, and NETCDF are supported'.format(fileType))

            # Read in the input directories for each forcing option.
            try:
                self.input_force_dirs = cfg['InputForcingDirectories']
            except KeyError:
                err_handler.err_out_screen('Unable to locate InputForcingDirectories in Input section '
                                           'in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate InputForcingDirectories in Input section '
                                           'in the configuration file.')
            if len(self.input_force_dirs) != self.number_inputs:
                err_handler.err_out_screen('Number of InputForcingDirectories must match the number '
                                           'of InputForcings in the configuration file.')
            # Loop through and ensure all input directories exist. Also strip out any whitespace
            # or new line characters.
            for dirTmp in range(0, len(self.input_force_dirs)):
                self.input_force_dirs[dirTmp] = self.input_force_dirs[dirTmp].strip()
                if(os.path.isdir(self.input_force_dirs[dirTmp]) == False and self.input_forcings[dirTmp] != 12 and self.input_forcings[dirTmp] != 21 and self.input_forcings[dirTmp] != 27):
                    err_handler.err_out_screen('Unable to locate forcing directory: ' +
                                               self.input_force_dirs[dirTmp])
                if(os.path.isdir(self.input_force_dirs[dirTmp]) == False and self.input_forcings[dirTmp] in [12,21,27]):
                    self.aws = True
     
            # Read in the mandatory enforcement options for input forcings.
            try:
                self.input_force_mandatory = cfg['InputMandatory']
            except KeyError:
                err_handler.err_out_screen('Unable to locate InputMandatory under Input section in configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate InputMandatory under Input section in configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper InputMandatory option specified in configuration file')

            if len(self.input_force_mandatory) != self.number_inputs:
                err_handler.err_out_screen('Please specify InputMandatory values for each corresponding input '
                                           'forcings in the configuration file.')
            # Check to make sure enforcement options makes sense.
            for enforceOpt in self.input_force_mandatory:
                if enforceOpt < 0 or enforceOpt > 1:
                    err_handler.err_out_screen('Invalid InputMandatory chosen in the configuration file. Please choose a value of 0 or 1 for each corresponding input forcing.')


        # Read in the output frequency
        try:
            self.output_freq = cfg['OutputFrequency']
        except ValueError:
            err_handler.err_out_screen('Improper OutputFrequency value specified  in the configuration file.')
        except KeyError:
            err_handler.err_out_screen('Unable to locate OutputFrequency in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate OutputFrequency in the configuration file.')
        if self.output_freq <= 0:
            err_handler.err_out_screen('Please specify an OutputFrequency that is greater than zero minutes.')

        if self.precip_only_flag == True:
            # Read in the custom supp output frequency
            try:
                self.customSuppPcpFreq = int(cfg['customSuppPcpFreq'])
            except ValueError:
                err_handler.err_out_screen('Improper customSuppPcpFreq value specified  in the configuration file.')
            except KeyError:
                err_handler.err_out_screen('Unable to locate customSuppPcpFreq in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate customSuppPcpFreq in the configuration file.')
            if self.output_freq <= 0:
                err_handler.err_out_screen('Please specify an customSuppPcpFreq that is greater than zero minutes.')

        # Read in the sub output hour
        try:
            self.sub_output_hour = int(cfg['SubOutputHour'])
        except ValueError:
            err_handler.err_out_screen('Improper SubOutputHour value specified  in the configuration file.')
        except KeyError:
            err_handler.err_out_screen('Unable to locate SubOutputHour in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate SubOutputHour in the configuration file.')
        if self.sub_output_hour < 0:
            err_handler.err_out_screen('Please specify an SubOutputHour that is greater than zero minutes.')
        if self.sub_output_hour == 0:
            self.sub_output_hour = None
        # Read in the output frequency
        try:
            self.sub_output_freq = int(cfg['SubOutFreq'])
        except ValueError:
            err_handler.err_out_screen('Improper SubOutFreq value specified  in the configuration file.')
        except KeyError:
            err_handler.err_out_screen('Unable to locate SubOutFreq in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate SubOutFreq in the configuration file.')
        if self.sub_output_freq < 0:
            err_handler.err_out_screen('Please specify an SubOutFreq that is greater than zero minutes.')
        if self.sub_output_freq == 0:
            self.sub_output_freq = None

        # Read in the scratch temporary directory, which also may contain output forcing file if requested.
        try:
            self.scratch_dir = cfg['ScratchDir']
        except ValueError:
            err_handler.err_out_screen('Improper ScratchDir specified in the configuration file.')
        except KeyError:
            err_handler.err_out_screen('Unable to locate ScratchDir in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate ScratchDir in the configuration file.')
        if not os.path.isdir(self.scratch_dir):
            err_handler.err_out_screen('Specified output directory: ' + self.scratch_dir + ' not found')

        # Read in compression option
        try:
            self.useCompression = cfg['compressOutput']
        except KeyError:
            err_handler.err_out_screen('Unable to locate compressOut in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate compressOut in the configuration file.')
        except ValueError:
            err_handler.err_out_screen('Improper compressOut value.')
        if self.useCompression < 0 or self.useCompression > 1:
            err_handler.err_out_screen('Please choose a compressOut value of 0 or 1.')

        # Read in floating-point option
        try:
            self.useFloats = cfg['floatOutput']
        except KeyError:
            # err_handler.err_out_screen('Unable to locate floatOutput in the configuration file.')
            self.useFloats = 0
        except configparser.NoOptionError:
            # err_handler.err_out_screen('Unable to locate floatOutput in the configuration file.')
            self.useFloats = 0
        except ValueError:
            err_handler.err_out_screen('Improper floatOutput value: {}'.format(cfg['includeLQFraq']))
        if self.useFloats < 0 or self.useFloats > 1:
            err_handler.err_out_screen('Please choose a floatOutput value of 0 or 1.')

        # Read in lqfrac option
        try:
            self.include_lqfrac = cfg['includeLQFrac']
        except KeyError:
            # err_handler.err_out_screen('Unable to locate includeLQFraq in the configuration file.')
            self.include_lqfrac = 0
        except configparser.NoOptionError:
            # err_handler.err_out_screen('Unable to locate includeLQFraq in the configuration file.')
            self.useFinclude_lqfracloats = 0
        except ValueError:
            err_handler.err_out_screen('Improper includeLQFrac value: {}'.format(cfg['includeLQFraq']))
        if self.include_lqfrac < 0 or self.include_lqfrac > 1:
            err_handler.err_out_screen('Please choose an includeLQFrac value of 0 or 1.')

        # Read in Forcing output option
        try:
            self.forcing_output = cfg['Output']
        except KeyError:
            self.forcing_output = 0
        except configparser.NoOptionError:
            self.forcing_output = 0
        except ValueError:
            err_handler.err_out_screen('Improper Forcing Output value: {}'.format(cfg['Output']))
        if self.forcing_output < 0 or self.forcing_output > 1:
            err_handler.err_out_screen('Please choose a Forcing Output value of 0 (No output) or 1 (output).')

        # Read AnA flag option
        try:
            # check both the Forecast section and if it's not there, the old BiasCorrection location
            self.ana_flag = cfg['AnAFlag']
            if self.ana_flag is None:
                raise KeyError
            else:
                self.ana_flag = int(self.ana_flag)
        except KeyError:
            err_handler.err_out_screen('Unable to locate AnAFlag in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate AnAFlag in the configuration file.')
        except ValueError:
            err_handler.err_out_screen('Improper AnAFlag value ')
        if self.ana_flag < 0 or self.ana_flag > 1:
            err_handler.err_out_screen('Please choose a AnAFlag value of 0 or 1.')

        # For the NextGen Forcings Engine BMI, we are assuming a realtime or reforecast simulation.
        try:
            self.look_back = cfg['LookBack']
            if self.look_back <= 0 and self.look_back != -9999:
                err_handler.err_out_screen('Please specify a positive LookBack or -9999 for realtime.')
        except ValueError:
            err_handler.err_out_screen('Improper LookBack value entered into the '
                                        'configuration file. Please check your entry.')
        except KeyError:
            err_handler.err_out_screen('Unable to locate LookBack in the configuration '
                                        'file. Please verify entries exist.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate LookBack in the configuration '
                                        'file. Please verify entries exist.')

        # Process the beginning date of reforecast forcings to process
        try:
            beg_date_tmp = cfg['RefcstBDateProc']
        except KeyError:
            err_handler.err_out_screen('Unable to locate RefcstBDateProc under Logistics section in '
                                        'configuration file.')
            beg_date_tmp = None
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate RefcstBDateProc under Logistics section in '
                                        'configuration file.')
            beg_date_tmp = None
        if beg_date_tmp != -9999:
            if len(beg_date_tmp) != 12:
                err_handler.err_out_screen('Improper RefcstBDateProc length entered into the '
                                            'configuration file. Please check your entry.')
            try:
                self.b_date_proc = datetime.datetime.strptime(beg_date_tmp, '%Y%m%d%H%M')
            except ValueError:
                err_handler.err_out_screen('Improper RefcstBDateProc value entered into the '
                                            'configuration file. Please check your entry.')
        else:
            self.b_date_proc = -9999

        # If the Retro flag is off, and lookback is off, then we assume we are
        # running a reforecast.
        if self.look_back == -9999:
            self.realtime_flag = False
            self.refcst_flag = True
        elif self.b_date_proc == -9999:
            self.realtime_flag = True
            self.refcst_flag = True
        else:
            # The processing window will be calculated based on current time and the
            # lookback option since this is a realtime instance.
            self.realtime_flag = False
            self.refcst_flag = False
            # self.b_date_proc = -9999
            # self.e_date_proc = -9999

        # Calculate the delta time between the beginning and ending time of processing.
        # self.process_window = self.e_date_proc - self.b_date_proc

        # Read in the ForecastFrequency option.
        try:
            self.fcst_freq = cfg['ForecastFrequency']
        except ValueError:
            err_handler.err_out_screen('Improper ForecastFrequency value entered into '
                                        'the configuration file. Please check your entry.')
        except KeyError:
            err_handler.err_out_screen('Unable to locate ForecastFrequency in the configuration '
                                        'file. Please verify entries exist.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate ForecastFrequency in the configuration '
                                        'file. Please verify entries exist.')
        if self.fcst_freq <= 0:
            err_handler.err_out_screen('Please specify a ForecastFrequency in the configuration '
                                        'file greater than zero.')
        # Currently, we only support daily or sub-daily forecasts. Any other iterations should
        # be done using custom config files for each forecast cycle.
        if self.fcst_freq > 1440:
            err_handler.err_out_screen('Only forecast cycles of daily or sub-daily are supported '
                                        'at this time')

        # Read in the ForecastShift option. This is ONLY done for the realtime instance as
        # it's used to calculate the beginning of the processing window.
        if True: # was: self.realtime_flag:
            try:
                self.fcst_shift = cfg['ForecastShift']
            except ValueError:
                err_handler.err_out_screen('Improper ForecastShift value entered into the '
                                            'configuration file. Please check your entry.')
            except KeyError:
                err_handler.err_out_screen('Unable to locate ForecastShift in the configuration '
                                            'file. Please verify entries exist.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate ForecastShift in the configuration '
                                            'file. Please verify entries exist.')
            if self.fcst_shift < 0:
                err_handler.err_out_screen('Please specify a ForecastShift in the configuration '
                                            'file greater than or equal to zero.')

            # Calculate the beginning/ending processing dates if we are running realtime
            if self.realtime_flag:
                time_handling.calculate_lookback_window(self)

        #if self.refcst_flag:
            # Calculate the number of forecasts to issue, and verify the user has chosen a
            # correct divider based on the dates
            #dt_tmp = self.e_date_proc - self.b_date_proc
            #if (dt_tmp.days * 1440 + dt_tmp.seconds / 60.0) % self.fcst_freq != 0:
            #    err_handler.err_out_screen('Please choose an equal divider forecast frequency for your '
            #                               'specified reforecast range.')
            #self.nFcsts = int((dt_tmp.days * 1440 + dt_tmp.seconds / 60.0) / self.fcst_freq)

            # Flag to constrain AORC forcing data cycle output
            #for optTmp in self.input_forcings:
                #if optTmp == 12:
                    #self.nFcsts = 1
        self.nFcsts = 1

        if self.look_back != -9999:
            time_handling.calculate_lookback_window(self)

        if self.precip_only_flag == False:
            # Read in the ForecastInputHorizons options.
            try:
                self.fcst_input_horizons = cfg['ForecastInputHorizons']
            except KeyError:
                err_handler.err_out_screen('Unable to locate ForecastInputHorizons under Forecast section in '
                                            'configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate ForecastInputHorizons under Forecast section in '
                                            'configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper ForecastInputHorizons option specified in '
                                            'configuration file')
            if len(self.fcst_input_horizons) != self.number_inputs:
                err_handler.err_out_screen('Please specify ForecastInputHorizon values for '
                                            'each corresponding input forcings for forecasts.')

            # Check to make sure the horizons options make sense. There will be additional
            # checking later when input choices are mapped to input products.
            for horizonOpt in self.fcst_input_horizons:
                if horizonOpt <= 0:
                    err_handler.err_out_screen('Please specify ForecastInputHorizon values greater '
                                                'than zero.')
        else:
            # Read in the ForecastInputHorizons options.
            try:
                self.fcst_input_horizons = cfg['ForecastInputHorizons']
            except KeyError:
                err_handler.err_out_screen('Unable to locate ForecastInputHorizons under Forecast section in '
                                            'configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate ForecastInputHorizons under Forecast section in '
                                            'configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper ForecastInputHorizons option specified in '
                                            'configuration file')
                if len(self.fcst_input_horizons) != 1:
                    err_handler.err_out_screen('Please specify ForecastInputHorizon values for '
                                               'each corresponding input forcings for forecasts.')

        if self.precip_only_flag == False:

            # Read in the ForecastInputOffsets options.
            try:
                self.fcst_input_offsets = cfg['ForecastInputOffsets']
            except KeyError:
                err_handler.err_out_screen('Unable to locate ForecastInputOffsets under Forecast '
                                            'section in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate ForecastInputOffsets under Forecast '
                                            'section in the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper ForecastInputOffsets option specified in '
                                            'the configuration file.')
            if len(self.fcst_input_offsets) != self.number_inputs:
                err_handler.err_out_screen('Please specify ForecastInputOffset values for each '
                                            'corresponding input forcings for forecasts.')
            # Check to make sure the input offset options make sense. There will be additional
            # checking later when input choices are mapped to input products.
            for inputOffset in self.fcst_input_offsets:
                if inputOffset < 0:
                    err_handler.err_out_screen(
                        'Please specify ForecastInputOffset values greater than or equal to zero.')

        # Calculate the length of the forecast cycle, based on the maximum
        # length of the input forcing length chosen by the user.
        self.cycle_length_minutes = max(self.fcst_input_horizons)

        # Ensure the number maximum cycle length is an equal divider of the output
        # time step specified by the user.
        if self.cycle_length_minutes % self.output_freq != 0:
            err_handler.err_out_screen('Please specify an output time step that is an equal divider of the '
                                        'maximum of the forecast time horizons specified.')
        
        if self.sub_output_hour == None:
            # Calculate the number of output time steps per forecast cycle.
            self.num_output_steps = int(self.cycle_length_minutes / self.output_freq)
            if self.precip_only_flag == True:
                self.num_supp_output_steps = int(self.cycle_length_minutes) / self.customSuppPcpFreq
            if self.ana_flag:
                self.actual_output_steps = np.int32(self.nFcsts)
            else:
                self.actual_output_steps = np.int32(self.num_output_steps)
        else:
            # Calculate the number of output time steps per forecast cycle.
            self.num_output_steps = int((self.cycle_length_minutes - (self.sub_output_hour*60))/ self.sub_output_freq) + int((self.sub_output_hour*60)/ self.output_freq) - 1
            if self.precip_only_flag == True:
                self.num_supp_output_steps = int(self.cycle_length_minutes) / self.customSuppPcpFreq
            if self.ana_flag:
                self.actual_output_steps = np.int32(self.nFcsts)
            else:
                self.actual_output_steps = np.int32(self.num_output_steps)


        # Process the grid type
        try:
            self.grid_type = cfg['GRID_TYPE']
        except KeyError:
            err_handler.err_out_screen('Unable to locate GRID_TYPE in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate GRID_TYPE in the configuration file.')
        if (self.grid_type.lower() != "gridded" and self.grid_type.lower() != "unstructured" and self.grid_type.lower() != "hydrofabric"):
            err_handler.err_out_screen('GRID_TYPE in the configuration file only accepts "unstructured", "gridded", or "hydrofabric" as options.')

        if(self.grid_type.lower() == "gridded"):
            # Process the geogrid variable information
            try:
                self.lon_var = cfg['LONVAR']
            except KeyError:
                err_handler.err_out_screen('Unable to locate LONVAR in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate LONVAR in the configuration file.')
            try:
                self.lat_var = cfg['LATVAR']
            except KeyError:
                err_handler.err_out_screen('Unable to locate LATVAR in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate LATVAR in the configuration file.')

        elif(self.grid_type.lower() == "unstructured"):
            # Process the geogrid variable information
            try:
                self.nodecoords_var = cfg['NodeCoords']
            except KeyError:
                err_handler.err_out_screen('Unable to locate NodeCoords for unstructured mesh in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate NodeCoords for unstructured mesh in the configuration file.')
            try:
                self.elemcoords_var = cfg['ElemCoords']
            except KeyError:
                err_handler.err_out_screen('Unable to locate ElemCoords for unstructured mesh in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate ElemCoords for unstructured mesh in the configuration file.')
            try:
                self.elemconn_var = cfg['ElemConn']
            except KeyError:
                err_handler.err_out_screen('Unable to locate ElemConn for unstructured mesh in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate ElemConn for unstructured mesh in the configuration file.')
            try:
                self.numelemconn_var = cfg['NumElemConn']
            except KeyError:
                err_handler.err_out_screen('Unable to locate NumElemConn for unstructured mesh in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate NumElemConn for unstructured mesh in the configuration file.')

        elif(self.grid_type.lower() == "hydrofabric"):
            # Process the geogrid variable information
            try:
                self.nodecoords_var = cfg['NodeCoords']
            except KeyError:
                err_handler.err_out_screen('Unable to locate NodeCoords for unstructured mesh in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate NodeCoords for unstructured mesh in the configuration file.')
            try:
                self.elemcoords_var = cfg['ElemCoords']
            except KeyError:
                err_handler.err_out_screen('Unable to locate ElemCoords for unstructured mesh in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate ElemCoords for unstructured mesh in the configuration file.')
            try:
                self.element_id_var = cfg['ElemID']
            except KeyError:
                err_handler.err_out_screen('Unable to locate ElemID for unstructured mesh in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate ElemID for unstructured mesh in the configuration file.')
            try:
                self.elemconn_var = cfg['ElemConn']
            except KeyError:
                err_handler.err_out_screen('Unable to locate ElemConn for unstructured mesh in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate ElemConn for unstructured mesh in the configuration file.')
            try:
                self.numelemconn_var = cfg['NumElemConn']
            except KeyError:
                err_handler.err_out_screen('Unable to locate NumElemConn for unstructured mesh in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate NumElemConn for unstructured mesh in the configuration file.')

        # Process geospatial information
        try:
            self.geogrid = cfg['GeogridIn']
        except KeyError:
            err_handler.err_out_screen('Unable to locate GeogridIn in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate GeogridIn in the configuration file.')

        # Check for the optional geospatial land metadata file.
        try:
            self.spatial_meta = cfg['SpatialMetaIn']
        except KeyError:
            err_handler.err_out_screen('Unable to locate SpatialMetaIn in the configuration file.')
        if len(self.spatial_meta) == 0:
            # No spatial metadata file found.
            self.spatial_meta = None
        else:
            if not os.path.isfile(self.spatial_meta):
                err_handler.err_out_screen('Unable to locate optional spatial metadata file: ' +
                                           self.spatial_meta)

        if self.precip_only_flag == False:
            # Check for the IgnoredBorderWidths
            try:
                self.ignored_border_widths = cfg['IgnoredBorderWidths']
            except (KeyError, configparser.NoOptionError):
                # if didn't specify, no worries, just set to 0
                self.ignored_border_widths = [0.0]*self.number_inputs
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper IgnoredBorderWidths option specified in the configuration file.'
                                           '({} was supplied'.format(cfg['Geospatial']['IgnoredBorderWidths']))
            if len(self.ignored_border_widths) != self.number_inputs:
                err_handler.err_out_screen('Please specify IgnoredBorderWidths values for each '
                                           'corresponding input forcings for SuppForcing.'
                                           '({} was supplied'.format(self.ignored_border_widths))
            if any(map(lambda x: x < 0, self.ignored_border_widths)):
                err_handler.err_out_screen('Please specify IgnoredBorderWidths values greater than or equal to zero:'
                                           '({} was supplied'.format(self.ignored_border_widths))

        if self.precip_only_flag == False:
            # Process regridding options.
            try:
                self.regrid_opt = cfg['RegridOpt']
            except KeyError:
                err_handler.err_out_screen('Unable to locate RegridOpt under the Regridding section '
                                           'in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate RegridOpt under the Regridding section '
                                           'in the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper RegridOpt options specified in the configuration file.')
            if len(self.regrid_opt) != self.number_inputs:
                err_handler.err_out_screen('Please specify RegridOpt values for each corresponding input '
                                           'forcings in the configuration file.')
            # Check to make sure regridding options makes sense.
            for regridOpt in self.regrid_opt:
                if regridOpt < 1 or regridOpt > 3:
                    err_handler.err_out_screen('Invalid RegridOpt chosen in the configuration file. Please choose a '
                                               'value of 1-2 for each corresponding input forcing.')
            try:
                # Read weight file directory (optional)
                self.weightsDir = cfg['RegridWeightsDir']
            except:
                # Assign the weights directory then to the temporary scratch directory
                self.weightsDir = self.scratch_dir
            if self.weightsDir is not None:
                # if we do have one specified, make sure it exists
                if not os.path.exists(self.weightsDir):
                    err_handler.err_out_screen('ESMF Weights file directory specifed ({}) but does not exist').format(
                        self.weightsDir)

        # Calculate the beginning/ending processing dates if we are running realtime
        if self.realtime_flag:
            time_handling.calculate_lookback_window(self)

        if self.precip_only_flag == False:
            # Read in temporal interpolation options.
            try:
                self.forceTemoralInterp = cfg['ForcingTemporalInterpolation']
            except KeyError:
                err_handler.err_out_screen('Unable to locate ForcingTemporalInterpolation under the Interpolation '
                                           'section in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate ForcingTemporalInterpolation under the Interpolation '
                                           'section in the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper ForcingTemporalInterpolation options specified in the '
                                           'configuration file.')
            if len(self.forceTemoralInterp) != self.number_inputs:
                err_handler.err_out_screen('Please specify ForcingTemporalInterpolation values for each '
                                           'corresponding input forcings in the configuration file.')
            # Ensure the forcingTemporalInterpolation values make sense.
            for temporalInterpOpt in self.forceTemoralInterp:
                if temporalInterpOpt < 0 or temporalInterpOpt > 2:
                    err_handler.err_out_screen('Invalid ForcingTemporalInterpolation chosen in the configuration file. '
                                               'Please choose a value of 0-2 for each corresponding input forcing.')

            # Read in the temperature downscaling options.
            # Create temporary array to hold flags of if we need input parameter files.
            param_flag = np.empty([len(self.input_forcings)], int)
            param_flag[:] = 0
            try:
                self.t2dDownscaleOpt = cfg['TemperatureDownscaling']
            except KeyError:
                err_handler.err_out_screen('Unable to locate TemperatureDownscaling under the Downscaling '
                                           'section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate TemperatureDownscaling under the Downscaling '
                                           'section of the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper TemperatureDownscaling options specified in the configuration file.')
            if len(self.t2dDownscaleOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify TemperatureDownscaling values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the downscaling options chosen make sense.
            count_tmp = 0
            for optTmp in self.t2dDownscaleOpt:
                if optTmp < 0 or optTmp > 2:
                    err_handler.err_out_screen('Invalid TemperatureDownscaling options specified in '
                                           'the configuration file.')
                if optTmp == 2:
                    param_flag[count_tmp] = 1
                count_tmp = count_tmp + 1

            # Read in the pressure downscaling options.
            try:
                self.psfcDownscaleOpt = cfg['PressureDownscaling']
            except KeyError:
                err_handler.err_out_screen('Unable to locate PressureDownscaling under the Downscaling '
                                           'section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate PressureDownscaling under the Downscaling '
                                           'section of the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper PressureDownscaling options specified in the configuration file.')
            if len(self.psfcDownscaleOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify PressureDownscaling values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the downscaling options chosen make sense.
            for optTmp in self.psfcDownscaleOpt:
                if optTmp < 0 or optTmp > 1:
                    err_handler.err_out_screen('Invalid PressureDownscaling options specified in the configuration file.')

            # Read in the shortwave downscaling options
            try:
                self.swDownscaleOpt = cfg['ShortwaveDownscaling']
            except KeyError:
                err_handler.err_out_screen('Unable to locate ShortwaveDownscaling under the Downscaling '
                                           'section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate ShortwaveDownscaling under the Downscaling '
                                           'section of the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper ShortwaveDownscaling options specified in the configuration file.')
            if len(self.swDownscaleOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify ShortwaveDownscaling values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the downscaling options chosen make sense.
            for optTmp in self.swDownscaleOpt:
                if optTmp < 0 or optTmp > 1:
                     err_handler.err_out_screen('Invalid ShortwaveDownscaling options specified in the configuration file.')

            # Read in humidity downscaling options.
            try:
                self.q2dDownscaleOpt = cfg['HumidityDownscaling']
            except KeyError:
                err_handler.err_out_screen('Unable to locate HumidityDownscaling under the Downscaling '
                                           'section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate HumidityDownscaling under the Downscaling '
                                           'section of the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper HumidityDownscaling options specified in the configuration file.')
            if len(self.q2dDownscaleOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify HumidityDownscaling values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the downscaling options chosen make sense.
            for optTmp in self.q2dDownscaleOpt:
                if optTmp < 0 or optTmp > 1:
                    err_handler.err_out_screen('Invalid HumidityDownscaling options specified in the configuration file.')

        # Read in the precipitation downscaling options
        try:
            self.precipDownscaleOpt = cfg['PrecipDownscaling']
        except KeyError:
            err_handler.err_out_screen('Unable to locate PrecipDownscaling under the Downscaling '
                                       'section of the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate PrecipDownscaling under the Downscaling '
                                       'section of the configuration file.')
        except json.decoder.JSONDecodeError:
            err_handler.err_out_screen('Improper PrecipDownscaling options specified in the configuration file.')
        if self.precip_only_flag == False:
            if len(self.precipDownscaleOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify PrecipDownscaling values for each corresponding '
                                           'input forcings in the configuration file.')
        # Ensure the downscaling options chosen make sense.
        count_tmp = 0
        for optTmp in self.precipDownscaleOpt:
            if optTmp < 0 or optTmp > 1:
                err_handler.err_out_screen('Invalid PrecipDownscaling options specified in the configuration file.')
            if optTmp == 1:
                param_flag[count_tmp] = 1
            count_tmp = count_tmp + 1

        # Read in the downscaling parameter directory.
        try:
            self.dScaleParamDirs = cfg['DownscalingParamDirs']
        except KeyError:
            err_handler.err_out_screen('Unable to locate DownscalingParamDirs in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate DownscalingParamDirs in the configuration file.')
        if len(self.dScaleParamDirs) != len(self.input_forcings):
            err_handler.err_out_screen('Please specify a downscaling parameter directory for each '
                                        'corresponding downscaling option that requires one.')
        # Loop through each downscaling parameter directory and make sure they exist.
        for dirTmp in range(0, len(self.dScaleParamDirs)):
            if not os.path.isdir(self.dScaleParamDirs[dirTmp]):
                err_handler.err_out_screen('Unable to locate parameter directory: ' + self.dScaleParamDirs[dirTmp])

        if([1] in self.q2dDownscaleOpt or [1] in self.swDownscaleOpt or [1] in self.psfcDownscaleOpt or [1,2] in self.t2dDownscaleOpt):
             # Process the geogrid information for downscaling
            try:
                self.sinalpha_var = cfg['SINALPHA']
            except:
                self.sinalpha_var = None
            try:
                self.cosalpha_var = cfg['COSALPHA']
            except:
                self.cosalpha_var = None
            if(self.grid_type.lower() == "hydrofabric"):
                try:
                    self.slope_var = cfg['SLOPE']
                except KeyError:
                    err_handler.err_out_screen('Unable to locate SLOPE variable in the hydrofabric configuration file. Required variable since user turned on a downscaling option.')
                except configparser.NoOptionError:
                    err_handler.err_out_screen('Unable to locate SLOPE variable in the hydrofabric configuration file. Required variable since user turned on a downscaling option.')
                try:
                    self.slope_azimuth_var = cfg['SLOPE_AZIMUTH']
                except KeyError:
                    err_handler.err_out_screen('Unable to locate SLOPE_AZIMUTH variable in the hydrofabric configuration file. Required variable since user turned on a downscaling option.')
                except configparser.NoOptionError:
                    err_handler.err_out_screen('Unable to locate SLOPE_AZIMUTH variable in the hydrofabric configuration file. Required variable since user turned on a downscaling option.')
            else:
                try:
                    self.slope_var = cfg['SLOPE']
                except:
                    self.slope_var = None
                try:
                    self.slope_azimuth_var = cfg['SLOPE_AZIMUTH']
                except:
                     self.slope_azimuth_var = None
                if(self.grid_type.lower() == "unstructured"):
                    try:
                        self.slope_var_elem = cfg['SLOPE_ELEM']
                    except:
                        self.slope_var_elem = None
                    try:
                        self.slope_azimuth_var_elem = cfg['SLOPE_AZIMUTH_ELEM']
                    except:
                         self.slope_azimuth_var_elem = None


            if(self.grid_type.lower() == "unstructured"):
                try:
                    self.hgt_elem_var = cfg['HGTVAR_ELEM']
                except KeyError:
                    err_handler.err_out_screen('Unable to locate HGTVAR_ELEM in the configuration file. Required variable since user turned on a downscaling option.')
                except configparser.NoOptionError:
                    err_handler.err_out_screen('Unable to locate HGTVAR_ELEM in the configuration file. Required variable since user turned on a downscaling option.')

            try:
                self.hgt_var = cfg['HGTVAR']
            except KeyError:
                err_handler.err_out_screen('Unable to locate HGTVAR in the configuration file. Required variable since user turned on a downscaling option.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate HGTVAR in the configuration file. Required variable since user turned on a downscaling option.')

        #   * Bias Correction Options *
        if self.precip_only_flag == False:
            # Read in temperature bias correction options
            try:
                self.t2BiasCorrectOpt = cfg['TemperatureBiasCorrection']
            except KeyError:
                err_handler.err_out_screen('Unable to locate TemperatureBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate TemperatureBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except json.JSONDecodeError:
                err_handler.err_out_screen('Improper TemperatureBiasCorrection options specified in '
                                           'the configuration file.')
            if len(self.t2BiasCorrectOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify TemperatureBiasCorrection values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.t2BiasCorrectOpt:
                if optTmp < 0 or optTmp > 4:
                    err_handler.err_out_screen('Invalid TemperatureBiasCorrection options specified in the '
                                               'configuration file.')

            # Read in surface pressure bias correction options.
            try:
                self.psfcBiasCorrectOpt = cfg['PressureBiasCorrection']
            except KeyError:
                err_handler.err_out_screen('Unable to locate PressureBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate PressureBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except json.JSONDecodeError:
                err_handler.err_out_screen('Improper PressureBiasCorrection options specified in the configuration file.')
            if len(self.psfcDownscaleOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify PressureBiasCorrection values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.psfcBiasCorrectOpt:
                if optTmp < 0 or optTmp > 1:
                    err_handler.err_out_screen('Invalid PressureBiasCorrection options specified in the '
                                               'configuration file.')
                if optTmp == 1:
                    # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                    self.runCfsNldasBiasCorrect = True

            # Read in humidity bias correction options.
            try:
                self.q2BiasCorrectOpt = cfg['HumidityBiasCorrection']
            except KeyError:
                err_handler.err_out_screen('Unable to locate HumidityBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate HumidityBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except json.JSONDecodeError:
                err_handler.err_out_screen('Improper HumdityBiasCorrection options specified in the configuration file.')
            if len(self.q2BiasCorrectOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify HumidityBiasCorrection values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.q2BiasCorrectOpt:
                if optTmp < 0 or optTmp > 2:
                    err_handler.err_out_screen('Invalid HumidityBiasCorrection options specified in the '
                                           'configuration file.')
                if optTmp == 1:
                    # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                    self.runCfsNldasBiasCorrect = True

            # Read in wind bias correction options.
            try:
                self.windBiasCorrect = cfg['WindBiasCorrection']
            except KeyError:
                err_handler.err_out_screen('Unable to locate WindBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate WindBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except json.JSONDecodeError:
                err_handler.err_out_screen('Improper WindBiasCorrection options specified in the configuration file.')
            if len(self.windBiasCorrect) != self.number_inputs:
                err_handler.err_out_screen('Please specify WindBiasCorrection values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.windBiasCorrect:
                if optTmp < 0 or optTmp > 4:
                    err_handler.err_out_screen('Invalid WindBiasCorrection options specified in the configuration file.')
                if optTmp == 1:
                    # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                    self.runCfsNldasBiasCorrect = True

            # Read in shortwave radiation bias correction options.
            try:
                self.swBiasCorrectOpt = cfg['SwBiasCorrection']
            except KeyError:
                err_handler.err_out_screen('Unable to locate SwBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate SwBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except json.JSONDecodeError:
                err_handler.err_out_screen('Improper SwBiasCorrection options specified in the configuration file.')
            if len(self.swBiasCorrectOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify SwBiasCorrection values for each corresponding '
                                           'input forcings in the configuration file.')
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.swBiasCorrectOpt:
                if optTmp < 0 or optTmp > 2:
                    err_handler.err_out_screen('Invalid SwBiasCorrection options specified in the configuration file.')
                if optTmp == 1:
                    # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                    self.runCfsNldasBiasCorrect = True

            # Read in longwave radiation bias correction options.
            try:
                self.lwBiasCorrectOpt = cfg['LwBiasCorrection']
            except KeyError:
                err_handler.err_out_screen('Unable to locate LwBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate LwBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except json.JSONDecodeError:
                err_handler.err_out_screen('Improper LwBiasCorrection options specified in '
                                           'the configuration file.')
            if len(self.lwBiasCorrectOpt) != self.number_inputs:
                err_handler.err_out_screen('Please specify LwBiasCorrection values for each corresponding '
                                       'input forcings in the configuration file.')
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.lwBiasCorrectOpt:
                if optTmp < 0 or optTmp > 4:
                    err_handler.err_out_screen('Invalid LwBiasCorrection options specified in the configuration file.')
                if optTmp == 1:
                    # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                    self.runCfsNldasBiasCorrect = True

            # Read in precipitation bias correction options.
            try:
                self.precipBiasCorrectOpt = cfg['PrecipBiasCorrection']
            except KeyError:
                err_handler.err_out_screen('Unable to locate PrecipBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate PrecipBiasCorrection under the '
                                           'BiasCorrection section of the configuration file.')
            except json.JSONDecodeError:
                err_handler.err_out_screen('Improper PrecipBiasCorrection options specified in the configuration file.')
            if self.precip_only_flag == False:
                if len(self.precipBiasCorrectOpt) != self.number_inputs:
                    err_handler.err_out_screen('Please specify PrecipBiasCorrection values for each corresponding '
                                               'input forcings in the configuration file.')
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.precipBiasCorrectOpt:
                if optTmp < 0 or optTmp > 1:
                    err_handler.err_out_screen('Invalid PrecipBiasCorrection options specified in the configuration file.')
                if optTmp == 1:
                    # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                    self.runCfsNldasBiasCorrect = True

            # Putting a constraint here that CFSv2-NLDAS bias correction (NWM only) is chosen, it must be turned on
            # for ALL variables.
            if self.runCfsNldasBiasCorrect:
                if min(self.precipBiasCorrectOpt) != 1 and max(self.precipBiasCorrectOpt) != 1:
                    err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                               'Precipitation under this configuration.')
                if min(self.lwBiasCorrectOpt) != 1 and max(self.lwBiasCorrectOpt) != 1:
                    err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                               'long-wave radiation under this configuration.')
                if min(self.swBiasCorrectOpt) != 1 and max(self.swBiasCorrectOpt) != 1:
                    err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                               'short-wave radiation under this configuration.')
                if min(self.t2BiasCorrectOpt) != 1 and max(self.t2BiasCorrectOpt) != 1:
                    err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                               'surface temperature under this configuration.')
                if min(self.windBiasCorrect) != 1 and max(self.windBiasCorrect) != 1:
                    err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                               'wind forcings under this configuration.')
                if min(self.q2BiasCorrectOpt) != 1 and max(self.q2BiasCorrectOpt) != 1:
                    err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                               'specific humidity under this configuration.')
                if min(self.psfcBiasCorrectOpt) != 1 and max(self.psfcBiasCorrectOpt) != 1:
                    err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction must be activated for '
                                               'surface pressure under this configuration.')
                # Make sure we don't have any other forcings activated. This can only be ran for CFSv2.
                for optTmp in self.input_forcings:
                    if optTmp != 7:
                        err_handler.err_out_screen('CFSv2-NLDAS NWM bias correction can only be used in '
                                                   'CFSv2-only configurations')

        # Read in supplemental precipitation options as an array of values to map.
        try:
            self.supp_precip_forcings = cfg['SuppPcp']
        except KeyError:
            err_handler.err_out_screen('Unable to locate SuppPcp under SuppForcing section in configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate SuppPcp under SuppForcing section in configuration file.')
        except json.decoder.JSONDecodeError:
            err_handler.err_out_screen('Improper SuppPcp option specified in configuration file')
        self.number_supp_pcp = len(self.supp_precip_forcings)

        # Read in the supp pcp types (GRIB[1|2], NETCDF)
        try:
            self.supp_precip_file_types = cfg['SuppPcpForcingTypes']
            self.supp_precip_file_types = [stype.strip() for stype in self.supp_precip_file_types]
            if self.supp_precip_file_types == ['']:
                self.supp_precip_file_types = []
        except KeyError:
            err_handler.err_out_screen('Unable to locate SuppPcpForcingTypes in SuppForcing section '
                                       'in the configuration file.')
        except configparser.NoOptionError:
            err_handler.err_out_screen('Unable to locate SuppPcpForcingTypes in SuppForcing section '
                                       'in the configuration file.')
        if len(self.supp_precip_file_types) != self.number_supp_pcp:
            err_handler.err_out_screen('Number of SuppPcpForcingTypes ({}) must match the number '
                                       'of SuppPcp inputs ({}) in the configuration file.'.format(len(self.supp_precip_file_types), self.number_supp_pcp))
        for fileType in self.supp_precip_file_types:
            if fileType not in ['GRIB1', 'GRIB2', 'NETCDF']:
                err_handler.err_out_screen('Invalid SuppForcing file type "{}" specified. '
                                   'Only GRIB1, GRIB2, and NETCDF are supported'.format(fileType))

        if self.number_supp_pcp > 0:
            # Check to make sure supplemental precip options make sense. Also read in the RQI threshold
            # if any radar products where chosen.
            for suppOpt in self.supp_precip_forcings:
                if suppOpt < 0 or suppOpt > 14:
                    err_handler.err_out_screen('Please specify SuppForcing values between 1 and 14.')
                # Read in RQI threshold to apply to radar products.
                if suppOpt in (1,2,7,10,11,12):
                    try:
                        self.rqiMethod = cfg['RqiMethod']
                    except KeyError:
                        err_handler.err_out_screen('Unable to locate RqiMethod under SuppForcing '
                                                   'section in the configuration file.')
                    except configparser.NoOptionError:
                        err_handler.err_out_screen('Unable to locate RqiMethod under SuppForcing '
                                                   'section in the configuration file.')
                    except json.decoder.JSONDecodeError:
                        err_handler.err_out_screen('Improper RqiMethod option in the configuration file.')

                    # Check that if we have more than one RqiMethod, it's the correct number
                    if type(self.rqiMethod) is list:
                        if len(self.rqiMethod) != self.number_supp_pcp:
                            err_handler.err_out_screen('Number of RqiMethods ({}) must match the number '
                                                       'of SuppPcp inputs ({}) in the configuration file, or '
                                                       'supply a single method for all inputs'.format(
                                                        len(self.rqiMethod), self.number_supp_pcp))
                    elif type(self.rqiMethod) is int:
                        # Support 'classic' mode of single method
                        self.rqiMethod = [self.rqiMethod] * self.number_supp_pcp

                    # Make sure the RqiMethod(s) makes sense.
                    for method in self.rqiMethod:
                        if method < 0 or method > 2:
                            err_handler.err_out_screen('Please specify RqiMethods of either 0, 1, or 2.')

                    try:
                        self.rqiThresh = cfg['RqiThreshold']
                    except KeyError:
                        err_handler.err_out_screen('Unable to locate RqiThreshold under '
                                                   'SuppForcing section in the configuration file.')
                    except configparser.NoOptionError:
                        err_handler.err_out_screen('Unable to locate RqiThreshold under '
                                                   'SuppForcing section in the configuration file.')
                    except json.decoder.JSONDecodeError:
                        err_handler.err_out_screen('Improper RqiThreshold option in the configuration file.')

                    # Check that if we have more than one RqiThreshold, it's the correct number
                    if type(self.rqiThresh) is list:
                        if len(self.rqiThresh) != self.number_supp_pcp:
                            err_handler.err_out_screen('Number of RqiThresholds ({}) must match the number '
                                                       'of SuppPcp inputs ({}) in the configuration file, or '
                                                       'supply a single threshold for all inputs'.format(
                                                        len(self.rqiThresh), self.number_supp_pcp))
                    elif type(self.rqiThresh) is float:
                        # Support 'classic' mode of single threshold
                        self.rqiThresh = [self.rqiThresh] * self.number_supp_pcp

                    # Make sure the RQI threshold makes sense.
                    for threshold in self.rqiThresh:
                        if threshold < 0.0 or threshold > 1.0:
                            err_handler.err_out_screen('Please specify RqiThresholds between 0.0 and 1.0.')

            # Read in the input directories for each supplemental precipitation product.
            try:
                self.supp_precip_dirs = cfg['SuppPcpDirectories']
            except KeyError:
                err_handler.err_out_screen('Unable to locate SuppPcpDirectories in SuppForcing section '
                                           'in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate SuppPcpDirectories in SuppForcing section '
                                           'in the configuration file.')
        
            # Loop through and ensure all supp pcp directories exist. Also strip out any whitespace
            # or new line characters.
            for dirTmp in range(0, len(self.supp_precip_dirs)):
                self.supp_precip_dirs[dirTmp] = self.supp_precip_dirs[dirTmp].strip()
                if not os.path.isdir(self.supp_precip_dirs[dirTmp]):
                    err_handler.err_out_screen('Unable to locate supp pcp directory: ' + self.supp_precip_dirs[dirTmp])

            #Special case for ExtAnA where we treat comma separated stage IV, MRMS data as one SuppPcp input 
            if (11 in self.supp_precip_forcings or 12 in self.supp_precip_forcings):
                if len(self.supp_precip_forcings) != 1:
                    err_handler.err_out_screen('CONUS or Alaska Stage IV/MRMS SuppPcp option is only supported as a standalone option')
                self.supp_precip_dirs = [",".join(self.supp_precip_dirs)]

            if len(self.supp_precip_dirs) != self.number_supp_pcp:
                err_handler.err_out_screen('Number of SuppPcpDirectories must match the number '
                                           'of SuppForcing in the configuration file.')

            # Process supplemental precipitation enforcement options
            try:
                self.supp_precip_mandatory = cfg['SuppPcpMandatory']
            except KeyError:
                err_handler.err_out_screen('Unable to locate SuppPcpMandatory under the SuppForcing section '
                                           'in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate SuppPcpMandatory under the SuppForcing section '
                                           'in the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper SuppPcpMandatory options specified in the configuration file.')
            if len(self.supp_precip_mandatory) != self.number_supp_pcp:
                err_handler.err_out_screen('Please specify SuppPcpMandatory values for each corresponding '
                                           'supplemental precipitation options in the configuration file.')
            # Check to make sure enforcement options makes sense.
            for enforceOpt in self.supp_precip_mandatory:
                if enforceOpt < 0 or enforceOpt > 1:
                    err_handler.err_out_screen('Invalid SuppPcpMandatory chosen in the configuration file. '
                                               'Please choose a value of 0 or 1 for each corresponding '
                                               'supplemental precipitation product.')

            # Read in the regridding options.
            try:
                self.regrid_opt_supp_pcp = cfg['RegridOptSuppPcp']
            except KeyError:
                err_handler.err_out_screen('Unable to locate RegridOptSuppPcp under the SuppForcing section '
                                           'in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate RegridOptSuppPcp under the SuppForcing section '
                                           'in the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper RegridOptSuppPcp options specified in the configuration file.')
            if len(self.regrid_opt_supp_pcp) != self.number_supp_pcp:
                err_handler.err_out_screen('Please specify RegridOptSuppPcp values for each corresponding supplemental '
                                           'precipitation product in the configuration file.')
            # Check to make sure regridding options makes sense.
            for regridOpt in self.regrid_opt_supp_pcp:
                if regridOpt < 1 or regridOpt > 3:
                    err_handler.err_out_screen('Invalid RegridOptSuppPcp chosen in the configuration file. '
                                               'Please choose a value of 1-3 for each corresponding '
                                               'supplemental precipitation product.')

            # Read in temporal interpolation options.
            try:
                self.suppTemporalInterp = cfg['SuppPcpTemporalInterpolation']
            except KeyError:
                err_handler.err_out_screen('Unable to locate SuppPcpTemporalInterpolation under the SuppForcing '
                                           'section in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate SuppPcpTemporalInterpolation under the SuppForcing '
                                           'section in the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper SuppPcpTemporalInterpolation options specified in the '
                                           'configuration file.')
            if len(self.suppTemporalInterp) != self.number_supp_pcp:
                err_handler.err_out_screen('Please specify SuppPcpTemporalInterpolation values for each '
                                           'corresponding supplemental precip products in the configuration file.')
            # Ensure the SuppPcpTemporalInterpolation values make sense.
            for temporalInterpOpt in self.suppTemporalInterp:
                if temporalInterpOpt < 0 or temporalInterpOpt > 2:
                    err_handler.err_out_screen('Invalid SuppPcpTemporalInterpolation chosen in the configuration file. '
                                               'Please choose a value of 0-2 for each corresponding input forcing')

            # Read in max time option
            try:
                self.supp_pcp_max_hours = cfg['SuppPcpMaxHours']
            except (KeyError, configparser.NoOptionError):
                self.supp_pcp_max_hours = None      # if missing, don't care, just assume all time

            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper SuppPcpMaxHours options specified in the '
                                           'configuration file.')

            if type(self.supp_pcp_max_hours) is list:
                if len(self.supp_pcp_max_hours) != self.number_supp_pcp:
                    err_handler.err_out_screen('Number of SuppPcpMaxHours ({}) must match the number '
                                               'of SuppPcp inputs ({}) in the configuration file, or '
                                               'supply a single threshold for all inputs'.format(
                            len(self.supp_pcp_max_hours), self.number_supp_pcp))
            elif type(self.supp_pcp_max_hours) is float:
                # Support 'classic' mode of single threshold
                self.supp_pcp_max_hours = [self.supp_pcp_max_hours] * self.number_supp_pcp

            # Read in the SuppPcpInputOffsets options.
            try:
                self.supp_input_offsets = cfg['SuppPcpInputOffsets']
            except KeyError:
                err_handler.err_out_screen('Unable to locate SuppPcpInputOffsets under SuppForcing '
                                           'section in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate SuppPcpInputOffsets under SuppForcing '
                                           'section in the configuration file.')
            except json.decoder.JSONDecodeError:
                err_handler.err_out_screen('Improper SuppPcpInputOffsets option specified in '
                                           'the configuration file.')
            if len(self.supp_input_offsets) != self.number_supp_pcp:
                err_handler.err_out_screen('Please specify SuppPcpInputOffsets values for each '
                                           'corresponding input forcings for SuppForcing.')
            # Check to make sure the input offset options make sense. There will be additional
            # checking later when input choices are mapped to input products.
            for inputOffset in self.supp_input_offsets:
                if inputOffset < 0:
                    err_handler.err_out_screen(
                            'Please specify SuppPcpInputOffsets values greater than or equal to zero.')

            # Read in the optional parameter directory for supplemental precipitation.
            try:
                self.supp_precip_param_dir = cfg['SuppPcpParamDir']
            except KeyError:
                err_handler.err_out_screen('Unable to locate SuppPcpParamDir under the SuppForcing section '
                                           'in the configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate SuppPcpParamDir under the SuppForcing section '
                                           'in the configuration file.')
            except ValueError:
                err_handler.err_out_screen('Improper SuppPcpParamDir option specified in the configuration file.')
            if not os.path.isdir(self.supp_precip_param_dir):
                err_handler.err_out_screen('Unable to locate SuppPcpParamDir: ' + self.supp_precip_param_dir)

        if self.precip_only_flag == False:
            # Read in Ensemble information
            # Read in CFS ensemble member information IF we have chosen CFSv2 as an input
            # forcing.
            for optTmp in self.input_forcings:
                if optTmp == 7:
                    try:
                        self.cfsv2EnsMember = cfg['cfsEnsNumber']
                    except KeyError:
                        err_handler.err_out_screen('Unable to locate cfsEnsNumber under the Ensembles '
                                                   'section of the configuration file')
                    except configparser.NoOptionError:
                        err_handler.err_out_screen('Unable to locate cfsEnsNumber under the Ensembles '
                                                   'section of the configuration file')
                    except json.JSONDecodeError:
                        err_handler.err_out_screen('Improper cfsEnsNumber options specified in the configuration file')
                    if self.cfsv2EnsMember < 1 or self.cfsv2EnsMember > 4:
                        err_handler.err_out_screen('Please chose an cfsEnsNumber value of 1,2,3 or 4.')

            # Read in information for the custom input NetCDF files that are to be processed.
            # Read in the ForecastInputHorizons options.
            try:
                self.customFcstFreq = cfg['custom_input_fcst_freq']
            except KeyError:
                err_handler.err_out_screen('Unable to locate custom_input_fcst_freq under Custom section in '
                                           'configuration file.')
            except configparser.NoOptionError:
                err_handler.err_out_screen('Unable to locate custom_input_fcst_freq under Custom section in '
                                           'configuration file.')
            except json.decoder.JSONDecodeError as je:
                err_handler.err_out_screen('Improper custom_input_fcst_freq  option specified in '
                                           'configuration file: ' + str(je))
            if len(self.customFcstFreq) != self.number_custom_inputs:
                err_handler.err_out_screen(f'Improper custom_input fcst_freq specified. '
                                           f'This number ({len(self.customFcstFreq)}) must '
                                           f'match the frequency of custom input forcings selected '
                                           f'({self.number_custom_inputs}).')

    @property
    def use_data_at_current_time(self):
        if self.supp_pcp_max_hours is not None:
            hrs_since_start = self.current_output_date - self.current_fcst_cycle
            return hrs_since_start <= datetime.timedelta(hours = self.supp_pcp_max_hours)
        else:
            return True 
