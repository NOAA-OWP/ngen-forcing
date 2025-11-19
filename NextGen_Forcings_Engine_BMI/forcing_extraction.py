import argparse
import importlib.util
import yaml
from datetime import datetime, timedelta
from types import SimpleNamespace
from pathlib import Path
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader, FixedFileDownloader, ScrapedFileDownloader


def retrieve_forcing(cfg: 'ConfigOptions'):
    """
    Download forecast forcing data based on requested sources in the forcing engine configuration file

    :param cfg: dictionary of forcing engine config parameters
    """
    # Get parameters from the forcing engine config file
    refcstbdate = cfg.b_date_proc
    input_forcings = cfg.input_forcings + [f"supp{val}" for val in cfg.supp_precip_forcings]
    if cfg.supp_precip_dirs is not None:
        input_forcing_dirs = cfg.input_force_dirs + cfg.supp_precip_dirs
    else:
        input_forcing_dirs = cfg.input_force_dirs
    input_horizons = cfg.fcst_input_horizons
    input_horizons = input_horizons + [input_horizons[0]] * len(cfg.supp_precip_forcings)
    ens_number = cfg.cfsv2EnsMember
    ana_flag = cfg.ana_flag
    look_back = cfg.look_back
    extraction_scriptPath = "/ngen-app/ngen-forcing/Forcing_Extraction_Scripts"

    # Set mapping between InputForcings codes and forcing extraction scripts
    forcing_src = {3: "Global/get_prod_GFS.py",
                   5: "CONUS/get_conus_HRRR.py",
                   6: "CONUS/get_conus_RAP.py",
                   7: "Global/get_CFSv2.py",
                   13: "Hawaii/get_prod_NAM_Nest_Hawaii.py",
                   14: "Puerto_Rico/get_prod_NAM_Nest_Puerto_Rico.py",
                   19: "Alaska/get_Alaska_HRRR.py",
                   24: "CONUS/get_prod_NBM_Conus.py",
                   "supp8": "CONUS/get_prod_NBM_Conus.py",
                   "supp9": "Alaska/get_prod_NBM_Alaska.py",
                   "supp11": "Alaska/get_Alaska_StageIV.py",
                   "supp12": "CONUS/get_conus_StageIV.py",
                   "supp15": "Puerto_Rico/get_prod_NBM_Puerto_Rico.py"
                   }

    # Set mapping between InputForcings codes and forcing extraction scripts
    forcing_ana_src = {5: "CONUS/get_conus_HRRR_AnA.py",
                       6: "CONUS/get_conus_RAP_AnA.py",
                       13: "Hawaii/get_prod_NAM_Nest_Hawaii.py",
                       14: "Puerto_Rico/get_prod_NAM_Nest_Puerto_Rico.py",
                       19: "Alaska/get_Alaska_HRRR.py",
                       20: "Alaska/get_Alaska_HRRR_AnA.py",
                       "supp1": "CONUS/get_conus_MRMS_Radar.py",
                       "supp2": "CONUS/get_conus_MRMS_MultiSensor.py",
                       "supp6": "Hawaii/get_MRMS_MultiSensor_Hawaii.py",
                       "supp10": "Alaska/get_MRMS_MultiSensor_Alaska.py",
                       "supp11": "Alaska/get_Alaska_StageIV.py",
                       "supp12": "CONUS/get_conus_StageIV.py"}

    # Extract forcing data from appropriate sources
    for i in range(len(input_forcings)):

        #print(f"i: {i}, input_forcings[i]: {input_forcings[i]}, input_horizons[i]: {input_horizons[i]}, ")

        # Format extraction path
        extract_outPath = input_forcing_dirs[i]

        # Set supp forcing hours timehandling placeholder
        supp_forcing_hours = None

        # Set lookback hours and extraction scripts
        if ana_flag == 0:
            look_back_hours = 1
            forcing_script = forcing_src.get(input_forcings[i])
            forcing_start_time = refcstbdate + timedelta(hours=1)
        elif ana_flag == 1:
            look_back_hours = int(look_back / 60)
            forcing_start_time = refcstbdate + timedelta(hours=(look_back_hours -1))
            if input_forcings[i] in ("supp1", "supp2", "supp6", "supp10", "supp11", "supp12"):
                supp_forcing_hours = 1
            else:
                supp_forcing_hours = 0
            forcing_script = forcing_ana_src.get(input_forcings[i])

        # Set path to extraction script
        extract_scriptPath = Path(extraction_scriptPath) / forcing_script

        # Dynamically import extraction module
        mod_name = f"forcing_{Path(extract_scriptPath).stem}"
        spec = importlib.util.spec_from_file_location(mod_name, extract_scriptPath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Retrieve correct ForecastDownloader subclass from extraction script
        base_classes = (ForecastDownloader, FixedFileDownloader, ScrapedFileDownloader)
        downloader_class = next(
            obj for name, obj in vars(module).items()
            if isinstance(obj, type) and issubclass(obj, base_classes) and obj not in base_classes
        )

        start_delt = None
        lag_delt = None
        
        if ana_flag == 1:
            start_delt = timedelta(hours=1)

        if supp_forcing_hours is not None:
            start_delt += timedelta(hours=supp_forcing_hours)

        if ana_flag == 1:
            lag_delt = 1

        # Format forcing extraction command
        downloader = downloader_class(
            out_dir=extract_outPath,
            start_time=forcing_start_time + start_delt if start_delt else forcing_start_time,
            lookback_hours=look_back_hours + lag_delt if lag_delt else look_back_hours,
            cleanback_hours=0,
            lagback_hours=0,
            ens_number=int(ens_number) if ens_number not in ('',None) else None
        )

        # Run the download
        downloader.run()


def main():
    parser = argparse.ArgumentParser(description="Download forecast forcing data")
    parser.add_argument("cfg", help="Path to YAML config file")
    args = parser.parse_args()

    # Load Yaml into dict
    with open(args.cfg, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Wrap config dict into simplenamespace to match ConfigOptions format
    cfg = SimpleNamespace(b_date_proc=datetime.strptime(cfg_dict['RefcstBDateProc'], "%Y-%m-%d %H:%M:%S"),
                          input_forcings=cfg_dict['InputForcings'],
                          supp_precip_forcings=cfg_dict['SuppPcp'],
                          input_force_dirs=cfg_dict['InputForcingDirectories'],
                          supp_precip_dirs=cfg_dict['SuppPcpDirectories'],
                          fcst_input_horizons=cfg_dict['ForecastInputHorizons'],
                          cfsv2EnsMember=cfg_dict['cfsEnsNumber'],
                          ana_flag=cfg_dict['AnAFlag'],
                          look_back=cfg_dict['LookBack'])

    # Extract forcing data
    retrieve_forcing(cfg)


if __name__ == "__main__":
    main()
