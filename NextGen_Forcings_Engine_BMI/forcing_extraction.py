import argparse
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import yaml

from Forcing_Extraction_Scripts.forecast_download_base import (
    FixedFileDownloader,
    ForecastDownloader,
    ScrapedFileDownloader,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.config import (
    ConfigOptions,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.consts import (
    FORCING_EXTRACTION,
)


def retrieve_forcing(config_options: ConfigOptions):
    """Download forecast forcing data based on requested sources in the forcing engine configuration file.

    :param config_options: dictionary of forcing engine config parameters
    """
    # Get parameters from the forcing engine config file
    refcstbdate = config_options.b_date_proc
    input_forcings = config_options.input_forcings + [
        f"supp{val}" for val in config_options.supp_precip_forcings
    ]
    if config_options.supp_precip_dirs is not None:
        input_forcing_dirs = (
            config_options.input_force_dirs + config_options.supp_precip_dirs
        )
    else:
        input_forcing_dirs = config_options.input_force_dirs
    input_horizons = config_options.fcst_input_horizons
    input_horizons = input_horizons + [input_horizons[0]] * len(
        config_options.supp_precip_forcings
    )
    ens_number = config_options.cfsv2EnsMember
    ana_flag = config_options.ana_flag
    look_back = config_options.look_back

    # Extract forcing data from appropriate sources
    for i in range(len(input_forcings)):
        # print(f"i: {i}, input_forcings[i]: {input_forcings[i]}, input_horizons[i]: {input_horizons[i]}, ")

        # Format extraction path
        extract_out_path = input_forcing_dirs[i]

        # Set supp forcing hours timehandling placeholder
        supp_forcing_hours = None

        # Set lookback hours and extraction scripts
        if ana_flag == 0:
            lookback_hours = 1
            forcing_script = FORCING_EXTRACTION["forcing_src"].get(input_forcings[i])
            forcing_start_time = refcstbdate + timedelta(hours=1)
        elif ana_flag == 1:
            lookback_hours = int(look_back / 60)
            forcing_start_time = refcstbdate + timedelta(hours=(lookback_hours - 1))
            if input_forcings[i] in (
                "supp1",
                "supp2",
                "supp6",
                "supp10",
                "supp11",
                "supp12",
            ):
                supp_forcing_hours = 1
            else:
                supp_forcing_hours = 0
            forcing_script = FORCING_EXTRACTION["forcing_ana_src"].get(
                input_forcings[i]
            )

        # Set path to extraction script
        extract_script_path = (
            Path(FORCING_EXTRACTION["extraction_script_path"]) / forcing_script
        )

        # Dynamically import extraction module
        mod_name = f"forcing_{Path(extract_script_path).stem}"
        spec = importlib.util.spec_from_file_location(mod_name, extract_script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Retrieve correct ForecastDownloader subclass from extraction script
        base_classes = (ForecastDownloader, FixedFileDownloader, ScrapedFileDownloader)
        downloader_class = next(
            obj
            for _, obj in vars(module).items()
            if isinstance(obj, type)
            and issubclass(obj, base_classes)
            and obj not in base_classes
        )

        if ana_flag == 1:
            forcing_start_time = forcing_start_time + timedelta(hours=1)

        if supp_forcing_hours is not None:
            forcing_start_time += timedelta(hours=supp_forcing_hours)

        if ana_flag == 1:
            lookback_hours = lookback_hours + 1

        # Format forcing extraction command
        downloader = downloader_class(
            out_dir=extract_out_path,
            start_time=forcing_start_time,
            lookback_hours=lookback_hours,
            cleanback_hours=0,
            lagback_hours=0,
            ens_number=int(ens_number) if ens_number not in ("", None) else None,
            input_horizon=input_horizons[i] if input_horizons[i] > 0 else None,
        )

        # Run the download
        downloader.run()


def main():
    """Extract forcing data."""
    parser = argparse.ArgumentParser(description="Download forecast forcing data")
    parser.add_argument("config_options", help="Path to YAML config file")
    args = parser.parse_args()

    # Load Yaml into dict
    with open(args.config_options, "r") as f:
        config_options_dict = yaml.safe_load(f)

    # Wrap config dict into simplenamespace to match ConfigOptions format
    config_options = SimpleNamespace(
        b_date_proc=datetime.strptime(
            config_options_dict["RefcstBDateProc"], "%Y-%m-%d %H:%M:%S"
        ),
        input_forcings=config_options_dict["InputForcings"],
        supp_precip_forcings=config_options_dict["SuppPcp"],
        input_force_dirs=config_options_dict["InputForcingDirectories"],
        supp_precip_dirs=config_options_dict["SuppPcpDirectories"],
        fcst_input_horizons=config_options_dict["ForecastInputHorizons"],
        cfsv2EnsMember=config_options_dict["cfsEnsNumber"],
        ana_flag=config_options_dict["AnAFlag"],
        look_back=config_options_dict["LookBack"],
    )

    # Extract forcing data
    retrieve_forcing(config_options)


if __name__ == "__main__":
    main()
