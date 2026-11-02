from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
import typing
import yaml

from config_models import ForcingExtractionConfig
from Forcing_Extraction_Scripts.forecast_download_base import (
    FixedFileDownloader,
    ForecastDownloader,
)


def retrieve_forcing(cfg: ForcingExtractionConfig) -> None:
    """Download forecast forcing data based on requested sources in configuration."""
    refcstbdate: datetime = cfg.b_date_proc
    input_forcings: list[typing.Any] = cfg.input_forcings + [
        f"supp{val}" for val in cfg.supp_precip_forcings
    ]
    if cfg.supp_precip_dirs is not None:
        input_forcing_dirs: list[str] = cfg.input_force_dirs + cfg.supp_precip_dirs
    else:
        input_forcing_dirs = list(cfg.input_force_dirs)
    input_horizons: list[int] = cfg.fcst_input_horizons
    input_horizons = input_horizons + [input_horizons[0]] * len(cfg.supp_precip_forcings)

    if "supp16" in input_forcings:
        index: int = input_forcings.index("supp16")
        input_forcings.append("supp2")
        input_forcing_dirs.append(f"{input_forcing_dirs[index]}/MRMS_Hourly")
        input_forcing_dirs[index] = f"{input_forcing_dirs[index]}/MRMS_SubHourly"
        input_horizons.append(input_horizons[index])

    ens_number: typing.Optional[typing.Union[int, str]] = cfg.cfsv2EnsMember
    ana_flag: int = cfg.ana_flag
    look_back: int = cfg.look_back
    extraction_scriptPath: str = "./Forcing_Extraction_Scripts"

    max_download_attempts: int = cfg.max_download_attempts
    download_attempt_interval: int = cfg.download_attempt_interval
    check_file_availability: int = cfg.check_file_availability

    forcing_src: dict[int | str, str] = {
        3: "Global/get_prod_GFS.py",
        5: "CONUS/get_conus_HRRR.py",
        6: "CONUS/get_conus_RAP.py",
        7: "Global/get_CFSv2.py",
        13: "Hawaii/get_prod_NAM_Nest_Hawaii.py",
        14: "Puerto_Rico/get_prod_NAM_Nest_Puerto_Rico.py",
        19: "Alaska/get_Alaska_HRRR.py",
        24: "CONUS/get_prod_NBM_Conus.py",
        26: "CONUS/get_conus_HRRR_subhourly.py",
        "supp8": "CONUS/get_prod_NBM_Conus.py",
        "supp9": "Alaska/get_prod_NBM_Alaska.py",
        "supp11": "Alaska/get_Alaska_StageIV.py",
        "supp12": "CONUS/get_conus_StageIV.py",
        "supp15": "Puerto_Rico/get_prod_NBM_Puerto_Rico.py",
    }
