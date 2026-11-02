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

    forcing_ana_src: dict[int | str, str] = {
        5: "CONUS/get_conus_HRRR_AnA.py",
        6: "CONUS/get_conus_RAP_AnA.py",
        13: "Hawaii/get_prod_NAM_Nest_Hawaii.py",
        14: "Puerto_Rico/get_prod_NAM_Nest_Puerto_Rico_AnA.py",
        19: "Alaska/get_Alaska_HRRR.py",
        20: "Alaska/get_Alaska_HRRR_AnA.py",
        26: "CONUS/get_conus_HRRR_subhourly_AnA.py",
        "supp1": "CONUS/get_conus_MRMS_Radar.py",
        "supp2": "CONUS/get_conus_MRMS_MultiSensor.py",
        "supp6": "Hawaii/get_MRMS_MultiSensor_Hawaii.py",
        "supp10": "Alaska/get_MRMS_MultiSensor_Alaska.py",
        "supp11": "Alaska/get_Alaska_StageIV.py",
        "supp12": "CONUS/get_conus_StageIV.py",
        "supp15": "Puerto_Rico/get_prod_NBM_Puerto_Rico_AnA.py",
        "supp16": "CONUS/get_conus_MRMS_subhourly.py",
    }

    for i in range(len(input_forcings)):
        extract_outPath: str = input_forcing_dirs[i]
        supp_forcing_hours: int | None = None

        if ana_flag == 0:
            look_back_hours: int = 1
            forcing_script: str | None = forcing_src.get(input_forcings[i])
            forcing_start_time: datetime = refcstbdate + timedelta(hours=1)
            gfs_extended: bool = (3 in input_forcings and len(input_forcings) == 1)
            subhourly_avg: bool = False
        elif ana_flag == 1:
            look_back_hours = int(look_back / 60)
            forcing_start_time = refcstbdate
            gfs_extended = False

            if "supp16" in input_forcings and input_forcings[i] in ["supp16", "supp2"]:
                look_back_hours += 1
                subhourly_avg = True
            else:
                subhourly_avg = False

            if input_forcings[i] in ("supp1", "supp2", "supp6", "supp10", "supp11", "supp12", "supp16"):
                supp_forcing_hours = 1
            else:
                supp_forcing_hours = 0
            forcing_script = forcing_ana_src.get(input_forcings[i])

        if forcing_script is None:
            continue

        extract_scriptPath: Path = Path(extraction_scriptPath) / forcing_script
        mod_name: str = f"forcing_{extract_scriptPath.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, extract_scriptPath)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load specification for {extract_scriptPath}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        base_classes = (ForecastDownloader, FixedFileDownloader)
        downloader_class = next(
            obj
            for name, obj in vars(module).items()
            if isinstance(obj, type)
            and issubclass(obj, base_classes)
            and obj not in base_classes
        )

        start_delt: timedelta | None = None
        lag_delt: int | None = None

        if ana_flag == 1:
            start_delt = timedelta(hours=1)
            lag_delt = 1

        if supp_forcing_hours is not None:
            start_delt = (start_delt or timedelta()) + timedelta(hours=supp_forcing_hours)

        downloader = downloader_class(
            out_dir=extract_outPath,
            start_time=forcing_start_time + start_delt if start_delt else forcing_start_time,
            lookback_hours=look_back_hours + lag_delt if lag_delt else look_back_hours,
            cleanback_hours=0,
            lagback_hours=0,
            ens_number=int(ens_number) if ens_number not in ("", None) else None,
            input_forcing=input_forcings[i],
            max_download_attempts=max_download_attempts,
            download_attempt_interval=download_attempt_interval,
            check_file_availability=check_file_availability,
            subhourly_avg=subhourly_avg,
            gfs_extended=gfs_extended,
        )

        downloader.run()
