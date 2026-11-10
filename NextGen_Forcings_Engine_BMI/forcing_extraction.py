from __future__ import annotations

import argparse
import importlib.util
import inspect
import logging
import sys
import typing
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from config_models import ForcingExtractionConfig
from Forcing_Extraction_Scripts.forecast_download_base import (
    DownloadError,
    ForecastDownloader,
)

LOG = logging.getLogger(__name__)

SCRIPT_DIR: Path = Path(__file__).resolve().parent / "Forcing_Extraction_Scripts"

#: Forecast-mode (AnAFlag 0) downloaders, keyed by InputForcings value or "supp<SuppPcp>".
FORCING_SRC: dict[int | str, str] = {
    3: "Global/get_prod_GFS.py",
    5: "CONUS/get_conus_HRRR.py",
    6: "CONUS/get_conus_RAP.py",
    7: "Global/get_CFSv2.py",
    8: "Hawaii/get_AWR_Hawaii.py",
    13: "Hawaii/get_prod_NAM_Nest_Hawaii.py",
    14: "Puerto_Rico/get_prod_NAM_Nest_Puerto_Rico.py",
    16: "Hawaii/get_prod_NAM_Nest_Hawaii.py",
    17: "Puerto_Rico/get_prod_NAM_Nest_Puerto_Rico.py",
    18: "Puerto_Rico/get_ARW_Puerto_Rico.py",
    19: "Alaska/get_Alaska_HRRR.py",
    24: "CONUS/get_prod_NBM_Conus.py",
    25: "CONUS/get_conus_NDFD.py",
    26: "CONUS/get_CONUS_HRRR_subhourly.py",
    "supp8": "CONUS/get_prod_NBM_Conus.py",
    "supp9": "Alaska/get_prod_NBM_Alaska.py",
    "supp11": "Alaska/get_Alaska_StageIV.py",
    "supp12": "CONUS/get_conus_StageIV.py",
    "supp15": "Puerto_Rico/get_prod_NBM_Puerto_Rico.py",
}

#: Analysis and Assimilation (AnAFlag 1) downloaders.
FORCING_ANA_SRC: dict[int | str, str] = {
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

#: Hourly observation products whose AnA window is shifted one hour later.
HOURLY_OBS_SUPP = ("supp1", "supp2", "supp6", "supp10", "supp11", "supp12", "supp16")


@dataclass
class ExtractionJob:
    """One downloader invocation."""

    forcing: int | str
    script: str
    out_dir: str
    horizon_minutes: int | None = None


def load_downloader_class(script: Path) -> type[ForecastDownloader]:
    """Import a downloader script and return the concrete downloader class it defines."""
    spec = importlib.util.spec_from_file_location(f"forcing_{script.stem}", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load specification for {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidates = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type)
        and issubclass(obj, ForecastDownloader)
        and obj.__module__ == module.__name__
        and not inspect.isabstract(obj)
    ]
    if len(candidates) != 1:
        raise ImportError(f"Expected one downloader class in {script}, found {[c.__name__ for c in candidates]}")
    return candidates[0]


def _supp_dirs(cfg: ForcingExtractionConfig) -> list[str]:
    raw = cfg.supp_precip_dirs
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    return list(raw or [])


def build_jobs(cfg: ForcingExtractionConfig) -> list[ExtractionJob]:
    """Translate a configuration into the list of downloader invocations it needs."""
    src = FORCING_SRC if cfg.ana_flag == 0 else FORCING_ANA_SRC
    horizons = list(cfg.fcst_input_horizons)
    jobs: list[ExtractionJob] = []

    def add(forcing: int | str, out_dir: str, horizon: int | None) -> None:
        script = src.get(forcing)
        if script is None:
            return
        jobs.append(ExtractionJob(forcing, script, out_dir, horizon))

    for i, forcing in enumerate(cfg.input_forcings):
        add(forcing, cfg.input_force_dirs[i], horizons[i] if i < len(horizons) else None)

    supp = [f"supp{val}" for val in cfg.supp_precip_forcings]
    if not supp:
        return jobs
    supp_dirs = _supp_dirs(cfg)
    supp_horizon = horizons[0] if horizons else None

    for j, supp_id in enumerate(supp):
        out_dir = supp_dirs[j]
        if supp_id == "supp16":
            # Sub-hourly MRMS cycling also needs hourly MultiSensor MRMS for bias correction.
            add("supp16", f"{out_dir}/MRMS_SubHourly", supp_horizon)
            add("supp2", f"{out_dir}/MRMS_Hourly", supp_horizon)
        else:
            add(supp_id, out_dir, supp_horizon)
    return jobs


def make_downloader(cfg: ForcingExtractionConfig, job: ExtractionJob) -> ForecastDownloader:
    """Instantiate the downloader for one job, with the processing window the engine run needs."""
    refcstbdate: datetime = cfg.b_date_proc
    supp_ids = {f"supp{val}" for val in cfg.supp_precip_forcings}
    if cfg.ana_flag == 0:
        look_back_hours = 1
        start_time = refcstbdate + timedelta(hours=1)
        subhourly_avg = False
    else:
        look_back_hours = int(cfg.look_back / 60) + 1
        start_delt = timedelta(hours=1)
        subhourly_avg = "supp16" in supp_ids and job.forcing in ("supp16", "supp2")
        if subhourly_avg:
            look_back_hours += 1
        if job.forcing in HOURLY_OBS_SUPP:
            start_delt += timedelta(hours=1)
        start_time = refcstbdate + start_delt

    ens_number = cfg.cfsv2EnsMember
    downloader_class = load_downloader_class(SCRIPT_DIR / job.script)
    return downloader_class(
        out_dir=job.out_dir,
        start_time=start_time,
        lookback_hours=look_back_hours,
        cleanback_hours=0,
        lagback_hours=0,
        ens_number=int(ens_number) if ens_number not in ("", None) else None,
        input_forcing=job.forcing,
        max_download_attempts=cfg.max_download_attempts,
        download_attempt_interval=cfg.download_attempt_interval,
        check_file_availability=cfg.check_file_availability,
        subhourly_avg=subhourly_avg,
        gfs_extended=(3 in cfg.input_forcings and len(cfg.input_forcings) == 1),
    )


def retrieve_forcing(cfg: ForcingExtractionConfig) -> None:
    """Download forecast forcing data based on requested sources in configuration."""
    missing: list[str] = []
    for job in build_jobs(cfg):
        downloader = make_downloader(cfg, job)
        if not downloader.run():
            missing.append(f"{job.forcing} ({job.script})")

    if missing:
        raise DownloadError("Files are missing upstream for: " + "; ".join(missing))


def main() -> None:
    """Download forecast forcing data using CLI YAML config input."""
    parser = argparse.ArgumentParser(description="Download forecast forcing data")
    parser.add_argument("cfg", type=Path, help="Path to YAML config file")
    args = parser.parse_args()

    cfg: ForcingExtractionConfig = ForcingExtractionConfig.from_yaml_file(args.cfg)
    try:
        retrieve_forcing(cfg)
    except DownloadError as exc:
        LOG.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
