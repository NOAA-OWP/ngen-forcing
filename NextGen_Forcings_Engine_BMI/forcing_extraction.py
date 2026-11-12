from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from config_models import ForcingExtractionConfig
from Forcing_Extraction_Scripts.forecast_download_base import (
    DownloadError,
    ForecastDownloader,
    Region,
    configure_logging,
)
from Forcing_Extraction_Scripts.sources import (
    CFSv2Downloader,
    GFSDownloader,
    HiResWARWDownloader,
    HRRRAnADownloader,
    HRRRDownloader,
    HRRRSubhourlyAnADownloader,
    HRRRSubhourlyDownloader,
    MRMSMultiSensorDownloader,
    MRMSPrecipRateDownloader,
    MRMSRadarOnlyDownloader,
    NAMNestAnADownloader,
    NAMNestDownloader,
    NBMAnADownloader,
    NBMDownloader,
    NDFDDownloader,
    RAPAnADownloader,
    RAPDownloader,
    Source,
    StageIVDownloader,
)

LOG = logging.getLogger(__name__)

#: Forecast-mode (AnAFlag 0) sources, keyed by InputForcings value or "supp<SuppPcp>".
#: Each value is the downloader class and the region it is run for.
FORCING_SRC: dict[int | str, Source] = {
    3: (GFSDownloader, Region.GLOBAL),
    5: (HRRRDownloader, Region.CONUS),
    6: (RAPDownloader, Region.CONUS),
    7: (CFSv2Downloader, Region.GLOBAL),
    8: (HiResWARWDownloader, Region.HAWAII),
    13: (NAMNestDownloader, Region.HAWAII),
    14: (NAMNestDownloader, Region.PUERTO_RICO),
    15: (NAMNestDownloader, Region.ALASKA),
    16: (NAMNestDownloader, Region.HAWAII),
    17: (NAMNestDownloader, Region.PUERTO_RICO),
    18: (HiResWARWDownloader, Region.PUERTO_RICO),
    19: (HRRRDownloader, Region.ALASKA),
    24: (NBMDownloader, Region.CONUS),
    25: (NDFDDownloader, Region.CONUS),
    26: (HRRRSubhourlyDownloader, Region.CONUS),
    "supp8": (NBMDownloader, Region.CONUS),
    "supp9": (NBMDownloader, Region.ALASKA),
    "supp11": (StageIVDownloader, Region.ALASKA),
    "supp12": (StageIVDownloader, Region.CONUS),
    "supp15": (NBMDownloader, Region.PUERTO_RICO),
}

#: Analysis and Assimilation (AnAFlag 1) sources.
FORCING_ANA_SRC: dict[int | str, Source] = {
    5: (HRRRAnADownloader, Region.CONUS),
    6: (RAPAnADownloader, Region.CONUS),
    13: (NAMNestAnADownloader, Region.HAWAII),
    14: (NAMNestAnADownloader, Region.PUERTO_RICO),
    19: (HRRRDownloader, Region.ALASKA),
    20: (HRRRAnADownloader, Region.ALASKA),
    26: (HRRRSubhourlyAnADownloader, Region.CONUS),
    "supp1": (MRMSRadarOnlyDownloader, Region.CONUS),
    "supp2": (MRMSMultiSensorDownloader, Region.CONUS),
    "supp6": (MRMSMultiSensorDownloader, Region.HAWAII),
    "supp10": (MRMSMultiSensorDownloader, Region.ALASKA),
    "supp11": (StageIVDownloader, Region.ALASKA),
    "supp12": (StageIVDownloader, Region.CONUS),
    "supp15": (NBMAnADownloader, Region.PUERTO_RICO),
    "supp16": (MRMSPrecipRateDownloader, Region.CONUS),
}

#: MRMS MultiSensor source for the second (MRMS) directory of the Stage IV/MRMS SuppPcp options.
STAGE4_MRMS_SRC: dict[str, Source] = {
    "supp11": (MRMSMultiSensorDownloader, Region.ALASKA),
    "supp12": (MRMSMultiSensorDownloader, Region.CONUS),
}

#: Hourly observation products whose AnA window is shifted one hour later.
HOURLY_OBS_SUPP = ("supp1", "supp2", "supp6", "supp10", "supp11", "supp12", "supp16")


@dataclass
class ExtractionJob:
    """One downloader invocation."""

    forcing: int | str
    downloader: type[ForecastDownloader]
    region: Region
    out_dir: str
    horizon_minutes: int | None = None
    local_layout: str | None = None

    @property
    def label(self) -> str:
        """The source and region, e.g. ``HRRR[Alaska]``."""
        return self.downloader.label(self.region)


def _supp_dirs(cfg: ForcingExtractionConfig) -> list[str]:
    raw = cfg.supp_precip_dirs
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    return list(raw or [])


def build_jobs(cfg: ForcingExtractionConfig) -> list[ExtractionJob]:
    """Translate a configuration into the list of downloader invocations it needs."""
    src = FORCING_SRC if cfg.ana_flag == 0 else FORCING_ANA_SRC
    mode = "forecast" if cfg.ana_flag == 0 else "AnA"
    horizons = list(cfg.fcst_input_horizons)
    jobs: list[ExtractionJob] = []

    def add(forcing: int | str, out_dir: str, horizon: int | None, source: Source | None = None,
            local_layout: str | None = None) -> None:
        source = source or src.get(forcing)
        if source is None:
            LOG.warning("No downloader is registered for forcing %r in %s mode; skipping it.", forcing, mode)
            return
        downloader, region = source
        jobs.append(ExtractionJob(forcing, downloader, region, out_dir, horizon, local_layout))

    if len(cfg.input_force_dirs) < len(cfg.input_forcings):
        raise ValueError("InputForcingDirectories must have one entry per InputForcings value.")
    for i, forcing in enumerate(cfg.input_forcings):
        add(forcing, cfg.input_force_dirs[i], horizons[i] if i < len(horizons) else None)

    supp = [f"supp{val}" for val in cfg.supp_precip_forcings]
    if not supp:
        return jobs
    supp_dirs = _supp_dirs(cfg)
    supp_horizon = horizons[0] if horizons else None

    stage4 = [s for s in supp if s in STAGE4_MRMS_SRC]
    if stage4:
        # The engine reads these as one comma-joined "<Stage IV dir>,<MRMS dir>" input (config.py),
        # so they are only supported as the sole SuppPcp option.
        if len(supp) != 1 or not supp_dirs:
            raise ValueError("SuppPcp 11 or 12 must be the only SuppPcp option and needs a Stage IV directory.")
        label = stage4[0]
        add(label, supp_dirs[0], supp_horizon)
        if len(supp_dirs) > 1:
            add(label, supp_dirs[1], supp_horizon, source=STAGE4_MRMS_SRC[label],
                local_layout=MRMSMultiSensorDownloader.LAYOUT_STAGE4_BLEND)
        else:
            LOG.warning("%s has no second (MRMS) directory in SuppPcpDirectories; MRMS will not be downloaded.", label)
        return jobs

    if len(supp_dirs) < len(supp):
        raise ValueError("SuppPcpDirectories must have one entry per SuppPcp value.")
    for supp_id, out_dir in zip(supp, supp_dirs):
        if supp_id == "supp16":
            # Sub-hourly MRMS cycling also needs hourly MultiSensor MRMS for bias correction.
            add("supp16", f"{out_dir}/MRMS_SubHourly", supp_horizon)
            add("supp2", f"{out_dir}/MRMS_Hourly", supp_horizon,
                local_layout=MRMSMultiSensorDownloader.LAYOUT_DATED)
        else:
            add(supp_id, out_dir, supp_horizon)
    return jobs


def validate(cfg: ForcingExtractionConfig) -> None:
    """Reject configurations the extraction cannot interpret."""
    if cfg.ana_flag not in (0, 1):
        raise ValueError(f"AnAFlag must be 0 or 1, got {cfg.ana_flag!r}.")
    if cfg.ana_flag == 1 and cfg.look_back < 60:
        raise ValueError(f"LookBack must be at least 60 minutes for an AnA run, got {cfg.look_back!r}.")


def make_downloader(cfg: ForcingExtractionConfig, job: ExtractionJob) -> ForecastDownloader:
    """Instantiate the downloader for one job, with the processing window the engine run needs."""
    refcstbdate: datetime = cfg.b_date_proc
    supp_ids = {f"supp{val}" for val in cfg.supp_precip_forcings}
    if cfg.ana_flag == 0:
        look_back_hours = 1
        start_time = refcstbdate + timedelta(hours=1)
        subhourly_avg = False
        max_forecast_hour = math.ceil(job.horizon_minutes / 60) if job.horizon_minutes else None
    else:
        look_back_hours = int(cfg.look_back / 60) + 1
        start_delt = timedelta(hours=1)
        subhourly_avg = "supp16" in supp_ids and job.forcing in ("supp16", "supp2")
        if subhourly_avg:
            look_back_hours += 1
        if job.forcing in HOURLY_OBS_SUPP:
            start_delt += timedelta(hours=1)
        start_time = refcstbdate + start_delt
        max_forecast_hour = None

    ens_number = cfg.cfsv2EnsMember
    options = {"local_layout": job.local_layout} if job.local_layout is not None else {}
    return job.downloader(
        region=job.region,
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
        max_forecast_hour=max_forecast_hour,
        **options,
    )


def retrieve_forcing(cfg: ForcingExtractionConfig) -> None:
    """Download forecast forcing data based on requested sources in configuration."""
    validate(cfg)
    missing: list[str] = []
    for job in build_jobs(cfg):
        downloader = make_downloader(cfg, job)
        if not downloader.run():
            missing.append(f"{job.forcing} ({job.label})")

    if missing:
        raise DownloadError("Files are missing upstream for: " + "; ".join(missing))


def main() -> None:
    """Download forecast forcing data using CLI YAML config input."""
    configure_logging()
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
