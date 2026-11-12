"""High-Resolution Window (HiResW) downloader. HiResW (except Guam) was retired on 2026-11-03."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader, Region

#: HiResW (except Guam) ended with the RRFS implementation on 2026-11-03; 06Z was the last cycle.
#: Earlier cycles are still fetched, for retrospective runs.
RETIRED_AFTER = datetime(2026, 11, 3, 6)
RETIREMENT_NOTE = "HiResW (except Guam) was retired on 2026-11-03 and replaced by RRFS (NWS Service Change Notices 26-47 and 26-48)"


@dataclass(frozen=True)
class HiResWDomain:
    """One HiResW domain."""

    #: Domain tag in file names.
    tag: str
    #: Cycle hours (UTC) the domain runs at.
    cycles: tuple[int, ...]


class HiResWARWDownloader(ForecastDownloader):
    """Downloader for HiResW WRF-ARW 2.5 km forecasts."""

    REGIONS = {Region.HAWAII: HiResWDomain("hi", (0, 12)), Region.PUERTO_RICO: HiResWDomain("pr", (6, 18))}

    retired_after = RETIRED_AFTER
    retirement_note = RETIREMENT_NOTE

    @property
    def base_url(self):
        return "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hiresw/prod"

    def should_process_hour(self, d_current):
        return d_current.hour in self.domain.cycles

    def get_download_targets(self, _):
        return range(1, 49)

    def build_output_dir(self, d_current, _=None):
        return os.path.join(self.out_dir, f"hiresw.{d_current.strftime('%Y%m%d')}")

    def build_file_url_and_name(self, d_current, forecast_hour, _=None):
        filename = f"hiresw.t{d_current.strftime('%H')}z.arw_2p5km.f{forecast_hour:02d}.{self.domain.tag}.grib2"
        url = os.path.join(self.base_url, f"hiresw.{d_current.strftime('%Y%m%d')}", filename)
        return url, filename

    @property
    def recursive_cleanup(self) -> bool:
        return True
