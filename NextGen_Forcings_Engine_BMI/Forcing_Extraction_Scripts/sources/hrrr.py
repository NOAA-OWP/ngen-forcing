"""High-Resolution Rapid Refresh (HRRR) downloaders."""

from __future__ import annotations

import os
from dataclasses import dataclass

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader, Region


@dataclass(frozen=True)
class HRRRDomain:
    """How one HRRR domain is published."""

    #: Directory under ``hrrr.YYYYMMDD/`` on the server and locally.
    subdir: str
    #: Text between the forecast hour and ``.grib2`` in file names.
    suffix: str
    #: Hours between cycles.
    cycle_step: int
    #: Forecast hours an AnA run reads from each cycle.
    ana_hours: tuple[int, ...]


CONUS = HRRRDomain("conus", "", 1, (1,))
ALASKA = HRRRDomain("alaska", ".ak", 3, (1, 2, 3, 4))


class _HRRRBase(ForecastDownloader):
    """File naming shared by the HRRR downloaders."""

    default_lookback = 30
    default_cleanback = 240
    default_lagback = 1

    #: File-name product: wrfsfc (hourly surface) or wrfsubh (15-minute).
    product = "wrfsfc"

    @property
    def base_url(self):
        return "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

    def should_process_hour(self, d_start):
        return d_start.hour % self.domain.cycle_step == 0

    def local_subdir(self) -> str:
        return self.domain.subdir

    def build_output_dir(self, d_start, _):
        return os.path.join(self.out_dir, f"hrrr.{d_start.strftime('%Y%m%d')}", self.local_subdir())

    def build_file_url_and_name(self, d_start, forecast_hour, _):
        filename = f"hrrr.t{d_start.strftime('%H')}z.{self.product}f{forecast_hour:02d}{self.domain.suffix}.grib2"
        url = os.path.join(self.base_url, f"hrrr.{d_start.strftime('%Y%m%d')}", self.domain.subdir, filename)
        return url, filename


class HRRRDownloader(_HRRRBase):
    """Downloader for HRRR surface forecasts."""

    REGIONS = {Region.CONUS: CONUS, Region.ALASKA: ALASKA}

    def get_download_targets(self, d_start):
        # 00, 06, 12 and 18Z cycles run to 48 hours; the others to 18 hours.
        return range(0, 49) if d_start.hour % 6 == 0 else range(0, 19)


class HRRRAnADownloader(_HRRRBase):
    """Downloader for the HRRR surface forecast hours an Analysis and Assimilation (AnA) run reads."""

    REGIONS = {Region.CONUS: CONUS, Region.ALASKA: ALASKA}

    def get_download_targets(self, _):
        return list(self.domain.ana_hours)


class _HRRRSubhourlyBase(_HRRRBase):
    """HRRR 15-minute output, stored locally under ``subhourly/``."""

    REGIONS = {Region.CONUS: CONUS}
    product = "wrfsubh"

    def local_subdir(self) -> str:
        return "subhourly"


class HRRRSubhourlyDownloader(_HRRRSubhourlyBase):
    """Downloader for HRRR sub-hourly (15-minute) forecasts."""

    def get_download_targets(self, _):
        return range(0, 19)


class HRRRSubhourlyAnADownloader(_HRRRSubhourlyBase):
    """Downloader for the HRRR sub-hourly (15-minute) forecast hour an AnA run reads."""

    def get_download_targets(self, _):
        return [1]
