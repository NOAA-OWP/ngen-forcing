"""North American Mesoscale (NAM) nest downloaders. NAM was retired on 2026-11-03."""

from __future__ import annotations

import os
import typing
from dataclasses import dataclass
from datetime import datetime, timedelta

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader, Region

#: NAM ended with the RRFS implementation on 2026-11-03; the last cycle in the S3 archive is 06Z that
#: day. Earlier cycles are still fetched, for retrospective runs.
RETIRED_AFTER = datetime(2026, 11, 3, 6)
RETIREMENT_NOTE = "NAM was retired on 2026-11-03 and replaced by RRFS (NWS Service Change Notices 26-47 and 26-48)"


@dataclass(frozen=True)
class NAMNest:
    """One NAM nest."""

    #: Nest name in file names.
    name: str
    #: Standalone defaults (hours).
    lookback: int = ForecastDownloader.default_lookback
    lagback: int = ForecastDownloader.default_lagback


CONUS = NAMNest("conusnest")
ALASKA = NAMNest("alaskanest", lookback=36, lagback=1)
HAWAII = NAMNest("hawaiinest")
PUERTO_RICO = NAMNest("priconest")


class _NAMNestBase(ForecastDownloader):
    """File naming shared by the NAM nest downloaders."""

    REGIONS = {Region.CONUS: CONUS, Region.ALASKA: ALASKA, Region.HAWAII: HAWAII, Region.PUERTO_RICO: PUERTO_RICO}

    retirement_note = RETIREMENT_NOTE

    @classmethod
    def retirement_date(cls, region):
        return RETIRED_AFTER

    @property
    def base_url(self) -> str:
        return "https://noaa-nam-pds.s3.amazonaws.com"

    @classmethod
    def window_defaults(cls, region: Region) -> tuple[int, int, int]:
        nest: NAMNest = cls.REGIONS[region]
        return nest.lookback, cls.default_cleanback, nest.lagback

    def _url_and_name(self, cycle: datetime, forecast_hour: int) -> tuple[str, str]:
        filename = f"nam.t{cycle.strftime('%H')}z.{self.domain.name}.hiresf{forecast_hour:02d}.tm00.grib2"
        return f"{self.base_url}/nam.{cycle.strftime('%Y%m%d')}/{filename}", filename


class NAMNestDownloader(_NAMNestBase):
    """Downloader for NAM nest forecasts."""

    def should_process_hour(self, d_start):
        return d_start.hour in [0, 6, 12, 18]

    def get_download_targets(self, _):
        return range(1, 61)

    def build_output_dir(self, d_start, _):
        return os.path.join(self.out_dir, f"nam.{d_start.strftime('%Y%m%d')}")

    def build_file_url_and_name(self, d_start, forecast_hour, _):
        return self._url_and_name(d_start, forecast_hour)


class NAMNestAnADownloader(_NAMNestBase):
    """NAM nest AnA: forecast hours 1-6 of the 00/06/12/18Z cycle covering each hour."""

    @staticmethod
    def nam_cycle(d_start: datetime) -> datetime:
        """Return the 00/06/12/18Z NAM cycle that covers d_start."""
        return d_start - timedelta(hours=d_start.hour % 6)

    def get_download_targets(self, d_start: datetime) -> list[typing.Any]:
        return list(range(1, 7))

    def build_output_dir(self, d_start: datetime, ens_number: str) -> str:
        return os.path.join(str(self.out_dir), f"nam.{self.nam_cycle(d_start).strftime('%Y%m%d')}")

    def build_file_url_and_name(self, d_start: datetime, target: typing.Any, ens_number: str) -> tuple[str, str]:
        return self._url_and_name(self.nam_cycle(d_start), target)
