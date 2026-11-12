"""Rapid Refresh (RAP) downloaders."""

from __future__ import annotations

import os
from dataclasses import dataclass

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader, Region


@dataclass(frozen=True)
class RAPGrid:
    """One RAP output grid."""

    #: AWIPS grid in file names.
    name: str
    #: File-name parts after the grid; one file per part is fetched for each forecast hour.
    parts: tuple[str, ...] = ("",)


class _RAPBase(ForecastDownloader):
    """File naming shared by the RAP downloaders."""

    REGIONS = {
        # The 13 km CONUS grid is split in two files, both required per forecast hour (input_forcing 6).
        Region.CONUS: RAPGrid("awp130", ("pgrb", "bgrb")),
        Region.ALASKA: RAPGrid("awp242"),  # 11 km
        Region.HAWAII: RAPGrid("awp243"),  # 0.4 degree eastern North Pacific
        Region.PUERTO_RICO: RAPGrid("awp200"),  # 16 km
    }

    @property
    def base_url(self):
        return "https://noaa-rap-pds.s3.amazonaws.com"

    def build_output_dir(self, d_start, _):
        return os.path.join(self.out_dir, f"rap.{d_start.strftime('%Y%m%d')}")

    def build_file_url_and_name(self, d_start, forecast_hour, _):
        results = []
        for part in self.domain.parts:
            filename = f"rap.t{d_start.strftime('%H')}z.{self.domain.name}{part}f{forecast_hour:02d}.grib2"
            url = os.path.join(self.base_url, f"rap.{d_start.strftime('%Y%m%d')}", filename)
            results.append((url, filename))
        return results


class RAPDownloader(_RAPBase):
    """Downloader for RAP forecasts."""

    default_lookback = 30
    default_cleanback = 240
    default_lagback = 1

    def get_download_targets(self, d_start):
        return range(0, 52) if d_start.hour in [3, 9, 15, 21] else range(0, 19)


class RAPAnADownloader(_RAPBase):
    """Downloader for the RAP forecast hour an Analysis and Assimilation (AnA) run reads."""

    def get_download_targets(self, _):
        return [1]
