"""National Blend of Models (NBM) downloaders."""

from __future__ import annotations

import os

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader, Region


class _NBMBase(ForecastDownloader):
    """File naming shared by the NBM core downloaders. ``REGIONS`` maps to the domain tag in file names."""

    REGIONS = {Region.CONUS: "co", Region.ALASKA: "ak", Region.HAWAII: "hi", Region.PUERTO_RICO: "pr",
               Region.GUAM: "gu"}

    @property
    def base_url(self):
        return "https://noaa-nbm-grib2-pds.s3.amazonaws.com"

    def build_output_dir(self, d_start, _):
        return os.path.join(self.out_dir, f"blend.{d_start.strftime('%Y%m%d')}", d_start.strftime('%H'), "core")

    def build_file_url_and_name(self, d_start, forecast_hour, _):
        filename = f"blend.t{d_start.strftime('%H')}z.core.f{forecast_hour:03d}.{self.domain}.grib2"
        url = os.path.join(self.base_url, f"blend.{d_start.strftime('%Y%m%d')}", d_start.strftime('%H'), "core", filename)
        return url, filename

    @property
    def recursive_cleanup(self) -> bool:
        return True


class NBMDownloader(_NBMBase):
    """Downloader for NBM core forecasts (GRIB2)."""

    def should_process_hour(self, d_start):
        return d_start.hour in [0, 6, 12, 18]

    def get_download_targets(self, _):
        hourly = range(1, 37)
        every_3h = range(39, 193, 3)
        every_6h = range(198, 265, 6)
        return list(hourly) + list(every_3h) + list(every_6h)


class NBMAnADownloader(_NBMBase):
    """Downloader for the NBM core forecast hour an Analysis and Assimilation (AnA) run reads."""

    def get_download_targets(self, _):
        # NBM core has no f000; f001 is the earliest forecast hour.
        return [1]
