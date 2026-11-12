"""Global Forecast System (GFS) downloader."""

from __future__ import annotations

import os

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader


class GFSDownloader(ForecastDownloader):
    """Downloader for GFS operational forecast data."""

    #: A global model: no regional variants.
    REGIONS = {None: None}

    default_lookback = 8
    default_cleanback = 384
    default_lagback = 4

    @property
    def base_url(self):
        return "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

    def should_process_hour(self, d_start):
        return d_start.hour in [0, 6, 12, 18]

    def get_download_targets(self, _):
        hourly = range(1, 121)
        every_3h = range(123, 385, 3)
        return list(hourly) + list(every_3h)

    def build_output_dir(self, d_start, _):
        return os.path.join(self.out_dir, f"gfs.{d_start.strftime('%Y%m%d')}", d_start.strftime("%H"), "atmos")

    def build_file_url_and_name(self, d_start, forecast_hour, _):
        filename = f"gfs.t{d_start.strftime('%H')}z.sfluxgrbf{forecast_hour:03d}.grib2"
        url = os.path.join(self.base_url, f"gfs.{d_start.strftime('%Y%m%d')}", d_start.strftime("%H"), "atmos", filename)
        return url, filename

    @property
    def recursive_cleanup(self) -> bool:
        return True
