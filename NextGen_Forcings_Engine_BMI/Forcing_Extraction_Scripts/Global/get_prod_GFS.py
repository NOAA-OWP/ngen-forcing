import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class GFSDownloader(ForecastDownloader):
    """Downloader for GFS operational forecast data."""
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
