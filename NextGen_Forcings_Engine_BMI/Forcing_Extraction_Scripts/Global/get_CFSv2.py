import os
from datetime import timedelta
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class CFSv2Downloader(ForecastDownloader):
    """Downloader for CFSv2 forecast data (6-hour interval outputs for 30 days)."""
    default_lookback = 24
    default_cleanback = 720
    default_lagback = 6

    @property
    def base_url(self):
        return "https://noaa-cfs-pds.s3.amazonaws.com"

    def get_download_targets(self, d_start):
        return range(0, 721, 6) if d_start.hour in [0, 6, 12, 18] else []

    def should_process_hour(self, d_start):
        return d_start.hour in [0, 6, 12, 18]
