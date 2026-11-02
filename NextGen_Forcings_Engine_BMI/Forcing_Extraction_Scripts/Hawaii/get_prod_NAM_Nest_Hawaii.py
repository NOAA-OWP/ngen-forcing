import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class NAMNestHawaiiDownloader(ForecastDownloader):
    """Downloader for NAM Nest Hawaii 3-km forecast data."""

    @property
    def base_url(self):
        return "https://noaa-nam-pds.s3.amazonaws.com"

    def should_process_hour(self, d_start):
        return d_start.hour in [0, 6, 12, 18]

    def get_download_targets(self, d_start):
        return range(1, 61) if d_start.hour in [0, 6, 12, 18] else []
