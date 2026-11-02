import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class NBMPuertoRicoDownloader(ForecastDownloader):
    """Downloader for NBM forecast data over Puerto Rico using predictable filenames."""

    @property
    def base_url(self):
        return "https://noaa-nbm-grib2-pds.s3.amazonaws.com"

    def should_process_hour(self, d_start):
        return d_start.hour in [0, 6, 12, 18]

    def get_download_targets(self, d_start):
        hourly = range(1, 37)
        every_3h = range(36, 193, 3)
        every_6h = range(198, 265, 6)
        return list(hourly) + list(every_3h) + list(every_6h) if d_start.hour in [0, 6, 12, 18] else []
