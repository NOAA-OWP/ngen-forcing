import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class NBMAnAPuertoRicoDownloader(ForecastDownloader):
    """Downloader for NBM forecast data over Puerto Rico (AnA)."""

    @property
    def base_url(self):
        return "https://noaa-nbm-grib2-pds.s3.amazonaws.com"

    def get_download_targets(self, _):
        return range(0, 2)

    def build_output_dir(self, d_start, _):
        return os.path.join(self.out_dir, f"blend.{d_start.strftime('%Y%m%d')}", d_start.strftime("%H"), "core")
