import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class NAMNestAlaskaDownloader(ForecastDownloader):
    """Downloader for NAM Alaska Nest forecast data (3-km resolution)."""
    default_lookback = 36
    default_cleanback = 240
    default_lagback = 1

    @property
    def base_url(self):
        return "https://noaa-nam-pds.s3.amazonaws.com"

    def get_download_targets(self, d_start):
        return range(1, 61) if d_start.hour in [0, 6, 12, 18] else []

    def build_output_dir(self, d_start, _):
        return os.path.join(self.out_dir, f"nam.{d_start.strftime('%Y%m%d')}")

    def build_file_url_and_name(self, d_start, fhr, _):
        fhr_str = str(fhr).zfill(2)
        filename = f"nam.t{d_start.strftime('%H')}z.alaskanest.hiresf{fhr_str}.tm00.grib2"
        url = os.path.join(self.base_url, f"nam.{d_start.strftime('%Y%m%d')}", filename)
        return url, filename

if __name__ == "__main__":
    downloader = NAMNestAlaskaDownloader.from_cli_args()
    downloader.run()
