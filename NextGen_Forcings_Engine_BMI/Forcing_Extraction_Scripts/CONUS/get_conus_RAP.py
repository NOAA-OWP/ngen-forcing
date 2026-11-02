import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class RAPDownloader(ForecastDownloader):
    """Downloader for CONUS RAP forecast data."""
    default_lookback = 30
    default_cleanback = 240
    default_lagback = 1

    @property
    def base_url(self):
        return "https://noaa-rap-pds.s3.amazonaws.com"

    def get_download_targets(self, d_start):
        return range(0, 52) if d_start.hour in [3, 9, 15, 21] else range(0, 19)

    def build_output_dir(self, d_start, _):
        return os.path.join(self.out_dir, f"rap.{d_start.strftime('%Y%m%d')}")

    def build_file_url_and_name(self, d_start, forecast_hour, _):
        fhr_str = str(forecast_hour).zfill(2)
        extensions = ["pgrb", "bgrb"]
        results = []
        for ext in extensions:
            filename = f"rap.t{d_start.strftime('%H')}z.awp130{ext}f{fhr_str}.grib2"
            url = os.path.join(self.base_url, f"rap.{d_start.strftime('%Y%m%d')}", filename)
            results.append((url, filename))
        return results

if __name__ == "__main__":
    downloader = RAPDownloader.from_cli_args()
    downloader.run()
