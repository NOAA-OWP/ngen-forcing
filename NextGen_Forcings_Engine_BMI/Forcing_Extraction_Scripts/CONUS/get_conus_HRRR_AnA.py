import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class HRRRAnAConusDownloader(ForecastDownloader):
    """Downloader for CONUS HRRR Analysis (AnA) surface data."""
    default_lookback = 30
    default_cleanback = 240
    default_lagback = 1

    @property
    def base_url(self):
        return "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

    def get_download_targets(self, _):
        return [1]

    def build_output_dir(self, d_start, _):
        return os.path.join(self.out_dir, f"hrrr.{d_start.strftime('%Y%m%d')}", "conus")

    def build_file_url_and_name(self, d_start, forecast_hour, _):
        fhr_str = str(forecast_hour).zfill(2)
        filename = f"hrrr.t{d_start.strftime('%H')}z.wrfsfcf{fhr_str}.grib2"
        url = os.path.join(self.base_url, f"hrrr.{d_start.strftime('%Y%m%d')}", "conus", filename)
        return url, filename

if __name__ == "__main__":
    downloader = HRRRAnAConusDownloader.from_cli_args()
    downloader.run()
