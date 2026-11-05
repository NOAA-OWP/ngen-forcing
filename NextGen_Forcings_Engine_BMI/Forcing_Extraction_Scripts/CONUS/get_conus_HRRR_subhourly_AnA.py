import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class HRRRSubhourlyAnADownloader(ForecastDownloader):
    """Downloader for CONUS HRRR sub-hourly AnA data."""
    default_lookback = 30
    default_cleanback = 240
    default_lagback = 1

    @property
    def base_url(self):
        return "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

    def should_process_hour(self, d_current):
        return True

    def get_download_targets(self, d_current):
        return [1]

    def build_output_dir(self, d_current, _):
        return os.path.join(self.out_dir, f"hrrr.{d_current.strftime('%Y%m%d')}", "subhourly")

    def build_file_url_and_name(self, d_current, target, _):
        fhr_str = f"{target:02d}"
        hour = d_current.strftime('%H')
        date = d_current.strftime('%Y%m%d')
        filename = f"hrrr.t{hour}z.wrfsubhf{fhr_str}.grib2"
        url = os.path.join(self.base_url, f"hrrr.{date}", "conus", filename)
        return url, filename

if __name__ == "__main__":
    downloader = HRRRSubhourlyAnADownloader.from_cli_args()
    downloader.run()
