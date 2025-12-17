import os

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader


class RAPAnADownloader(ForecastDownloader):
    """
    Downloader for CONUS RAP AnA (Analysis) data.

    - Only forecast hours f01 and f02 are downloaded.
    - Files: rap.t{HH}z.awp130bgrbf{01|02}.grib2
    """

    @property
    def base_url(self):
        return "https://noaa-rap-pds.s3.amazonaws.com"

    def get_download_targets(self, _):
        # Download only forecast hour 01
        return [1]

    def build_output_dir(self, d_start, _):
        # Example: output/rap.20250415/
        return os.path.join(self.out_dir, f"rap.{d_start.strftime('%Y%m%d')}")

    def build_file_url_and_name(self, d_start, forecast_hour, _):
        """
        Build both the URL and the filename for RAP forecast hour files.
        Ex: rap.t00z.awp130bgrbf01.grib2
        """
        fhr_str = str(forecast_hour).zfill(2)
        filename = f"rap.t{d_start.strftime('%H')}z.awp130bgrbf{fhr_str}.grib2"
        url = os.path.join(self.base_url, f"rap.{d_start.strftime('%Y%m%d')}", filename)
        return url, filename


if __name__ == "__main__":
    downloader = RAPAnADownloader.from_cli_args()
    downloader.run()
