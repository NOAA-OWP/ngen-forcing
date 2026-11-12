import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class NBMAnAPuertoRicoDownloader(ForecastDownloader):
    """Downloader for NBM forecast data over Puerto Rico (AnA)."""

    @property
    def base_url(self):
        return "https://noaa-nbm-grib2-pds.s3.amazonaws.com"

    def get_download_targets(self, _):
        # NBM core has no f000; f001 is the earliest forecast hour.
        return [1]

    def build_output_dir(self, d_start, _):
        return os.path.join(self.out_dir, f"blend.{d_start.strftime('%Y%m%d')}", d_start.strftime("%H"), "core")

    def build_file_url_and_name(self, d_start, target, _):
        fhr_str = f"f{str(target).zfill(3)}"
        filename = f"blend.t{d_start.strftime('%H')}z.core.{fhr_str}.pr.grib2"
        url = os.path.join(self.base_url, f"blend.{d_start.strftime('%Y%m%d')}", d_start.strftime("%H"), "core", filename)
        return url, filename

    @property
    def recursive_cleanup(self) -> bool:
        return True

if __name__ == "__main__":
    NBMAnAPuertoRicoDownloader.cli_main()
