import os
from Forcing_Extraction_Scripts.forecast_download_base import (
    NAM_HIRESW_RETIRED_AFTER,
    NAM_HIRESW_RETIREMENT_NOTE,
    ForecastDownloader,
)

class ARWHawaiiDownloader(ForecastDownloader):
    """Downloader for WRF-ARW 2.5 km forecast data over Hawaii."""
    retired_after = NAM_HIRESW_RETIRED_AFTER
    retirement_note = NAM_HIRESW_RETIREMENT_NOTE

    @property
    def base_url(self):
        return "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hiresw/prod"

    def should_process_hour(self, d_current):
        return d_current.hour in [0, 12]

    def get_download_targets(self, d_current):
        return range(1, 49)

    def build_output_dir(self, d_current, _=None):
        return os.path.join(self.out_dir, f"hiresw.{d_current.strftime('%Y%m%d')}")

    def build_file_url_and_name(self, d_current, target, _=None):
        fhr = str(target).zfill(2)
        filename = f"hiresw.t{d_current.strftime('%H')}z.arw_2p5km.f{fhr}.hi.grib2"
        url = os.path.join(self.base_url, f"hiresw.{d_current.strftime('%Y%m%d')}", filename)
        return url, filename

    @property
    def recursive_cleanup(self) -> bool:
        return True

if __name__ == "__main__":
    ARWHawaiiDownloader.cli_main()
