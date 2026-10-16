import os
import shutil

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader


class AlaskaStageIVDownloader(ForecastDownloader):
    """
    Downloader for Alaska Stage IV precipitation analysis.

    - Server organizes files in pcpanl.YYYYMMDD/ directories.
    - Filenames follow: st4_ak.YYYYMMDDHH.06h.grb2
    - Files are only available every 6 hours (00z, 06z, 12z, 18z).
    - Output directory is flattened — all files saved directly to outDir.
    """

    default_lookback = 36
    default_cleanback = 240
    default_lagback = 0

    @property
    def base_url(self):
        return "https://nomads.ncep.noaa.gov/pub/data/nccf/com/pcpanl/v4.1"

    def should_process_hour(self, d_start):
        return d_start.hour % 6 == 0

    def get_download_targets(self, d_start):
        return [0]  # Single file per valid hour

    def build_output_dir(self, d_start, _):
        # All files go to the main output directory (flat structure)
        return self.out_dir

    def build_file_url_and_name(self, d_start, _, __):
        """
        Constructs the URL and filename for Alaska Stage IV files.
        - Located in pcpanl.YYYYMMDD/
        - Named as: st4_ak.YYYYMMDDHH.06h.grb2
        """
        date_folder = f"pcpanl.{d_start.strftime('%Y%m%d')}"
        filename = f"st4_ak.{d_start.strftime('%Y%m%d%H')}.06h.grb2"
        url = os.path.join(self.base_url, date_folder, filename)
        return url, filename

    def _cleanup_old_data(self):
        """
        Deletes the entire output directory if it exists.
        """
        if os.path.isdir(self.out_dir):
            print(f"Removing old StageIV data from: {self.out_dir}")
            shutil.rmtree(self.out_dir)


if __name__ == "__main__":
    downloader = AlaskaStageIVDownloader.from_cli_args()
    downloader.run()
