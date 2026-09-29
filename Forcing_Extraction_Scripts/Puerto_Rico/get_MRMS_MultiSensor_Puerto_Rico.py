from abc import ABC

from Forcing_Extraction_Scripts.forecast_download_base import FixedFileDownloader


class MRMSMultiSensorPuertoRicoDownloader(FixedFileDownloader, ABC):
    """
    Downloader for MRMS MultiSensor QPE over Puerto Rico.

    - Pass1 and Pass2 are downloaded separately.
    - Files organized by date in respective folders.
    """

    @property
    def base_url(self):
        return "https://noaa-mrms-pds.s3.amazonaws.com/CARIB"

    def build_output_dir(self, _, __):
        return self.out_dir

    def get_file_specs(self, d_start):
        specs = []
        for pass_num in ["Pass1", "Pass2"]:
            subdir = (
                f"MultiSensor_QPE_01H_{pass_num}_00.00/{d_start.strftime('%Y%m%d')}"
            )
            filename = f"MRMS_MultiSensor_QPE_01H_{pass_num}_00.00_{d_start.strftime('%Y%m%d')}-{d_start.strftime('%H')}0000.grib2.gz"
            specs.append((subdir, filename))
        return specs


if __name__ == "__main__":
    downloader = MRMSMultiSensorPuertoRicoDownloader.from_cli_args()
    downloader.run()
