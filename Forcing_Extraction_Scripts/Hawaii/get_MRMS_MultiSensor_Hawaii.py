from abc import ABC

from Forcing_Extraction_Scripts.forecast_download_base import FixedFileDownloader


class MRMSMultiSensorHawaiiDownloader(FixedFileDownloader, ABC):
    """
    Downloader for MRMS MultiSensor QPE (Pass1 and Pass2) for Hawaii.

    - Downloads two files per hour: one from Pass1 and one from Pass2.
    - Files are stored under: MultiSensor_QPE_01H_Pass1/YYYYMMDD/...
    """

    @property
    def base_url(self):
        return "https://noaa-mrms-pds.s3.amazonaws.com/HAWAII"

    def build_output_dir(self, _, __):
        # Output is stored under separate Pass1/Pass2 folders by date
        return self.out_dir

    def get_file_specs(self, d_start):
        specs = []
        for pass_num in ["Pass1", "Pass2"]:
            subdir = f"MultiSensor_QPE_01H_{pass_num}_00.00/{d_start.strftime('%Y%m%d')}"
            filename = f"MRMS_MultiSensor_QPE_01H_{pass_num}_00.00_{d_start.strftime('%Y%m%d')}-{d_start.strftime('%H')}0000.grib2.gz"
            specs.append((subdir, filename))
        return specs


if __name__ == "__main__":
    downloader = MRMSMultiSensorHawaiiDownloader.from_cli_args()
    downloader.run()
