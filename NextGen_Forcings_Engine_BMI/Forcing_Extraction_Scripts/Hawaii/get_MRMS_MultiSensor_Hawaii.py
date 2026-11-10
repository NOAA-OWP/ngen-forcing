from Forcing_Extraction_Scripts.forecast_download_base import MRMSMultiSensorDownloader


class MRMSMultiSensorHawaiiDownloader(MRMSMultiSensorDownloader):
    """Downloader for MRMS MultiSensor hourly QPE (Pass1 and Pass2) for Hawaii."""

    region = "HAWAII"


if __name__ == "__main__":
    MRMSMultiSensorHawaiiDownloader.cli_main()
