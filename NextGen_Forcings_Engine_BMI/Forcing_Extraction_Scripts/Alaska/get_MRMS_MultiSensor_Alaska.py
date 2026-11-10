from Forcing_Extraction_Scripts.forecast_download_base import MRMSMultiSensorDownloader


class MRMSMultiSensorAlaskaDownloader(MRMSMultiSensorDownloader):
    """Downloader for MRMS MultiSensor hourly QPE (Pass1 and Pass2) for Alaska."""

    region = "ALASKA"


if __name__ == "__main__":
    MRMSMultiSensorAlaskaDownloader.cli_main()
