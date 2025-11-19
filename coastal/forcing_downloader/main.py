#! /usr/bin/env python

import os
import shutil
import re
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
#from download_data.data_downloader import DataDownloader
from data_downloader import DataDownloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str, help='output directory')
    parser.add_argument('domain', type=str, help='domain name, one of hawaii, prvi, pacfic, or atlgulf')
    parser.add_argument('start_time', type=str, help='Start time of the SCHISM simulation period in a string that is supported by the Python dateutil.parser package, for example, "2024-01-02 00:00"')
    parser.add_argument('end_time', type=str, help='End time of the SCHISM simulation period in a string that is supported by the Python dateutil.parser package, for example, "2024-01-02 23:00"')
    parser.add_argument('meteo_source', type=str, help='nwm_retro or nwm_ana')
    parser.add_argument('hydrology_source', type=str, help='nwm or ngen')
    parser.add_argument('coastal_water_level_source', type=str, help='stofs, tpxo, glofs (great lakes only)')

    args = parser.parse_args()

    # Download data
    downloader = DataDownloader(
        start_time=args.start_time,
        end_time=args.end_time,
        meteo_source=args.meteo_source,
        hydrology_source=args.hydrology_source,
        coastal_water_level_source=args.coastal_water_level_source,
        raw_download_dir=args.output_dir,
        domain=args.domain
    )
    downloader.download_all()

if __name__ == "__main__":
   try:
      main()
   except Exception as e:
      logger.error("Failed to get program options.", exc_info=True)

