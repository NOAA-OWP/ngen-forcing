import os
import time
import shutil
from datetime import datetime, timedelta
import requests
from typing import Optional, Dict
from .glofs_downloader import download_glofs_range

class DataDownloader:
    """
    Downloader for meteo/hydro/coastal sources.
    - Writes to raw_download_dir/<category>/<source>/
    """

    def __init__(
        self,
        coastal_model: str,
        start_time: str,
        end_time: str,
        meteo_source: str,
        hydrology_source: str,
        coastal_water_level_source: str,
        raw_download_dir: str,
        nwm_domain: str,
        # optional: provide local paths for sources that already exist on disk
        # local_paths: dict | None = None,
        local_paths: Optional[Dict] = None,
        domain_info: Optional[Dict] = None
    ):
        self.start_dt = self._parse_time(start_time)
        self.end_dt   = self._parse_time(end_time)
        self.meteo    = meteo_source.lower()
        self.hydro    = hydrology_source.lower()
        self.coastal  = coastal_water_level_source.lower()
        self.raw_root = os.path.normpath(raw_download_dir)
        self.local_paths = local_paths or {}
        self.domain_info = domain_info
        self.coastal_model = coastal_model.lower() 
        self.domain = nwm_domain.lower() 

    # ---------- Public API ----------
    def download_all(self):
        if not os.path.isdir(self.raw_root):
            os.makedirs(self.raw_root, exist_ok=True)

        # Meteo
        if self.meteo in ("nwm_retro", "nwm_ana"):
            self._download_meteo()
        else:
            print(f"[meteo] Unknown source '{self.meteo}', skipping.")

        # Hydrology
        if self.hydro in ("nwm", "ngen"):
            self._download_hydro()
        else:
            print(f"[hydro] Unknown source '{self.hydro}', skipping.")

        # Coastal water level
        if self.coastal in ("stofs", "tpxo", "glofs"):
            self._download_coastal()
        else:
            print(f"[coastal] Unknown source '{self.coastal}', skipping.")
        # --- Meteo retrospective ---

    # --- Meteo retrospective (NWM R3 FORCING LDASIN) ---

    def _download_nwm_meteo_retrospective(self):
        """
        Download NWM Retrospective 3.0 CONUS FORCING (LDASIN_DOMAIN1) hourly files.

        Source pattern (no .nc extension in the object name):
          https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com/
              CONUS/netcdf/FORCING/{YYYY}/{YYYYMMDDHHMM}.LDASIN_DOMAIN1

        We save them as:
          data/raw/meteo/nwm_retro/nwm_forcing_{YYYYMMDD}_{HH}.nc
        to match the processor's expected filenames.
        """
        base = "https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com"
        outdir = self._ensure_dir("meteo", "nwm_retro")

        domain_str= 'CONUS' if self.domain == 'conus' \
                    else 'Hawaii' if self.domain == 'hawaii' else 'PR'    \
                    if self.domain == "prvi"  else 'UNKNOWN'

        for dt in self._iter_hours():
            year = dt.strftime("%Y")
            if domain_str == 'CONUS':
                stamp_ymdhm = dt.strftime("%Y%m%d%H") + "00"  # minute is always 00 in this product
            else:
                stamp_ymdhm = dt.strftime("%Y%m%d%H")

            url = f"{base}/{domain_str}/netcdf/FORCING/{year}/{stamp_ymdhm}.LDASIN_DOMAIN1"

            date_str = dt.strftime("%Y%m%d")
            hour_str = f"{dt.hour:02d}"
            stamp_ymdh = dt.strftime("%Y%m%d%H")

            dest = os.path.join(outdir, f"{stamp_ymdhm}.LDASIN_DOMAIN1.nc") if  \
                  self.coastal_model == "sfincs" else  \
                   os.path.join(outdir, f"{stamp_ymdh}.LDASIN_DOMAIN1")

            self._safe_download(url, dest)


    # --- Hydro retrospective ---
    def _download_nwm_channel_rt_retrospective_orig(self):
        """
        Analogous to _download_nwm_channel_rt_ana, but retrospective path.
        """
        base = "https://<RETRO_ENDPOINT>"  # TODO
        outdir = self._ensure_dir("hydro", "nwm_retro")
        for dt in self._iter_hours():
            date_str = dt.strftime("%Y%m%d")
            hour_str = f"{dt.hour:02d}"
            url = f"{base}/nwm.{date_str}/.../nwm.t{hour_str}z....channel_rt...nc"  # TODO confirm
            dest = os.path.join(outdir, f"nwm_channel_rt_{date_str}_{hour_str}.nc")
            self._safe_download(url, dest)

    # --- Streamflow retrospective (NWM R3 CHRTOUT) ---
    def _download_nwm_channel_rt_retrospective(self):
        """
        Download NWM Retrospective 3.0 CONUS CHRTOUT hourly files.

        Source pattern (no .nc extension in the object name):
          https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com/
              CONUS/netcdf/CHRTOUT/{YYYY}/{YYYYMMDDHHMM}.CHRTOUT_DOMAIN1

        We save them as:
          data/raw/streamflow/nwm_retro/CHRTOUT_{YYYYMMDD}_{HH}.nc
        """
        base = "https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com"
        outdir = self._ensure_dir("streamflow", "nwm_retro")

        if self.coastal_model == "sfincs" : 
          for dt in self._iter_hours():
            year = dt.strftime("%Y")
            stamp_ymdhm = dt.strftime("%Y%m%d%H") + "00"  # always minute 00
            url = f"{base}/CONUS/netcdf/CHRTOUT/{year}/{stamp_ymdhm}.CHRTOUT_DOMAIN1"

            date_str = dt.strftime("%Y%m%d")
            hour_str = f"{dt.hour:02d}"
            dest = os.path.join(outdir, f"{stamp_ymdhm}.CHRTOUT_DOMAIN1.nc")

            self._safe_download(url, dest)

        else:
          domain_str= 'CONUS' if self.domain == 'conus' \
                    else 'Hawaii' if self.domain == 'hawaii' else 'PR'    \
                    if self.domain == "prvi"  else 'UNKNOWN'

          for dt in self._iter_hours():
            year = dt.strftime("%Y")
            stamp_ymdhm = dt.strftime("%Y%m%d%H") + "00"  # always minute 00
            url = f"{base}/{domain_str}/netcdf/CHRTOUT/{year}/{stamp_ymdhm}.CHRTOUT_DOMAIN1"

            date_str = dt.strftime("%Y%m%d")
            hour_str = f"{dt.hour:02d}"

            dest = os.path.join(outdir, f"{stamp_ymdhm}.CHRTOUT_DOMAIN1.nc") if \
                           self.coastal_model == "sfincs" else  \
                         os.path.join(outdir, f"{stamp_ymdhm}.CHRTOUT_DOMAIN1")
            self._safe_download(url, dest)

            if self.domain == 'hawaii':
                for t in {15,30,45}:
                    stamp_ymdhm = dt.strftime("%Y%m%d%H") + f"{t:02d}"  
                    url = f"{base}/Hawaii/netcdf/CHRTOUT/{year}/{stamp_ymdhm}.CHRTOUT_DOMAIN1"
                    dest = os.path.join(outdir, f"{stamp_ymdhm}.CHRTOUT_DOMAIN1.nc") if \
                           self.coastal_model == "sfincs" else  \
                           os.path.join(outdir, f"{stamp_ymdhm}.CHRTOUT_DOMAIN1")

                    self._safe_download(url, dest)

    # --- STOFS ---
    def _download_stofs(self):
        """
        Download STOFS CONUS East combined surge+tide (cwl) GRIB2 files.

        S3 layout (no listing page):
          https://noaa-gestofs-pds.s3.amazonaws.com/stofs_2d_glo.YYYYMMDD/stofs_2d_glo.tHHz.conus.east.cwl.grib2

        Note: STOFS publishes 4 cycles/day (00, 06, 12, 18 Z). We only attempt those
        to avoid noisy 404s.
        """
        date_name_changed = datetime(2023, 1, 8, 0, 0, 0) #1/8/2023 uses new naming
        base = "https://noaa-gestofs-pds.s3.amazonaws.com"
        if self.start_dt < date_name_changed:
           product = "estofs"
        else:
           product = "stofs_2d_glo"

        region = "conus.east.cwl" if  \
                 self.coastal_model == "sfincs" else "fields.cwl"

        outdir = self._ensure_dir("coastal", "stofs")

        if self.coastal_model == "sfincs" : 
          for dt in self._iter_hours():
            # Only try valid STOFS cycles
            if dt.hour not in (0, 6, 12, 18):
                continue

            date_str = dt.strftime("%Y%m%d")
            hour = f"{dt.hour:02d}"

            url = f"{base}/{product}.{date_str}/{product}.t{hour}z.{region}.grib2"

            # Keep the original filename from the URL
            fname = os.path.basename(url)
            dest = os.path.join(outdir, fname)

            self._safe_download(url, dest)
        else:

          date_str = self.start_dt.strftime("%Y%m%d")
          hour = f"{( self.start_dt.hour ) % 6 * 6:02d}"
          url = f"{base}/{product}.{date_str}/{product}.t{hour}z.{region}.nc"

          # Keep the original filename from the URL
          fname = os.path.basename(url)
          dest = os.path.join(outdir, fname)
          self._safe_download(url, dest)


    # --- TPXO ---
    def _download_tpxo(self):
        """
        TPXO is typically licensed/data-managed; often distributed as static files (binary/NetCDF).
        For now: support a 'local mirror' copy pattern so the workflow is uniform.
        """
        local = self.local_paths.get("tpxo")
        outdir = self._ensure_dir("coastal", "tpxo")
        if local:
            self._copy_tree(local, outdir)
        else:
            print("[coastal:tpxo] no remote download; set local_paths['tpxo']")


    def _download_glofs(self):
        """
        Download GLOFS forecast/nowcast files over the configured time window
        using download_glofs_range().
        """
        from .glofs_downloader import download_glofs_range

        outdir = self._ensure_dir("coastal", "glofs")

        access_area = self.domain_info["domain"][0].get("access_area", "lake-erie-operational-forecast-system-leofs")

        try:
            files = download_glofs_range(
                start=self.start_dt,
                end=self.end_dt,
                step=timedelta(hours=1),
                outdir=outdir,
                access_area=access_area,
                base_url=os.environ.get(
                    "GLOFS_BASE_URL",
                    "https://www.ncei.noaa.gov/data/"
                    "operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access",
                ),
                allow_cached=True,
            )
            print(f"[coastal:glofs] wrote {len(files)} files to {outdir}")
        except Exception as e:
            print(f"[coastal:glofs] failed: {e}")



    def _download_glofs_schism(self):
        """
        Download GLOFS files for the configured [start_dt, end_dt) range.

        - Groups hourly window into unique UTC dates and calls `download_glofs`
          once per day.
        - Output goes under: <raw_download_dir>/coastal/glofs/<domain>/
        - Domains default to all ("leofs,lmhofs,loofs,lsofs"), override via:
              GLOFS_DOMAINS="leofs,lsofs"
        - Overwrite behavior defaults to False, override via:
              GLOFS_OVERWRITE=1
        """
        outdir = self._ensure_dir("coastal", "glofs")

        # Domains from env or default to all four

        '''
        domains_env = os.environ.get("GLOFS_DOMAINS", "")
        if domains_env.strip():
            domains = [d.strip() for d in domains_env.split(",") if d.strip()]
        else:
            domains = ["leofs", "lmhofs", "loofs", "lsofs"]

        # Overwrite flag from env (0/1, false/true)
        overwrite_env = os.environ.get("GLOFS_OVERWRITE", "").strip().lower()
        overwrite = overwrite_env in ("1", "true", "yes", "y")
        '''

        # Collect unique yyyymmdd strings across the hourly window
        unique_dates = sorted({dt.strftime("%Y%m%d") for dt in self._iter_hours()})

        for ymd in unique_dates:
            try:
                download_glofs(
                    utcdate=ymd,
                    domains=["leofs"],
                    output_dir=outdir,
                    overwrite=False,
                )
            except Exception as e:
                print(f"[coastal:glofs] date {ymd} failed: {e}")


    # ---------- Internals ----------
    def _download_meteo(self):
        if self.meteo == "nwm_ana":
            print("\ndownload_nwm_meteo_ana:")
            self._download_nwm_meteo_ana()
        elif self.meteo == "nwm_retro":
            print("\ndownload_nwm_meteo_retrospective")
            self._download_nwm_meteo_retrospective()
        else:
            print(f"[meteo] Unhandled meteo source '{self.meteo}'")

    def _download_hydro(self):
        if self.hydro == "nwm" and self.meteo == "nwm_ana":
            print("\ndownload_nwm_channel_rt_ana:")
            self._download_nwm_channel_rt_ana()
        elif self.hydro == "nwm" and self.meteo == "nwm_retro":
            print("\ndownload_nwm_channel_rt_ana:")
            self._download_nwm_channel_rt_retrospective()
        elif self.hydro == "ngen":
            print(f"[hydro:ngen] not available yet")
            # TODO: link to ngen - not a priority right now

    def _download_coastal(self):
        if self.coastal == "stofs":
            print("\n[coastal:stofs]")
            self._download_stofs()
        elif self.coastal == "tpxo":
            if self.coastal_model == "sfincs" : 
              print("\n[coastal:tpxo]")
              self._download_tpxo()
            else:
              pass
        elif self.coastal == "glofs":
            print("[coastal:glofs]")
            # self._download_glofs()
        else:
            print(f"[hydro] Unhandled hydro source '{self.hydro}'")

    # ---------- Concrete downloaders ----------
    def _download_nwm_meteo_ana(self):
        """
        Example URL pattern (provided):
        https://storage.googleapis.com/national-water-model/nwm.{date}/forcing_analysis_assim/nwm.t{hour:02d}z.analysis_assim.forcing.tm00.conus.nc
        - date as YYYYMMDD
        - hour as 00..23
        """
        base = "https://storage.googleapis.com/national-water-model"
        outdir = self._ensure_dir("meteo", "nwm_ana")

        if self.coastal_model == "sfincs" : 
          for dt in self._iter_hours():
            date_str = dt.strftime("%Y%m%d")
            hour_str = f"{dt.hour:02d}"
            url = f"{base}/nwm.{date_str}/forcing_analysis_assim/nwm.t{hour_str}z.analysis_assim.forcing.tm00.conus.nc"
            filename = f"nwm_forcing_{date_str}_{hour_str}.nc"
            dest = os.path.join(outdir, filename)
            self._safe_download(url, dest)

        else:
          for dt in self._iter_hours():
            temp_dt = dt + timedelta(hours = 2)
            date_str = temp_dt.strftime("%Y%m%d")
            hour_str = f"{temp_dt.hour:02d}"
            domain_str= '' if self.domain == 'conus' \
                    else "_puertorico" if self.domain == 'prvi' else f"_{self.domain}"
            domain_str2= 'conus' if self.domain == 'conus' \
                    else "puertorico" if self.domain == 'prvi' else f"{self.domain}"

            url = f"{base}/nwm.{date_str}/forcing_analysis_assim{domain_str}/nwm.t{hour_str}z.analysis_assim.forcing.tm02.{domain_str2}.nc"
            #filename = f"nwm_forcing_{date_str}_{hour_str}.nc"
            filename = f"{date_str}{dt.hour:02d}.LDASIN_DOMAIN1"
            dest = os.path.join(outdir, filename)
            self._safe_download(url, dest)

    def _download_nwm_channel_rt_ana(self):
        """
        Example URL pattern (provided):
        https://storage.googleapis.com/national-water-model/nwm.{date}/analysis_assim/nwm.t{hour:02d}z.analysis_assim.channel_rt.tm00.conus.nc
        """
        base = "https://storage.googleapis.com/national-water-model"
        outdir = self._ensure_dir("hydro", "nwm")

        if self.coastal_model == "sfincs" : 
          for dt in self._iter_hours():
            date_str = dt.strftime("%Y%m%d")
            hour_str = f"{dt.hour:02d}"
            url = f"{base}/nwm.{date_str}/analysis_assim/nwm.t{hour_str}z.analysis_assim.channel_rt.tm00.conus.nc"
            filename = f"nwm_channel_rt_{date_str}_{hour_str}.nc"
            dest = os.path.join(outdir, filename)
            self._safe_download(url, dest)

        else:
          for dt in self._iter_hours():
            temp_dt = dt + timedelta(hours = 2)
            date_str = temp_dt.strftime("%Y%m%d")
            hour_str = f"{temp_dt.hour:02d}"
            domain_str= '' if self.domain == 'conus' \
                    else "_puertorico" if self.domain == 'prvi' else f"_{self.domain}"
            domain_str2= 'conus' if self.domain == 'conus' \
                    else "puertorico" if self.domain == 'prvi' else f"{self.domain}"

            if self.domain == 'hawaii':
               date_str = dt.strftime("%Y%m%d")
               url = f"{base}/nwm.{date_str}/analysis_assim{domain_str}/nwm.t{hour_str}z.analysis_assim.channel_rt.tm0200.{domain_str2}.nc"
               filename = f"{date_str}{dt.hour:02d}00.CHRTOUT_DOMAIN1"
               dest = os.path.join(outdir, filename)
               self._safe_download(url, dest)
               for t in [45, 30, 15, 00]:
                  temp_dt = dt + timedelta(hours = 1)
                  temp_dt = temp_dt - timedelta( minutes = t)
                  temp_date_str = temp_dt.strftime("%Y%m%d")
                  url = f"{base}/nwm.{temp_date_str}/analysis_assim{domain_str}/nwm.t{hour_str}z.analysis_assim.channel_rt.tm01{t:02d}.{domain_str2}.nc"
                  filename = f"{date_str}{temp_dt.hour:02d}{(60-t)%60:02d}.CHRTOUT_DOMAIN1"
                  dest = os.path.join(outdir, filename)
                  self._safe_download(url, dest)
            else:
               url = f"{base}/nwm.{date_str}/analysis_assim{domain_str}/nwm.t{hour_str}z.analysis_assim.channel_rt.tm02.{domain_str2}.nc"
               #filename = f"nwm_channel_rt_{date_str}_{hour_str}.nc"
               temp_date_str = dt.strftime("%Y%m%d")
               filename = f"{temp_date_str}{dt.hour:02d}00.CHRTOUT_DOMAIN1"
               dest = os.path.join(outdir, filename)
               self._safe_download(url, dest)


    # ---------- Utilities ----------
    def _iter_hours(self):
        """Yield datetimes at hourly steps: [start_dt, end_dt)"""
        current = self.start_dt
        if self.coastal_model == "schism":
            end_time = self.end_dt + timedelta(hours=1)
        else:
            end_time = self.end_dt

        while current < end_time:
            yield current
            current += timedelta(hours=1)

    # --- Hydro retrospective ---
    def _download_nwm_channel_rt_retrospective_orig(self):
        """
        Analogous to _download_nwm_channel_rt_ana, but retrospective path.
        """
        base = "https://<RETRO_ENDPOINT>"  # TODO
        outdir = self._ensure_dir("hydro", "nwm_retro")
        for dt in self._iter_hours():
            date_str = dt.strftime("%Y%m%d")
            hour_str = f"{dt.hour:02d}"
            url = f"{base}/nwm.{date_str}/.../nwm.t{hour_str}z....channel_rt...nc"  # TODO confirm
            dest = os.path.join(outdir, f"nwm_channel_rt_{date_str}_{hour_str}.nc")
            self._safe_download(url, dest)

    # --- Streamflow retrospective (NWM R3 CHRTOUT) ---
    def _download_nwm_channel_rt_retrospective(self):
        """
        Download NWM Retrospective 3.0 CONUS CHRTOUT hourly files.

        Source pattern (no .nc extension in the object name):
          https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com/
              CONUS/netcdf/CHRTOUT/{YYYY}/{YYYYMMDDHHMM}.CHRTOUT_DOMAIN1

        We save them as:
          data/raw/streamflow/nwm_retro/CHRTOUT_{YYYYMMDD}_{HH}.nc
        """
        base = "https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com"
        outdir = self._ensure_dir("streamflow", "nwm_retro")

        if self.coastal_model == "sfincs" : 
          for dt in self._iter_hours():
            year = dt.strftime("%Y")
            stamp_ymdhm = dt.strftime("%Y%m%d%H") + "00"  # always minute 00
            url = f"{base}/CONUS/netcdf/CHRTOUT/{year}/{stamp_ymdhm}.CHRTOUT_DOMAIN1"

            date_str = dt.strftime("%Y%m%d")
            hour_str = f"{dt.hour:02d}"
            dest = os.path.join(outdir, f"{stamp_ymdhm}.CHRTOUT_DOMAIN1.nc")

            self._safe_download(url, dest)

        else:
          domain_str= 'CONUS' if self.domain == 'conus' \
                    else 'Hawaii' if self.domain == 'hawaii' else 'PR'    \
                    if self.domain == "prvi"  else 'UNKNOWN'

          for dt in self._iter_hours():
            year = dt.strftime("%Y")
            stamp_ymdhm = dt.strftime("%Y%m%d%H") + "00"  # always minute 00
            url = f"{base}/{domain_str}/netcdf/CHRTOUT/{year}/{stamp_ymdhm}.CHRTOUT_DOMAIN1"

            date_str = dt.strftime("%Y%m%d")
            hour_str = f"{dt.hour:02d}"

            dest = os.path.join(outdir, f"{stamp_ymdhm}.CHRTOUT_DOMAIN1.nc") if \
                           self.coastal_model == "sfincs" else  \
                         os.path.join(outdir, f"{stamp_ymdhm}.CHRTOUT_DOMAIN1")
            self._safe_download(url, dest)

            if self.domain == 'hawaii':
                for t in {15,30,45}:
                    stamp_ymdhm = dt.strftime("%Y%m%d%H") + f"{t:02d}"  
                    url = f"{base}/Hawaii/netcdf/CHRTOUT/{year}/{stamp_ymdhm}.CHRTOUT_DOMAIN1"
                    dest = os.path.join(outdir, f"{stamp_ymdhm}.CHRTOUT_DOMAIN1.nc") if \
                           self.coastal_model == "sfincs" else  \
                           os.path.join(outdir, f"{stamp_ymdhm}.CHRTOUT_DOMAIN1")

                    self._safe_download(url, dest)

    # --- STOFS ---
    def _download_stofs(self):
        """
        Download STOFS CONUS East combined surge+tide (cwl) GRIB2 files.

        S3 layout (no listing page):
          https://noaa-gestofs-pds.s3.amazonaws.com/stofs_2d_glo.YYYYMMDD/stofs_2d_glo.tHHz.conus.east.cwl.grib2

        Note: STOFS publishes 4 cycles/day (00, 06, 12, 18 Z). We only attempt those
        to avoid noisy 404s.
        """
        date_name_changed = datetime(2023, 1, 8, 0, 0, 0) #1/8/2023 uses new naming
        base = "https://noaa-gestofs-pds.s3.amazonaws.com"
        if self.start_dt < date_name_changed:
           product = "estofs"
        else:
           product = "stofs_2d_glo"

        region = "conus.east.cwl" if  \
                 self.coastal_model == "sfincs" else "fields.cwl"

        outdir = self._ensure_dir("coastal", "stofs")

        if self.coastal_model == "sfincs" : 
          for dt in self._iter_hours():
            # Only try valid STOFS cycles
            if dt.hour not in (0, 6, 12, 18):
                continue

            date_str = dt.strftime("%Y%m%d")
            hour = f"{dt.hour:02d}"

            url = f"{base}/{product}.{date_str}/{product}.t{hour}z.{region}.grib2"

            # Keep the original filename from the URL
            fname = os.path.basename(url)
            dest = os.path.join(outdir, fname)

            self._safe_download(url, dest)
        else:

          date_str = self.start_dt.strftime("%Y%m%d")
          hour = f"{( self.start_dt.hour ) % 6 * 6:02d}"
          url = f"{base}/{product}.{date_str}/{product}.t{hour}z.{region}.nc"

          # Keep the original filename from the URL
          fname = os.path.basename(url)
          dest = os.path.join(outdir, fname)
          self._safe_download(url, dest)


    # --- TPXO ---
    def _download_tpxo(self):
        """
        TPXO is typically licensed/data-managed; often distributed as static files (binary/NetCDF).
        For now: support a 'local mirror' copy pattern so the workflow is uniform.
        """
        local = self.local_paths.get("tpxo")
        outdir = self._ensure_dir("coastal", "tpxo")
        if local:
            self._copy_tree(local, outdir)
        else:
            print("[coastal:tpxo] no remote download; set local_paths['tpxo']")


    def _download_glofs(self):
        """
        Download GLOFS forecast/nowcast files over the configured time window
        using download_glofs_range().
        """
        from .glofs_downloader import download_glofs_range

        outdir = self._ensure_dir("coastal", "glofs")

        access_area = self.domain_info["domain"][0].get("access_area", "lake-erie-operational-forecast-system-leofs")

        try:
            files = download_glofs_range(
                start=self.start_dt,
                end=self.end_dt,
                step=timedelta(hours=1),
                outdir=outdir,
                access_area=access_area,
                base_url=os.environ.get(
                    "GLOFS_BASE_URL",
                    "https://www.ncei.noaa.gov/data/"
                    "operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access",
                ),
                allow_cached=True,
            )
            print(f"[coastal:glofs] wrote {len(files)} files to {outdir}")
        except Exception as e:
            print(f"[coastal:glofs] failed: {e}")



    def _download_glofs_schism(self):
        """
        Download GLOFS files for the configured [start_dt, end_dt) range.

        - Groups hourly window into unique UTC dates and calls `download_glofs`
          once per day.
        - Output goes under: <raw_download_dir>/coastal/glofs/<domain>/
        - Domains default to all ("leofs,lmhofs,loofs,lsofs"), override via:
              GLOFS_DOMAINS="leofs,lsofs"
        - Overwrite behavior defaults to False, override via:
              GLOFS_OVERWRITE=1
        """
        outdir = self._ensure_dir("coastal", "glofs")

        # Domains from env or default to all four

        '''
        domains_env = os.environ.get("GLOFS_DOMAINS", "")
        if domains_env.strip():
            domains = [d.strip() for d in domains_env.split(",") if d.strip()]
        else:
            domains = ["leofs", "lmhofs", "loofs", "lsofs"]

        # Overwrite flag from env (0/1, false/true)
        overwrite_env = os.environ.get("GLOFS_OVERWRITE", "").strip().lower()
        overwrite = overwrite_env in ("1", "true", "yes", "y")
        '''

        # Collect unique yyyymmdd strings across the hourly window
        unique_dates = sorted({dt.strftime("%Y%m%d") for dt in self._iter_hours()})

        for ymd in unique_dates:
            try:
                download_glofs(
                    utcdate=ymd,
                    domains=["leofs"],
                    output_dir=outdir,
                    overwrite=False,
                )
            except Exception as e:
                print(f"[coastal:glofs] date {ymd} failed: {e}")


    # ---------- Internals ----------
    def _download_meteo(self):
        if self.meteo == "nwm_ana":
            print("\ndownload_nwm_meteo_ana:")
            self._download_nwm_meteo_ana()
        elif self.meteo == "nwm_retro":
            print("\ndownload_nwm_meteo_retrospective")
            self._download_nwm_meteo_retrospective()
        else:
            print(f"[meteo] Unhandled meteo source '{self.meteo}'")

    def _download_hydro(self):
        if self.hydro == "nwm" and self.meteo == "nwm_ana":
            print("\ndownload_nwm_channel_rt_ana:")
            self._download_nwm_channel_rt_ana()
        elif self.hydro == "nwm" and self.meteo == "nwm_retro":
            print("\ndownload_nwm_channel_rt_ana:")
            self._download_nwm_channel_rt_retrospective()
        elif self.hydro == "ngen":
            print(f"[hydro:ngen] not available yet")
            # TODO: link to ngen - not a priority right now

    def _download_coastal(self):
        if self.coastal == "stofs":
            print("\n[coastal:stofs]")
            self._download_stofs()
        elif self.coastal == "tpxo":
            print("\n[coastal:tpxo]")
            self._download_tpxo()
        elif self.coastal == "glofs":
            print("[coastal:glofs]")
            # self._download_glofs()
        else:
            print(f"[hydro] Unhandled hydro source '{self.hydro}'")

    # ---------- Concrete downloaders ----------
    def _download_nwm_meteo_ana(self):
        """
        Example URL pattern (provided):
        https://storage.googleapis.com/national-water-model/nwm.{date}/forcing_analysis_assim/nwm.t{hour:02d}z.analysis_assim.forcing.tm00.conus.nc
        - date as YYYYMMDD
        - hour as 00..23
        """
        base = "https://storage.googleapis.com/national-water-model"
        outdir = self._ensure_dir("meteo", "nwm_ana")

        if self.coastal_model == "sfincs" : 
          for dt in self._iter_hours():
            date_str = dt.strftime("%Y%m%d")
            hour_str = f"{dt.hour:02d}"
            url = f"{base}/nwm.{date_str}/forcing_analysis_assim/nwm.t{hour_str}z.analysis_assim.forcing.tm00.conus.nc"
            filename = f"nwm_forcing_{date_str}_{hour_str}.nc"
            dest = os.path.join(outdir, filename)
            self._safe_download(url, dest)

        else:
          for dt in self._iter_hours():
            temp_dt = dt + timedelta(hours = 2)
            date_str = temp_dt.strftime("%Y%m%d")
            hour_str = f"{temp_dt.hour:02d}"
            domain_str= '' if self.domain == 'conus' \
                    else "_puertorico" if self.domain == 'prvi' else f"_{self.domain}"
            domain_str2= 'conus' if self.domain == 'conus' \
                    else "puertorico" if self.domain == 'prvi' else f"{self.domain}"

            url = f"{base}/nwm.{date_str}/forcing_analysis_assim{domain_str}/nwm.t{hour_str}z.analysis_assim.forcing.tm02.{domain_str2}.nc"
            #filename = f"nwm_forcing_{date_str}_{hour_str}.nc"
            filename = f"{date_str}{dt.hour:02d}.LDASIN_DOMAIN1"
            dest = os.path.join(outdir, filename)
            self._safe_download(url, dest)

    def _download_nwm_channel_rt_ana(self):
        """
        Example URL pattern (provided):
        https://storage.googleapis.com/national-water-model/nwm.{date}/analysis_assim/nwm.t{hour:02d}z.analysis_assim.channel_rt.tm00.conus.nc
        """
        base = "https://storage.googleapis.com/national-water-model"
        outdir = self._ensure_dir("hydro", "nwm")

        if self.coastal_model == "sfincs" : 
          for dt in self._iter_hours():
            date_str = dt.strftime("%Y%m%d")
            hour_str = f"{dt.hour:02d}"
            url = f"{base}/nwm.{date_str}/analysis_assim/nwm.t{hour_str}z.analysis_assim.channel_rt.tm00.conus.nc"
            filename = f"nwm_channel_rt_{date_str}_{hour_str}.nc"
            dest = os.path.join(outdir, filename)
            self._safe_download(url, dest)

        else:
          for dt in self._iter_hours():
            temp_dt = dt + timedelta(hours = 2)
            date_str = temp_dt.strftime("%Y%m%d")
            hour_str = f"{temp_dt.hour:02d}"
            domain_str= '' if self.domain == 'conus' \
                    else "_puertorico" if self.domain == 'prvi' else f"_{self.domain}"
            domain_str2= 'conus' if self.domain == 'conus' \
                    else "puertorico" if self.domain == 'prvi' else f"{self.domain}"

            if self.domain == 'hawaii':
               date_str = dt.strftime("%Y%m%d")
               url = f"{base}/nwm.{date_str}/analysis_assim{domain_str}/nwm.t{hour_str}z.analysis_assim.channel_rt.tm0200.{domain_str2}.nc"
               filename = f"{date_str}{dt.hour:02d}00.CHRTOUT_DOMAIN1"
               dest = os.path.join(outdir, filename)
               self._safe_download(url, dest)
               for t in [45, 30, 15, 00]:
                  temp_dt = dt + timedelta(hours = 1)
                  temp_dt = temp_dt - timedelta( minutes = t)
                  temp_date_str = temp_dt.strftime("%Y%m%d")
                  url = f"{base}/nwm.{temp_date_str}/analysis_assim{domain_str}/nwm.t{hour_str}z.analysis_assim.channel_rt.tm01{t:02d}.{domain_str2}.nc"
                  filename = f"{date_str}{temp_dt.hour:02d}{(60-t)%60:02d}.CHRTOUT_DOMAIN1"
                  dest = os.path.join(outdir, filename)
                  self._safe_download(url, dest)
            else:
               url = f"{base}/nwm.{date_str}/analysis_assim{domain_str}/nwm.t{hour_str}z.analysis_assim.channel_rt.tm02.{domain_str2}.nc"
               #filename = f"nwm_channel_rt_{date_str}_{hour_str}.nc"
               temp_date_str = dt.strftime("%Y%m%d")
               filename = f"{temp_date_str}{dt.hour:02d}00.CHRTOUT_DOMAIN1"
               dest = os.path.join(outdir, filename)
               self._safe_download(url, dest)


    # ---------- Utilities ----------
    def _iter_hours(self):
        """Yield datetimes at hourly steps: [start_dt, end_dt)"""
        current = self.start_dt
        if self.coastal_model == "schism":
            end_time = self.end_dt + timedelta(hours=1)
        else:
            end_time = self.end_dt
        while current < end_time:
            yield current
            current += timedelta(hours=1)

    def _ensure_dir(self, category: str, source: str) -> str:
        """Ensure a subfolder raw_download_dir/<category>/<source>/ exists."""
        d = os.path.join(self.raw_root, category, source)
        os.makedirs(d, exist_ok=True)
        return d


    def _safe_download(self, url: str, dest: str, max_retries: int = 3, backoff_sec: float = 1.5, overwrite: bool = False):
        """
        Stream download with simple retries.

        Parameters
        ----------
        url : str
            Remote URL to download.
        dest : str
            Destination file path.
        max_retries : int, optional
            Number of retry attempts on failure. Default = 3.
        backoff_sec : float, optional
            Backoff multiplier between retries. Default = 1.5.
        overwrite : bool, optional
            If False (default), skip download if file already exists.
            If True, always download (replace any existing file).
        """
        # Skip if file exists and overwrite is False
        if not overwrite and os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[safe_download] exists (skipped): {dest}")
            return

        for attempt in range(1, max_retries + 1):
            try:
                print(f"GET {url} -> {dest}")
                with requests.get(url, stream=True, timeout=60) as r:
                    if r.status_code != 200:
                        raise RuntimeError(f"HTTP {r.status_code}")
                    tmp = dest + ".part"
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                    # move into place
                    if os.path.exists(dest):
                        os.remove(dest)
                    os.replace(tmp, dest)
                return
            except Exception as e:
                print(f"    attempt {attempt}/{max_retries} failed: {e}")
                time.sleep(backoff_sec * attempt)

        print(f"    FAILED after {max_retries} attempts: {url}")


    @staticmethod
    def _copy_tree(src_dir: str, dst_dir: str):
        """Shallow copy: files at top-level of src_dir into dst_dir (no recursion by default)."""
        if not os.path.isdir(src_dir):
            print(f"  WARNING: local path not found: {src_dir}")
            return
        os.makedirs(dst_dir, exist_ok=True)
        for name in os.listdir(src_dir):
            s = os.path.join(src_dir, name)
            d = os.path.join(dst_dir, name)
            if os.path.isfile(s):
                shutil.copy2(s, d)

    @staticmethod
    def _parse_time(s: str) -> datetime:
        """
        Parse the config format.
        """

        # Try the exact pattern first
        fmts = [
            "%Y-%m-%dT%H-%M-%SZ",  # 2025-05-11T00-00-00Z
            "%Y-%m-%dT%H:%M:%SZ",  # ISO-like with colons
            "%Y-%m-%d %H:%M:%S",   # fallback
        ]
        for fmt in fmts:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        # Last resort: let datetime try fromisoformat after normalizing Z
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception as e:
            raise ValueError(f"Unrecognized time format: {s}") from e
