import os
import time
import shutil
from datetime import datetime, timedelta
import requests
from typing import Optional, Dict

class DataDownloader:
    """
    Downloader for meteo/hydro/coastal sources.
    - Writes to raw_download_dir/<category>/<source>/
    """

    def __init__(
        self,
        start_time: str,
        end_time: str,
        meteo_source: str,
        hydrology_source: str,
        coastal_water_level_source: str,
        raw_download_dir: str,
        domain: str,
        # optional: provide local paths for sources that already exist on disk
        # local_paths: dict | None = None,
        local_paths: Optional[Dict] = None
    ):
        self.start_dt = self._parse_time(start_time)
        self.end_dt   = self._parse_time(end_time)
        self.meteo    = meteo_source.lower()
        self.hydro    = hydrology_source.lower()
        self.coastal  = coastal_water_level_source.lower()
        self.raw_root = os.path.normpath(raw_download_dir)
        self.domain = domain.lower()
        self.local_paths = local_paths or {}

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

    def _download_nwm_meteo_retrospective_orig(self):
        """
        NWM Retrospective (R2/R3 etc.) usually lives at NOAA/NCAR endpoints.
        TODO: Confirm exact URL pattern for your deployment.
        Sketch (example only):
          https://noaa-nwm-retrospective.s3.amazonaws.com/.../nwm.t{HH}z.analysis_assim.forcing.conus.nc
        """
        base = "https://<RETRO_ENDPOINT>"  # TODO
        outdir = self._ensure_dir("meteo", "nwm_retro")
        for dt in self._iter_hours():
            date_str = dt.strftime("%Y%m%d")
            hour_str = f"{dt.hour:02d}"
            url = f"{base}/nwm.{date_str}/forcing/.../nwm.t{hour_str}z....nc"  # TODO confirm
            dest = os.path.join(outdir, f"nwm_forcing_{date_str}_{hour_str}.nc")
            self._safe_download(url, dest)

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
            dest = os.path.join(outdir, f"{stamp_ymdh}.LDASIN_DOMAIN1")

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

        domain_str= 'CONUS' if self.domain == 'conus' \
                    else 'Hawaii' if self.domain == 'hawaii' else 'PR'    \
                    if self.domain == "prvi"  else 'UNKNOWN'

        for dt in self._iter_hours():
            year = dt.strftime("%Y")
            stamp_ymdhm = dt.strftime("%Y%m%d%H") + "00"  # always minute 00
            url = f"{base}/{domain_str}/netcdf/CHRTOUT/{year}/{stamp_ymdhm}.CHRTOUT_DOMAIN1"

            date_str = dt.strftime("%Y%m%d")
            hour_str = f"{dt.hour:02d}"
            dest = os.path.join(outdir, f"{stamp_ymdhm}.CHRTOUT_DOMAIN1")
            self._safe_download(url, dest)
            if self.domain == 'hawaii':
                for t in {15,30,45}:
                    stamp_ymdhm = dt.strftime("%Y%m%d%H") + f"{t:02d}"  
                    url = f"{base}/Hawaii/netcdf/CHRTOUT/{year}/{stamp_ymdhm}.CHRTOUT_DOMAIN1"
                    dest = os.path.join(outdir, f"{stamp_ymdhm}.CHRTOUT_DOMAIN1")
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

        #region = "conus.east.cwl"
        region = "fields.cwl"
        outdir = self._ensure_dir("coastal", "stofs")

        date_str = self.start_dt.strftime("%Y%m%d")
        hour = f"{( self.start_dt.hour ) % 6 * 6:02d}"

        #url = f"{base}/{product}.{date_str}/{product}.t{hour}z.{region}.grib2"
        url = f"{base}/{product}.{date_str}/{product}.t{hour}z.{region}.nc"

        # Keep the original filename from the URL
        fname = os.path.basename(url)
        dest = os.path.join(outdir, fname)

        self._safe_download(url, dest)

        #for dt in self._iter_hours():
        #    # Only try valid STOFS cycles
        #    if dt.hour not in (0, 6, 12, 18):
        #        continue
#
#            date_str = dt.strftime("%Y%m%d")
#            hour = f"{dt.hour:02d}"
#
#            url = f"{base}/{product}.{date_str}/{product}.t{hour}z.{region}.grib2"
#
#            # Keep the original filename from the URL
#            fname = os.path.basename(url)
#            dest = os.path.join(outdir, fname)
#
#            self._safe_download(url, dest)
    

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

    # --- GLOFS ---
    def _download_glofs_old(self):
        """
        GLOFS via NCEI THREDDS can be slow programmatically.
        Two modes:
          1) OPeNDAP (your current scripts) – no local download, read-on-the-fly
          2) If required: stage NetCDFs via OPeNDAP to disk with xarray.to_netcdf
        For now: prefer on-the-fly access; just ensure directory exists for parity.
        """
        self._ensure_dir("coastal", "glofs")
        print("[coastal:glofs] using on-the-fly OPeNDAP in processing; no downloads here.")



    # ---------- Internals ----------
    def _download_meteo(self):
        if self.meteo == "nwm_ana":
            self._download_nwm_meteo_ana()
        elif self.meteo == "nwm_retro":
            print("[meteo:nwm_retro] not available yet")
            self._download_nwm_meteo_retrospective()
        else:
            print(f"[meteo] Unhandled meteo source '{self.meteo}'")

    def _download_hydro(self):
        if self.hydro == "nwm" and self.meteo == "nwm_ana":
            self._download_nwm_channel_rt_ana()
        elif self.hydro == "nwm" and self.meteo == "nwm_retro":
            self._download_nwm_channel_rt_retrospective()
        elif self.hydro == "ngen":
            print(f"[hydro:ngen] not available yet")
            # TODO: link to ngen - not a priority right now

    def _download_coastal(self):
        if self.coastal == "stofs":
            print("[coastal:stofs]")
            self._download_stofs()
        elif self.coastal == "tpxo":
            print("[coastal:tpxo]")
            self._download_tpxo()
        elif self.coastal == "glofs":
            print("[coastal:glofs]")
            self._download_glofs()
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


    def _download_glofs(self, model: Optional[str] = None):
        """
        Download GLOFS forecast NetCDF files to the local cache, trying multiple
        filename/host patterns. If a file already exists locally, it is skipped.
        """
        import os

        model_map = {
            "leofs": "lake-erie-operational-forecast-system-leofs",
            "loofs": "lower-ohio-operational-forecast-system-loofs",
            "lsofs": "lake-st-clair-operational-forecast-system-lsofs",
            "lmhofs": "lake-michigan-huron-operational-forecast-system-lmhofs",
        }

        short_model = (
            model
            or getattr(self, "glofs_model", None)
            or os.environ.get("GLOFS_MODEL")
            or "leofs"
        ).lower()

        if short_model not in model_map:
            raise ValueError(f"Unknown GLOFS model '{short_model}'.")

        full_model = model_map[short_model]
        base_access = (
            "https://www.ncei.noaa.gov/data/"
            "operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access"
        )
        base_fileserver = "https://www.ncei.noaa.gov/thredds/fileServer"

        outdir = self._ensure_dir("coastal", "glofs")

        def _try_all_locations(url_paths: list[str]) -> None:
            bases = [base_access, base_fileserver]
            for rel in url_paths:
                for base in bases:
                    url = f"{base}/{rel}"
                    dest = os.path.join(outdir, os.path.basename(url))
                    if os.path.exists(dest) and os.path.getsize(dest) > 0:
                        print(f"[GLOFS] exists: {dest}")
                        return
                    try:
                        print(f"GET {url} -> {dest}")
                        self._safe_download(url, dest)
                        return
                    except Exception as e:
                        print(f"[GLOFS] miss {url}: {e}")

        for dt in self._iter_hours():
            datestr = dt.strftime("%Y%m%d")
            year = dt.strftime("%Y")
            month = dt.strftime("%m")
            cycle_hour = (dt.hour // 6) * 6
            cycle = f"t{cycle_hour:02d}z"
            suffix = f"n{dt.hour % 6:03d}"

            rel_a = f"{full_model}/{year}/{month}/{short_model}.{cycle}.{datestr}.fields.{suffix}.nc"
            rel_b = f"{full_model}/{year}/{month}/nos.{short_model}.fields.{suffix}.{datestr}.{cycle}.nc"
            rel_a_legacy = f"model-{short_model}/{year}/{month}/{short_model}.{cycle}.{datestr}.fields.{suffix}.nc"
            rel_b_legacy = f"model-{short_model}/{year}/{month}/nos.{short_model}.fields.{suffix}.{datestr}.{cycle}.nc"

            _try_all_locations([rel_a, rel_b, rel_a_legacy, rel_b_legacy])


    def _download_glofs2(self, model: Optional[str] = None):
        """
        Download GLOFS forecast files as NetCDFs from NCEI data directory.
        """
        # Map short model codes to full directory names
        model_map = {
            "leofs": "lake-erie-operational-forecast-system-leofs",
            "loofs": "lower-ohio-operational-forecast-system-loofs",
            "lsofs": "lake-st-clair-operational-forecast-system-lsofs",
            "lmhofs": "lake-michigan-huron-operational-forecast-system-lmhofs",
            # add others as needed
        }

        short_model = (
            model
            or getattr(self, "glofs_model", None)
            or os.environ.get("GLOFS_MODEL")
            or "leofs"
        )
        full_model = model_map[short_model]

        base = (
            "https://www.ncei.noaa.gov/data/"
            "operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access"
        )

        outdir = self._ensure_dir("coastal", "glofs")

        for dt in self._iter_hours():
            datestr = dt.strftime("%Y%m%d")
            year = dt.strftime("%Y")
            month = dt.strftime("%m")

            cycle_hour = (dt.hour // 6) * 6
            cycle = f"t{cycle_hour:02d}z"
            suffix = f"n{dt.hour % 6:03d}"

            relpath = f"{full_model}/{year}/{month}/{short_model}.{cycle}.{datestr}.fields.{suffix}.nc"
            url = f"{base}/{relpath}"
            dest = os.path.join(outdir, f"{short_model}.{cycle}.{datestr}.fields.{suffix}.nc")

            self._safe_download(url, dest)


    def _download_glofs_new(self, model: Optional[str] = None):

        """
        Download GLOFS forecast files as NetCDFs from NCEI THREDDS fileServer.

        Mapping of hour -> (cycle, suffix):
          cycle  = t{(hour//6)*6}z   e.g., hour=13 -> t12z
          suffix = n{hour%6:03d}     e.g., hour=13 -> n001

        URL pattern (fileServer):
          https://www.ncei.noaa.gov/thredds/fileServer/model-{model}/{YYYY}/{MM}/
              {model}.{cycle}.{YYYYMMDD}.fields.{suffix}.nc

        Notes:
        - `model` can be one of: leofs, loofs, lsofs, lmhofs (etc. per NCEI).
        - Destination: {raw_download_dir}/coastal/glofs/{model}/{model}.{cycle}.{YYYYMMDD}.fields.{suffix}.nc
        """
        base = "https://www.ncei.noaa.gov/thredds/fileServer"
        model = (
            model
            or getattr(self, "glofs_model", None)
            or os.environ.get("GLOFS_MODEL")
            or "leofs"
        )

        outdir = self._ensure_dir("coastal", "glofs")

        for dt in self._iter_hours():
            datestr = dt.strftime("%Y%m%d")
            year = dt.strftime("%Y")
            month = dt.strftime("%m")

            cycle_hour = (dt.hour // 6) * 6
            cycle = f"t{cycle_hour:02d}z"
            suffix = f"n{dt.hour % 6:03d}"

            relpath = f"model-{model}/{year}/{month}/{model}.{cycle}.{datestr}.fields.{suffix}.nc"
            url = f"{base}/{relpath}"
            dest = os.path.join(outdir, f"{model}.{cycle}.{datestr}.fields.{suffix}.nc")

            self._safe_download(url, dest)


    # ---------- Utilities ----------
    def _iter_hours(self):
        """Yield datetimes at hourly steps: [start_dt, end_dt)"""
        current = self.start_dt
        while current < self.end_dt:
            yield current
            current += timedelta(hours=1)

    def _ensure_dir(self, category: str, source: str) -> str:
        """Ensure a subfolder raw_download_dir/<category>/<source>/ exists."""
        d = os.path.join(self.raw_root, category, source)
        os.makedirs(d, exist_ok=True)
        return d

    def _safe_download(self, url: str, dest: str, max_retries: int = 3, backoff_sec: float = 1.5):
        """Stream download with simple retries; overwrite if file already exists (idempotent runs)."""
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
                print(f"  attempt {attempt}/{max_retries} failed: {e}")
                time.sleep(backoff_sec * attempt)
        print(f"  FAILED after {max_retries} attempts: {url}")

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
