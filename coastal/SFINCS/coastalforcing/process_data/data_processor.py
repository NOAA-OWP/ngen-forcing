import os
import traceback
from datetime import datetime, timedelta
from typing import Iterable, Optional
import numpy as np
import xarray as xr
import pyproj
import geopandas as gpd  # kept if you plan to use later
from shapely.affinity import rotate
from shapely.geometry import box
import pandas as pd
from .glofs_sfincs import build_bzs_from_glofs_legacy
from .mateo_sfincs import write_sfincs_meteo_from_nwm

# from .stofs_timeseries_legacy import process_stofs_timeseries

class DataProcessor:
    """
    Processes raw downloads into SFINCS-ready inputs.

    Expects:
      - raw_root: base raw dir (e.g., 'data/raw/')
      - model: 'sfincs' (others can be added later)
      - domain_info: loaded YAML for the domain (we use ['domain'][0]['path'])
      - sim_dir: where the run folder is/should be created (same as in main)
      - start_time, end_time: strings like 'YYYY-MM-DDTHH-MM-SSZ'
      - meteo_source, hydrology_source: e.g., 'nwm_ana', 'nwm'
    """

    def __init__(
        self,
        coastal_model: str,
        domain_info: dict,
        sim_dir: str,
        start_time: str,
        end_time: str,
        meteo_source: str,
        hydrology_source: str,
        coastal_water_level_source: str,
        raw_download_dir: str,
        buffer_m: float = 2000.0,
        ngen_dis_netcdf: Optional[str] = None,
        glofs_model: Optional[str] = None,       # e.g., "leofs", "loofs", "lsofs", "lmhofs"
        stofs_region: Optional[str] = None,      # e.g., "conus.east.cwl"
        tpxo_relative_path: Optional[str] = None,
        tpxo_model_control: Optional[str] = None,  # e.g. relative path w.r.t tpxo_predict_exe path, "TPXO/Model_tpxo10_atlas"
        tpxo_env: Optional[dict] = None,
    ):
        self.model = coastal_model.lower()
        self.domain_info = domain_info
        self.sim_dir = sim_dir
        self.start_time_str = start_time
        self.end_time_str = end_time
        self.meteo = meteo_source.lower()
        self.hydro = hydrology_source.lower()
        self.coastal = coastal_water_level_source.lower()
        self.raw_root = os.path.normpath(raw_download_dir)
        self.buffer_m = buffer_m

        self.start_dt = self._parse_time(self.start_time_str)
        self.end_dt = self._parse_time(self.end_time_str)

        # Resolved once by main.py; we rely on this being absolute now
        self.domain_path = self.domain_info["domain"][0]["path"]
        self.ngen_dis_netcdf = ngen_dis_netcdf
        self.tpxo_env = tpxo_env

        '''
        # Coastal products (optional)
        # Try to read from domain_info if present, otherwise fallback to defaults.
        di = self.domain_info if isinstance(self.domain_info, dict) else {}
        self.di_coastal = di.get("coastal", {}) if isinstance(di.get("coastal", {}), dict) else {}

        self.glofs_model = (
            glofs_model
            or di.get("glofs_model")
            or di_coastal.get("glofs_model")
            or "leofs"     # default Great Lakes model
        )
        self.stofs_region = (
            stofs_region
            or di.get("stofs_region")
            or di_coastal.get("stofs_region")
            or "conus.east.cwl"  # default STOFS tile you used
        )
        '''
        
        self.domain_path = self.domain_info["domain"][0]["path"]
        domain_epsg = self.domain_info["domain"][0]["epsg"]

        if domain_epsg is not None:
            self.target_epsg = int(domain_epsg)
        else:
            sfgrid = xr.open_dataset(os.path.join(self.domain_path, "sfincs.nc"))
            self.target_epsg = int(sfgrid.attrs.get("epsg", 32614))  # fallback to 32614
            sfgrid.close()
        print(f"[init] Using target EPSG: {self.target_epsg}")


        self.tpxo_model_control = tpxo_model_control    # e.g. "TPXO/Model_tpxo10_atlas"
        self.tpxo_relative_path = tpxo_relative_path

        os.makedirs(self.sim_dir, exist_ok=True)

    # ----------------- Public orchestrator -----------------
    def process_all(self):
        if self.model != "sfincs":
            print(f"[process] Model '{self.model}' not implemented yet.")
            return
        else:
            # Meteo

            try:
                if self.meteo == "nwm_ana":
                    print("\nProcessing sfincs meteo from nwm_ana")
                    self._process_sfincs_meteo_from_nwm_ana()
                elif self.meteo == "nwm_retro":
                    # TODO: Verify
                    # print("[process][meteo] nwm_retro implementaion needs to be verified.")
                    print("\nProcessing sfincs meteo from nwm_metro")
                    self._process_sfincs_meteo_from_nwm_retro()
                else:
                    print(f"[process][meteo] Unknown source '{self.meteo}', skipping.")
            except Exception as e:
                print(f"ERROR : {str(e)}")
                traceback.print_exc()

            # Discharge
            try:
                if self.hydro == "nwm" and self.meteo == "nwm_ana":
                    print("\nProcessing sfincs dis from nwm_ana")
                    self._process_sfincs_dis_from_nwm_ana()
                elif self.hydro == "nwm" and self.meteo == "nwm_retro":
                    # TODO: Verify implementation
            
                    print("\nProcessing sfincs dis from nwm_retro")
                    self._process_sfincs_dis_from_nwm_retro()

                elif self.hydro == "ngen":
                    # TODO: Verify implementation - not a priority right now
                
                    print("\nProcessing sfincs dis from ngen netcdf")
                    self._process_sfincs_dis_from_ngen_netcdf()
                else:
                    print(f"[process][hydro] Unknown source '{self.hydro}', skipping.")
            except Exception as e:
                print(f"ERROR : {str(e)}")
                traceback.print_exc()

            # Coastal Water Levels
            # TODO: implement all
            try:
                if self.coastal == "stofs":
                    print("\nProcessing sfincs coastal: stofs")
                    self.run_stofs_timeseries_legacy()
                # self._process_coastal_stofs()
                elif self.coastal == "tpxo":
                    # TODO: Verify implementation - not a priority right now
                    print("\nProcessing sfincs coastal: tpxo")
                    self._process_coastal_tpxo()
                elif self.coastal == "glofs":
                    print("\nProcessing sfincs coastal: glofs")
                    self._process_coastal_glofs()
                else:
                    print(f"[process][waterlevels] Unknown source '{self.coastal}', skipping.")
            except Exception as e:
                print(f"ERROR : {str(e)}")
                traceback.print_exc()


    # ----------------- Meteo -----------------
    def _process_sfincs_meteo_from_nwm_ana(self):
        summary = write_sfincs_meteo_from_nwm(
            start_date=self.start_dt,
            end_date=self.end_dt,
            mode="ana",
            domain_nc_path=os.path.join(self.domain_path, "sfincs.nc"),   # <- make this a full path in your environment
            out_dir=self.sim_dir,                               # where to write sfincs.amu/.amv/.ampr/.amp
            raw_root=self.raw_root,                           # root containing the daily NWM folders
            target_epsg=self.target_epsg,
            buffer_m=2000.0,
            flip_vertical=True,
        )

        print("Done:", summary)


    def _process_sfincs_meteo_from_nwm_retro(self):
        """
        Build SFINCS meteo time series from NWM retrospective (LDASIN) FORCING files.
        Expected hourly files under: data/raw/meteo/nwm_retro/
          - YYYYMMDDHH00.LDASIN_DOMAIN1[.nc]  (S3-style or locally normalized)
          - nwm_forcing_YYYYMMDD_HH.nc        (optional normalized alias)

        Behavior is intentionally identical to `_process_sfincs_meteo_from_nwm_ana`
        except for input discovery, so outputs align 1:1 with SFINCS 'meteo_on_equidistant_grid'.
        """

        summary = write_sfincs_meteo_from_nwm(
            start_date=self.start_dt,
            end_date=self.end_dt,
            mode="retro",
            domain_nc_path=os.path.join(self.domain_path, "sfincs.nc"),   # <- make this a full path in your environment
            out_dir=self.sim_dir,                               # where to write sfincs.amu/.amv/.ampr/.amp
            raw_root=self.raw_root,                           # root containing the daily NWM folders
            target_epsg=self.target_epsg,
            buffer_m=2000.0,
            flip_vertical=True,
            decode_packed=True
        )

        print("Done:", summary)


    # ----------------- Hydro -----------------
    def _process_sfincs_dis_from_nwm_ana(self):
        """
        Read hourly NWM channel_rt files and map feature_id -> columns defined in sfincs.src,
        writing an sfincs.dis time series in 'seconds since start_time'.
        """
        # Read feature IDs from the *.src placed in the run folder (already copied)
        # Expect each line like: <x> <y> "<feature_id>" <...>
        src_path = os.path.join(self.sim_dir, "sfincs_nwm.src")  # if you rename to generic 'sfincs.src', adjust here
        print(f"processing {src_path}")
        if not os.path.exists(src_path):
            # fallback for generic name
            alt = os.path.join(self.sim_dir, "sfincs.src")
            src_path = alt if os.path.exists(alt) else src_path
        if not os.path.exists(src_path):
            print("[process][hydro] No sfincs.src/sfincs_nwm.src found in sim_dir; cannot build sfincs.dis")
            return

        with open(src_path, "r", encoding="utf-8") as f:
            # third token is quoted feature_id
            feature_ids = [int(line.split()[2].strip('"')) for line in f if line.strip()]

        num_features = len(feature_ids)

        rows = []
        ref_time = self.start_dt.replace(tzinfo=None)  # seconds since start_time

        for dt in self._iter_hours():
            date_str = dt.strftime("%Y%m%d"); hour_str = f"{dt.hour:02d}"
            path = os.path.join(self.raw_root, "hydro", "nwm", f"nwm_channel_rt_{date_str}_{hour_str}.nc")
            if not os.path.exists(path):
                # skip silently; downloader may have partial hours
                continue
            try:
                ds = xr.open_dataset(path)
                streamflow = ds["streamflow"].values           # shape: [n_feature]

                # Handle fill values
                fv = ds["streamflow"].encoding.get("_FillValue")
                if fv is None:
                    fv = ds["streamflow"].attrs.get("_FillValue")
                if fv is not None:
                    streamflow = np.where(streamflow == fv, np.nan, streamflow)

                feature_id_data = ds["feature_id"].values

                # Prefer dataset 'time' if present; else use dt
                if "time" in ds:
                    ts = pd.to_datetime(ds["time"].values[0]).to_pydatetime()
                else:
                    ts = dt

                # Ensure 'ts' is naive before subtracting naive ref_time
                if getattr(ts, "tzinfo", None) is not None:
                    ts = ts.replace(tzinfo=None)

                seconds_since_ref = int((ts - ref_time).total_seconds())

                # Map feature_id to column index
                id_index_map = {int(fid): i for i, fid in enumerate(feature_id_data)}
                row = [np.nan] * num_features
                for i, fid in enumerate(feature_ids):
                    j = id_index_map.get(fid)
                    if j is not None:
                        row[i] = float(streamflow[j])

                rows.append([seconds_since_ref] + row)
            except Exception as e:
                print(f"[process][hydro] Failed {path}: {e}")
                traceback.print_exc()

        if not rows:
            print("[process][hydro] No rows written to sfincs.dis (no inputs found).")
            return

        out = np.array(rows, dtype=float)
        out_path = os.path.join(self.sim_dir, "sfincs.dis")
        # first col integer seconds, others float
        fmt = ["%.0f"] + ["%.5f"] * num_features
        np.savetxt(out_path, out, fmt=" ".join(fmt))
        print(f"[process][hydro] Wrote {out_path}")


    def _process_sfincs_dis_from_ngen_netcdf(self):
        from .ngen_dis import build_dis_from_ngen_netcdf
        from datetime import datetime
    
        if not self.ngen_dis_netcdf:
            return  # nothing to do

        # choose src file already in the run folder
        src_path = os.path.join(self.sim_dir, "sfincs_nwm.src")
        if not os.path.exists(src_path):
            alt = os.path.join(self.sim_dir, "sfincs.src")
            src_path = alt if os.path.exists(alt) else src_path
        if not os.path.exists(src_path):
            print("[process][hydro:single] No sfincs.src/sfincs_nwm.src found in sim_dir; cannot build sfincs.dis")
            return

        out_path = os.path.join(self.sim_dir, "sfincs.dis")
        num_pts, matched, nrows = build_dis_from_ngen_netcdf(
            sfincs_src=src_path,
            troute_nc=self.ngen_dis_netcdf,
            output_dis=out_path,
            flow_var="flow",
            time_var="time",
            id_var="feature_id",
            fill_missing=0.0,
            # align time column with workflow (seconds since start)
            start_time=self.start_dt,
        )
        print(f"[process][hydro:single] Wrote {out_path} | points={num_pts} matched={matched} rows={nrows}")


    def _process_sfincs_dis_from_nwm_retro(self):
        """
        Build sfincs.dis from NWM retrospective CHRTOUT hourly files.
        Looks for hourly files like:
          <raw_root>/streamflow/nwm_retro/YYYYMMDDHH00.CHRTOUT_DOMAIN1[.nc]
        Falls back to nwm_channel_rt_YYYYMMDD_HH.nc if present.

        Writes: <sim_dir>/sfincs.dis
        """
        # --- Locate source list of feature_ids from the run's *.src file ---
        src_path = os.path.join(self.sim_dir, "sfincs_nwm.src")
        if not os.path.exists(src_path):
            alt = os.path.join(self.sim_dir, "sfincs.src")
            src_path = alt if os.path.exists(alt) else src_path
        if not os.path.exists(src_path):
            print("[process][hydro:retro] No sfincs.src/sfincs_nwm.src; cannot build sfincs.dis")
            return

        # parse "<x> <y> \"<feature_id>\" ..." per line
        feature_ids = self._parse_feature_ids_from_src(src_path)
        num_features = len(feature_ids)
        if num_features == 0:
            print("[process][hydro:retro] No feature_ids parsed from src file.")
            return

        # --- Iterate hours, gather rows ---
        out_rows = []
        ref_time = self.start_dt.replace(tzinfo=None)
        base_dir = os.path.join(self.raw_root, "streamflow", "nwm_retro")

        for dt in self._iter_hours():
            # Preferred retrospective filenames: YYYYMMDDHH00.CHRTOUT_DOMAIN1[.nc]
            stamp = dt.strftime("%Y%m%d%H") + "00"
            cand1 = os.path.join(base_dir, f"{stamp}.CHRTOUT_DOMAIN1.nc")
            cand2 = os.path.join(base_dir, f"{stamp}.CHRTOUT_DOMAIN1")
            # Fallback to old pattern if user mirrored that
            date_str = dt.strftime("%Y%m%d")
            hour_str = f"{dt.hour:02d}"
            cand3 = os.path.join(self.raw_root, "hydro", "nwm_retro", f"nwm_channel_rt_{date_str}_{hour_str}.nc")

            path = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else (cand3 if os.path.exists(cand3) else None))
            if path is None:
                # missing hour; skip
                continue

            try:
                ds = xr.open_dataset(path)

                # --- Read and scale streamflow safely (packed ints possible) ---
                sf_var = ds["streamflow"]
                sf = sf_var.values  # (feature_id,)
                # Apply CF mask/scale if not already decoded as float
                # (xarray usually decodes, but be defensive)
                if np.issubdtype(sf.dtype, np.integer):
                    scale = float(sf_var.attrs.get("scale_factor", 1.0))
                    offs = float(sf_var.attrs.get("add_offset", 0.0))
                    sf = sf.astype("float64") * scale + offs

                # Replace known fill/missing with NaN
                for key in ("_FillValue", "missing_value"):
                    fv = sf_var.encoding.get(key, None)
                    if fv is None:
                        fv = sf_var.attrs.get(key, None)
                    if fv is not None:
                        sf = np.where(sf == fv, np.nan, sf)

                # --- feature_id array from file ---
                file_ids = ds["feature_id"].values
                # Some files store as int64; ensure Python int for dict keys
                file_ids = file_ids.astype("int64", copy=False)
                id_index_map = {int(fid): i for i, fid in enumerate(file_ids)}

                # --- Timestamp for this file ---
                ts = dt  # fallback
                try:
                    tvar = ds["time"]
                    tval = tvar.values[0]
                    if np.issubdtype(getattr(tval, "dtype", type(tval)), np.datetime64):
                        ts = pd.to_datetime(tval).to_pydatetime()
                    else:
                        # e.g., integer minutes since epoch
                        units = tvar.attrs.get("units", "")
                        if units.startswith("minutes since 1970-01-01"):
                            ts = datetime(1970, 1, 1) + timedelta(minutes=int(tval))
                except Exception:
                    pass
                if getattr(ts, "tzinfo", None) is not None:
                    ts = ts.replace(tzinfo=None)

                seconds_since_ref = int((ts - ref_time).total_seconds())

                # --- Assemble row aligned to src feature_id order ---
                row_vals = [np.nan] * num_features
                for col, fid in enumerate(feature_ids):
                    j = id_index_map.get(int(fid))
                    if j is not None:
                        row_vals[col] = float(sf[j])

                out_rows.append([seconds_since_ref] + row_vals)

            except Exception as e:
                print(f"[process][hydro:retro] Failed {path}: {e}")
                traceback.print_exc()

        # --- Write output ---
        if not out_rows:
            print("[process][hydro:retro] No rows written to sfincs.dis")
            return

        out = np.array(out_rows, dtype=float)
        out_path = os.path.join(self.sim_dir, "sfincs.dis")
        fmt = ["%.0f"] + ["%.5f"] * num_features
        np.savetxt(out_path, out, fmt=" ".join(fmt))
        print(f"[process][hydro:retro] Wrote {out_path}")

#---------------------------------------------COASTAL---------------

    def _process_coastal_glofs(self):
        """
        Use the legacy GLOFS → SFINCS timeseries routine to build sfincs.bzs
        from sfincs.bnd for the configured GLOFS model.
        """
        from .glofs_sfincs import build_bzs_from_glofs_legacy

        bnd_path = os.path.join(self.sim_dir, "sfincs.bnd")
        if not os.path.exists(bnd_path):
            print(f"[process][coastal:glofs] boundary file not found: {bnd_path}")
            return

        out_bzs = os.path.join(self.sim_dir, "sfincs.bzs")
        utm_epsg_str = f"EPSG:{int(self.target_epsg)}"
        access_area = self.domain_info["domain"][0].get("access_area", "lake-erie-operational-forecast-system-leofs")
        model = access_area.split("-")[-1]
        if model not in ["leofs", "lmhofs", "lsofs", "loofs"]:
            print(f"Invalid acces area : {access_area}")

        # print(f"[process][coastal:glofs] starting legacy flow: model={self.glofs_model}")
        try:
            written = build_bzs_from_glofs_legacy(
                model=model,
                bnd_file=bnd_path,
                bzs_outfile=out_bzs,
                start_dt=self.start_dt,
                end_dt=self.end_dt,
                time_step_hours=1,
                utm_epsg=utm_epsg_str,
                base_dir=None,
                add_360_longitudes=True,
            )
            print(f"[process][coastal:glofs] wrote {written}")

            '''
            written = build_bzs_from_glofs_legacy(
                model="leofs",
                bnd_file=bnd_path,
                bzs_outfile=out_bzs,
                start_dt=self.start_dt,
                end_dt=self.end_dt,
                time_step_hours=1,
                utm_epsg=utm_epsg_str,
                base_dir=None,
                add_360_longitudes=True,,
                downloads_dir: Optional[str] = None,
                extra_search_dirs: Optional[List[str]] = None,
                base_dir: Optional[str] = None,  # ignored; kept for legacy signature
            )
            '''
        except Exception as e:
            print(f"[process][coastal:glofs] failed: {e}")
            traceback.print_exc()


    def _process_coastal_stofs(self):
        """
        Use the legacy STOFS → SFINCS timeseries routine to build sfincs.bzs
        from sfincs.bnd + GRIB2 forcing for the configured period.
        """

        try:
            bnd = os.path.join(self.domain_path, "sfincs.bnd")
            if not os.path.exists(bnd):
                print(f"[process][coastal:stofs] boundary file not found: {bnd_path}")
                return
            grib = os.path.join(self.raw_root, "coastal", "stofs", f"stofs_2d_glo_{self.start_dt:%Y%m%d}_00.grib2")
            out  = os.path.join(self.sim_dir, "sfincs.bzs")
            self.run_stofs_timeseries_legacy(bnd_file=bnd, grib_file=grib, bzs_output=out, utm_crs_epsg=self.target_epsg, variable_name="unknown")
        except Exception as e:
            print(f"[process][coastal:stofs] failed: {e}")
            traceback.print_exc()


    def _process_coastal_tpxo(self):
        """
        Build TPXO forcing for SFINCS using only *relative* paths inside setup.tpxo_sfincs.
        All intermediate files go under coastalforcing/process_data/TPXO/.
        Final sfincs.bzs is written into the current run folder (self.sim_dir).
        """

        from process_data.tpxo_sfincs import run_tpxo_pipeline_for_sfincs

        try:
            result = run_tpxo_pipeline_for_sfincs(
                sfincs_bnd_file=os.path.join(self.sim_dir, "sfincs.bnd"),
                start_time=self.start_dt,
                end_time=self.end_dt,
                utm_epsg=self.target_epsg,

                # Everything below stays short & relative (no long absolute strings):
                predict_tide_exe_rel=os.path.join(self.tpxo_relative_path, "predict_tide"),
                model_control_rel=os.path.join(self.tpxo_relative_path, self.tpxo_model_control),
                tp_dir_rel=self.tpxo_relative_path,
                lat_lon_time_rel=os.path.join(self.tpxo_relative_path, "tpxo_lat_lon_time"),
                tpxo_out_rel=os.path.join(self.tpxo_relative_path, "tpxo_out.txt"),
                setup_rel=os.path.join(self.tpxo_relative_path, "setup.tpxo_sfincs"),
                out_bzs_path=os.path.join(self.sim_dir, "sfincs.bzs"),
                step_seconds=600,
                prepend_header_block=True,
                header_model_label="tpxo10_atlas",
                header_constituents=None,   # or a list like ["m2","s2",...]
                env_extra=getattr(self, "tpxo_env", None),  # e.g., LD_LIBRARY_PATH if needed
            )
            print(f"[process][coastal:tpxo] done → {result['sfincs_bzs']}")
        except Exception as e:
            import traceback
            print(f"[process][coastal:tpxo] failed: {e}")
            traceback.print_exc()

    

    # ----------------- Utilities -----------------

    def _parse_feature_ids_from_src(self, src_path: str) -> list[int]:
        """
        Parse SFINCS src-like file where the 3rd token is a quoted feature_id,
        e.g., line:  x  y  "2430123"
        Returns a list of ints in file order.
        """
        ids = []
        with open(src_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) >= 3:
                    try:
                        ids.append(int(parts[2].strip('"')))
                    except Exception:
                        pass
        if not ids:
            print(f"[parse] No feature_ids parsed from {src_path}")
        return ids

    def _iter_hours(self) -> Iterable[datetime]:
        current = self.start_dt
        while current < self.end_dt:
            yield current
            current += timedelta(hours=1)

    @staticmethod
    def _parse_time(s: str) -> datetime:
        fmts = ["%Y-%m-%dT%H-%M-%SZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"]
        for fmt in fmts:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                pass
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


    import pandas as pd
    import xarray as xr
    import pyproj
    import numpy as np
    from typing import Union
    from scipy.interpolate import griddata, interp1d
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation  # only used if you uncomment save


    def run_stofs_timeseries_legacy(self):
        """
        Build coastal water-level time series from STOFS GRIB2:
          - read SFINCS boundary points (UTM of the domain)
          - convert boundary X/Y to STOFS Lambert Conformal coords
          - for each available STOFS cycle file in the time window (t00z/t06z/t12z/t18z),
            read the hourly field, sample at boundary points, and append to a raw .bzs
          - resample to 10-minute cadence (cubic) → final sfincs.bzs

        Assumptions:
          - STOFS files were downloaded to data/raw/coastal/stofs with original filenames kept:
              'stofs_2d_glo.t{HH}z.conus.east.cwl.grib2'
          - Variables present might be 'cwl', 'slev', 'zeta', or 'unknown'
          - Grid is regular in Lambert Conformal with fixed dx,dy and a known origin used below
        """
        try:
            import xarray as xr
            import numpy as np
            import pandas as pd
            import pyproj
            from scipy.interpolate import RegularGridInterpolator, interp1d
        except Exception as e:
            print(f"[process][coastal:stofs] missing dep: {e}")
            return

        # ---------- locate boundary file ----------
        # Prefer run folder; fall back to domain folder.
        cand = [
            os.path.join(self.sim_dir, "sfincs.bnd"),
            os.path.join(self.domain_path, "SFINCS_V1", "sfincs.bnd"),
            os.path.join(self.domain_path, "sfincs.bnd"),
        ]
        bnd_file = next((p for p in cand if os.path.exists(p)), None)
        if not bnd_file:
            print("[process][coastal:stofs] no sfincs.bnd found")
            return

        # ---------- read boundary points (UTM) ----------
        try:
            bnd_df = pd.read_csv(bnd_file, sep=r"\s+", header=None, names=["x", "y"], engine="python")
        except Exception as e:
            print(f"[process][coastal:stofs] failed reading {bnd_file}: {e}")
            traceback.print_exc()
            return

        if bnd_df.empty:
            print(f"[process][coastal:stofs] empty boundary file: {bnd_file}")
            return

        # ---------- discover STOFS files from window ----------
        outdir = os.path.join(self.raw_root, "coastal", "stofs")
        wanted = []
        seen = set()
        for dt in self._iter_hours():
            if dt.hour not in (0, 6, 12, 18):
                continue
            fname = f"stofs_2d_glo.t{dt.hour:02d}z.conus.east.cwl.grib2"
            fpath = os.path.join(outdir, fname)
            if os.path.exists(fpath) and fpath not in seen:
                wanted.append(fpath)
                seen.add(fpath)

        if not wanted:
            print(f"[process][coastal:stofs] no STOFS files found in {outdir} for the window")
            return

        print("[process][coastal:stofs] using {} file(s):".format(len(wanted)))
        for w in wanted:
            print("    " + os.path.basename(w))

        # ---------- define CRS and grid geometry ----------
        # SFINCS boundary CRS (UTM) and STOFS Lambert Conformal CRS
        utm = pyproj.CRS(f"EPSG:{self.target_epsg}")
        lcc = pyproj.CRS.from_proj4(
            "+proj=lcc +lat_1=25 +lat_2=25 +lat_0=25 +lon_0=265 +x_0=0 +y_0=0 +R=6371200 +units=m +no_defs"
        )
        to_lcc = pyproj.Transformer.from_crs(utm, lcc, always_xy=True)

        # STOFS grid spacing and a consistent Lambert origin used to reconstruct X/Y axes
        dx = 2539.703
        dy = 2539.703
        # Origin reference given in (lon0, lat0) – we only need its *Lambert* coordinates
        # so compute once via WGS84→LCC.
        wgs84 = pyproj.CRS("EPSG:4326")
        wgs_to_lcc = pyproj.Transformer.from_crs(wgs84, lcc, always_xy=True)
        lon0, lat0 = 238.445999, 20.191999
        x0, y0 = wgs_to_lcc.transform(lon0, lat0)

        # Convert boundary points to Lambert coordinates
        xb, yb = to_lcc.transform(bnd_df["x"].values, bnd_df["y"].values)
        sample_pts = np.column_stack([yb, xb])  # (y, x) order for RegularGridInterpolator

        # Containers for raw samples and matching seconds since start
        all_rows = []
        all_times = []

        start_naive = self.start_dt.replace(tzinfo=None)

        # ---------- iterate STOFS files ----------
        for path in wanted:
            fname = os.path.basename(path)
            try:
                # Open only 'surface' level messages; cfgrib will expose 'unknown' or 'cwl'/'slev'/'zeta'
                ds = xr.open_dataset(
                    path,
                    engine="cfgrib",
                    backend_kwargs={"indexpath": "", "filter_by_keys": {"typeOfLevel": "surface"}},
                )
            except Exception as e:
                print(f"[process][coastal:stofs] failed to open default: {e}")
                continue

            # pick variable name
            varname = None
            for cand in ("cwl", "slev", "zeta", "unknown"):
                if cand in ds.data_vars:
                    varname = cand
                    break
            if varname is None:
                print(f"[process][coastal:stofs] {fname}: no suitable variable in {list(ds.data_vars)}")
                continue

            data = ds[varname]
            dims = tuple(data.dims)
            shape = tuple(data.shape)
            print(f"[process][coastal:stofs] {fname}: var='{varname}', dims={dims}, shape={shape}")

            # Expect (step, y, x)
            if len(shape) != 3 or dims[1] != "y" or dims[2] != "x":
                print(f"[process][coastal:stofs] {fname}: unexpected dims {dims}; skipping")
                continue

            nframes, ny, nx = shape

            # Build LCC grid axes (meters) so we can use RegularGridInterpolator
            x_axis = x0 + np.arange(nx) * dx
            y_axis = y0 + np.arange(ny) * dy

            # ---- robust time handling (handles scalar 'time', 1-D 'valid_time', and 'step') ----
            try:
                if ("valid_time" in ds) and (ds["valid_time"].size == nframes):
                    vt = np.asarray(ds["valid_time"].values)  # datetime64
                    print(f"[debug] using valid_time: shape={vt.shape}, dtype={vt.dtype}")
                    vt_sec = vt.astype("datetime64[s]").astype("int64")
                    start_sec = int(pd.Timestamp(start_naive).value // 10**9)
                    sec_since_start = (vt_sec - start_sec).astype(int)

                else:
                    tvals = np.asarray(ds["time"].values) if "time" in ds else None
                    step_vals = np.asarray(ds["step"].values) if "step" in ds else None
                    print(f"[debug] time: shape={None if tvals is None else tvals.shape}, "
                          f"ndim={None if tvals is None else tvals.ndim}, dtype={None if tvals is None else tvals.dtype}")
                    print(f"[debug] step: shape={None if step_vals is None else step_vals.shape}, "
                          f"ndim={None if step_vals is None else step_vals.ndim}, dtype={None if step_vals is None else step_vals.dtype}")

                    if (tvals is not None) and (step_vals is not None) and (step_vals.size == nframes):
                        # scalar time vs vector time
                        if tvals.ndim == 0:
                            base_dt = pd.to_datetime(tvals).to_pydatetime().replace(tzinfo=None)
                        else:
                            base_dt = pd.to_datetime(tvals[0]).to_pydatetime().replace(tzinfo=None)

                        if np.issubdtype(step_vals.dtype, np.timedelta64):
                            steps_sec = (step_vals / np.timedelta64(1, "s")).astype(int)
                        else:
                            steps_sec = step_vals.astype(int)

                        base_sec = int(pd.Timestamp(base_dt).value // 10**9)
                        start_sec = int(pd.Timestamp(start_naive).value // 10**9)
                        sec_since_start = (base_sec + steps_sec - start_sec).astype(int)
                        print(f"[debug] sec_since_start from time+step: len={sec_since_start.shape[0]}")
                    elif step_vals is not None:
                        if np.issubdtype(step_vals.dtype, np.timedelta64):
                            sec_since_start = (step_vals / np.timedelta64(1, "s")).astype(int)
                        else:
                            sec_since_start = step_vals.astype(int)
                        print(f"[debug] sec_since_start from step only: len={sec_since_start.shape[0]}")
                    else:
                        sec_since_start = np.arange(nframes, dtype=int) * 3600
                        print("[debug] sec_since_start fallback hourly sequence used")

                if sec_since_start.shape[0] != nframes:
                    print(f"[warn] time vector length {sec_since_start.shape[0]} != nframes {nframes}; using hourly fallback")
                    sec_since_start = np.arange(nframes, dtype=int) * 3600

            except Exception as e:
                print(f"[process][coastal:stofs] time decode failed for {fname}: {e}")
                sec_since_start = np.arange(nframes, dtype=int) * 3600

            # ---- sample each frame at boundary points using RegularGridInterpolator ----
            try:
                for k in range(nframes):
                    frame = np.asarray(data.isel(step=k).values)
                    # Build interpolator over (y, x)
                    rgi = RegularGridInterpolator(
                        (y_axis, x_axis),
                        frame,
                        bounds_error=False,
                        fill_value=np.nan,
                    )
                    vals = rgi(sample_pts)  # shape: (n_points,)
                    all_rows.append(vals.astype(float))
                all_times.extend(sec_since_start.tolist())
                print(f"[process][coastal:stofs] {fname}: sampled {nframes} frames at {sample_pts.shape[0]} points")
            except Exception as e:
                print(f"[process][coastal:stofs] sampling failed for {fname}: {e}")

            # be nice to memory
            try:
                ds.close()
            except Exception:
                pass

        # ---------- write raw and 10-min interpolated .bzs ----------
        if not all_rows or not all_times:
            print("[process][coastal:stofs] no samples generated")
            return

        times_sec = np.asarray(all_times, dtype=int)
        order = np.argsort(times_sec)
        times_sec = times_sec[order]
        values = np.asarray(all_rows, dtype=float)[order, :]  # (n_time, n_points)

        # collapse duplicate times (keep last occurrence)
        uniq_t, idx = np.unique(times_sec, return_index=True)
        times_sec = times_sec[idx]
        values = values[idx, :]


        raw_path = os.path.join(self.sim_dir, "sfincs_raw.bzs")
        with open(raw_path, "w") as f:
            for t, row in zip(times_sec, values):
                # add +0.25 to all values
                temp=row
                # row = row + 1.0
                print(f"{temp} : {row}")
                line = f"{int(t)} " + " ".join(f"{v:.4f}" if np.isfinite(v) else "0.0000" for v in row)
                f.write(line + "\n")
        print(f"[process][coastal:stofs] wrote {raw_path} ({values.shape[0]} rows, {values.shape[1]} points)")


        '''
        raw_path = os.path.join(self.sim_dir, "sfincs_raw.bzs")
        with open(raw_path, "w") as f:
            for t, row in zip(times_sec, values):
                line = f"{int(t)} " + " ".join(f"{v:.4f}" if np.isfinite(v) else "0.0000" for v in row)
                f.write(line + "\n")
        print(f"[process][coastal:stofs] wrote {raw_path} ({values.shape[0]} rows, {values.shape[1]} points)")
        '''

        # Interpolate to 10-minute cadence
        t0, t1 = int(times_sec[0]), int(times_sec[-1])
        new_t = np.arange(t0, t1 + 1, 600, dtype=int)

        interp_mat = np.zeros((new_t.size, values.shape[1]), dtype=float)
        for j in range(values.shape[1]):
            col = values[:, j]
            mask = np.isfinite(col)
            if mask.sum() >= 2:
                f = interp1d(times_sec[mask], col[mask], kind="cubic", fill_value="extrapolate", assume_sorted=True)
                interp_mat[:, j] = f(new_t)
            else:
                interp_mat[:, j] = 0.0

        final_path = os.path.join(self.sim_dir, "sfincs.bzs")
        with open(final_path, "w") as f:
            for t, row in zip(new_t, interp_mat):
                # add +0.25 to all values
                temp=row
                # row = row + 1.0
                print(f"{temp} : {row}")
                line = f"{int(t)} " + " ".join(f"{v:.4f}" if np.isfinite(v) else "0.0000" for v in row)
                f.write(line + "\n")
        print(f"[process][coastal:stofs] wrote {final_path} ({interp_mat.shape[0]} rows @10-min)")

        '''
        final_path = os.path.join(self.sim_dir, "sfincs.bzs")
        with open(final_path, "w") as f:
            for t, row in zip(new_t, interp_mat):
                line = f"{int(t)} " + " ".join(f"{v:.4f}" if np.isfinite(v) else "0.0000" for v in row)
                f.write(line + "\n")
        print(f"[process][coastal:stofs] wrote {final_path} ({interp_mat.shape[0]} rows @10-min)")
        '''


    def run_tpxo_timeseries_for_sfincs(
        self,
        *,
        raw_dt_seconds: int = 3600,
        out_dt_seconds: int = 600,
        bnd_filename: str = "sfincs.bnd",
        raw_bzs_name: str = "sfincs_raw.bzs",
        bzs_name: str = "sfincs.bzs",
        tpxo_predict_cmd: str = None,   # optional shell string
        predict_exe: str = None,        # path to predict_tide binary (overrides env)
        model_dir: str = None,          # path to OTPS/TPXO model folder (overrides env)
        utm_epsg=None,                  # <-- NEW: e.g., 32614 or "EPSG:32614"; overrides self.target_epsg
        verbose: bool = True,
    ):
        """
        Build TPXO inputs from SFINCS boundary, call external predictor, and write .bzs files.

        CRS:
          - If utm_epsg is provided (e.g., 32614 or "EPSG:32614"), boundary coords are interpreted in that CRS.
          - Else falls back to self.target_epsg from the SFINCS grid.

        External predictor:
          - Provide either:
              * tpxo_predict_cmd (full shell string), OR
              * predict_exe + model_dir, OR
              * env vars TPXO_PREDICT_EXE and TPXO_MODEL_DIR
        """
        import os
        import subprocess
        from datetime import timedelta, datetime
        import numpy as np
        import pandas as pd
        import pyproj

        # --- Resolve source CRS (boundary file CRS) ---
        def _to_epsg_int(val):
            if val is None:
                return None
            if isinstance(val, int):
                return val
            s = str(val).strip()
            if s.upper().startswith("EPSG:"):
                s = s.split(":")[1]
            try:
                return int(s)
            except Exception:
                return None

        src_epsg = _to_epsg_int(utm_epsg) or int(self.target_epsg)
        dst_epsg = 4326  # WGS84 for TPXO
        if verbose:
            print(f"[tpxo] Using EPSG:{src_epsg} → WGS84 for TPXO")

        # --- Paths ---
        bnd_path = os.path.join(self.sim_dir, bnd_filename)
        stations_file = os.path.join(self.sim_dir, "tpxo_stations.txt")
        times_file = os.path.join(self.sim_dir, "tpxo_times.txt")
        tpxo_output_path = os.path.join(self.sim_dir, "tpxo_out.txt")
        raw_bzs_path = os.path.join(self.sim_dir, raw_bzs_name)
        bzs_path = os.path.join(self.sim_dir, bzs_name)

        if not os.path.exists(bnd_path):
            print(f"[tpxo] boundary not found: {bnd_path}")
            return

        # --- Read boundary (x y ["..."]) ---
        xs, ys = [], []
        with open(bnd_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) >= 2:
                    try:
                        xs.append(float(parts[0])); ys.append(float(parts[1]))
                    except Exception:
                        pass
        if not xs:
            print(f"[tpxo] no valid points in {bnd_path}")
            return

        # --- Transform to lon/lat ---
        src = pyproj.CRS.from_epsg(src_epsg)
        dst = pyproj.CRS.from_epsg(dst_epsg)
        trf = pyproj.Transformer.from_crs(src, dst, always_xy=True)
        lons, lats = trf.transform(xs, ys)

        # --- Write stations file (lon lat per line) ---
        with open(stations_file, "w", encoding="utf-8") as f:
            for lo, la in zip(lons, lats):
                f.write(f"{lo:.6f} {la:.6f}\n")
        if verbose:
            print(f"[tpxo] Wrote stations → {stations_file}")

        # --- Time list (raw_dt_seconds) ---
        start_naive = self.start_dt.replace(tzinfo=None)
        end_naive = self.end_dt.replace(tzinfo=None)
        if end_naive <= start_naive:
            print("[tpxo] empty time range; nothing to do")
            return

        times = []
        step = timedelta(seconds=int(raw_dt_seconds))
        t = start_naive
        while t <= end_naive:
            times.append(t)
            t += step

        with open(times_file, "w", encoding="utf-8") as f:
            for t in times:
                f.write(t.strftime("%Y/%m/%d %H:%M:%S") + "\n")
        if verbose:
            print(f"[tpxo] Wrote {len(times)} times → {times_file}")

        # --- Run external predictor ---
        if tpxo_predict_cmd:
            if verbose:
                print("[tpxo] Running external predictor (shell string):")
                print(f"    {tpxo_predict_cmd}")
                print(f"    → {tpxo_output_path}")
            try:
                with open(tpxo_output_path, "w", encoding="utf-8") as fout:
                    subprocess.run(tpxo_predict_cmd, shell=True, check=True,
                                   stdout=fout, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print(f"[tpxo] ERROR running predictor: {e}")
                return
        else:
            exe = predict_exe or os.environ.get("TPXO_PREDICT_EXE")
            mdir = model_dir or os.environ.get("TPXO_MODEL_DIR")
            if not exe or not os.path.exists(exe):
                print("[tpxo] predictor exe not found. Set predict_exe or TPXO_PREDICT_EXE.")
                return
            if not mdir or not os.path.isdir(mdir):
                print("[tpxo] model dir not found. Set model_dir or TPXO_MODEL_DIR.")
                return
            argv = [exe, "-z", "-m", mdir, "-l", stations_file, "-t", times_file]
            if verbose:
                print("[tpxo] Running external predictor:")
                print("    " + " ".join(argv))
                print(f"    → writing to {tpxo_output_path}")
            try:
                with open(tpxo_output_path, "w", encoding="utf-8") as fout:
                    subprocess.run(argv, check=True, stdout=fout, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print(f"[tpxo] ERROR running predictor: {e}")
                return

        # --- Parse predictor output robustly ---
        def _try_parse_dt(s: str):
            for fmt in ("%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"):
                try:
                    return datetime.strptime(s, fmt)
                except Exception:
                    pass
            return None

        rows = []
        with open(tpxo_output_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                dt = None
                vstart = 0
                if len(parts) >= 3:
                    dt = _try_parse_dt(parts[0] + " " + parts[1])
                    if dt is not None:
                        vstart = 2
                if dt is None and len(parts) >= 1:
                    dt = _try_parse_dt(parts[0].replace("_", " "))
                    if dt is not None:
                        vstart = 1
                if dt is None:
                    continue
                try:
                    vals = [float(x) for x in parts[vstart:]]
                except Exception:
                    continue
                rows.append((dt, vals))

        if not rows:
            print(f"[tpxo] no data parsed from {tpxo_output_path}")
            return

        rows.sort(key=lambda r: r[0])
        npts = len(rows[0][1])
        ts = np.array([r[0] for r in rows])
        secs = np.array([(t - start_naive).total_seconds() for t in ts], dtype=int)
        data = np.array([r[1] for r in rows], dtype=float)  # (nt, npts)

        # --- Write raw hourly bzs ---

        with open(raw_bzs_path, "w", encoding="utf-8") as f:
            for i, sec in enumerate(secs):
                row = data[i] + 0.25     # <-- add offset
                print(f"{data[i]} : {row}")
                f.write(str(int(sec)))
                f.write(" ")
                f.write(" ".join(f"{v:.4f}" for v in row))
                f.write("\n")
        if verbose:
            print(f"[tpxo] wrote raw → {raw_bzs_path} (nt={len(secs)}, npts={npts})")

        '''
        with open(raw_bzs_path, "w", encoding="utf-8") as f:
            for i, sec in enumerate(secs):
                f.write(str(int(sec)))
                f.write(" ")
                f.write(" ".join(f"{v:.4f}" for v in data[i]))
                f.write("\n")
        if verbose:
            print(f"[tpxo] wrote raw → {raw_bzs_path} (nt={len(secs)}, npts={npts})")
        '''

        # --- Interpolate to uniform out_dt_seconds ---
        tmin, tmax = int(secs[0]), int(secs[-1])
        tgt = np.arange(tmin, tmax + 1, int(out_dt_seconds), dtype=int)
        out = np.empty((tgt.size, npts), dtype=float)
        for j in range(npts):
            y = data[:, j]
            mask = np.isfinite(y)
            if mask.sum() >= 2:
                out[:, j] = np.interp(tgt, secs[mask], y[mask])
            elif mask.sum() == 1:
                out[:, j] = y[mask][0]
            else:
                out[:, j] = 0.0

        with open(bzs_path, "w", encoding="utf-8") as f:
            for i, sec in enumerate(tgt):
                row = out[i] + 0.25      # <-- add offset
                print(f"{out[i]} : {row}")
                f.write(str(int(sec)))
                f.write(" ")
                f.write(" ".join(f"{v:.4f}" for v in row))
                f.write("\n")
        if verbose:
            print(f"[tpxo] wrote bzs  → {bzs_path} (nt={tgt.size}, npts={npts}, dt={out_dt_seconds}s)")
        '''    
        with open(bzs_path, "w", encoding="utf-8") as f:
            for i, sec in enumerate(tgt):
                f.write(str(int(sec)))
                f.write(" ")
                f.write(" ".join(f"{v:.4f}" for v in out[i]))
                f.write("\n")
        if verbose:
            print(f"[tpxo] wrote bzs  → {bzs_path} (nt={tgt.size}, npts={npts}, dt={out_dt_seconds}s)")
        '''

