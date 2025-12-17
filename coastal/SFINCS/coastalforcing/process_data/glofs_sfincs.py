# coastalforcing/process_data/glofs_sfincs.py
"""
GLOFS → SFINCS .bzs generator (LOCAL → OPeNDAP → on-demand download)

Flow per required NetCDF (sample + each hourly):
  1) Look for pre-downloaded files on disk (both new and NOS legacy names).
  2) If not found, try opening remote OPeNDAP dodsC URL directly (no local write).
  3) If that fails, call download_glofs_range() just for that hour, then open locally.

Notes
- We always check BOTH filename patterns when searching locally:
    A) {model}.t{HH}z.{YYYYMMDD}.fields.n{xxx}.nc
    B) nos.{model}.fields.n{xxx}.{YYYYMMDD}.t{HH}z.nc
- For OPeNDAP we use only the A-pattern URL (as requested).
- add_360_longitudes=True will wrap negative lons into [0, 360], like your legacy.
"""

from __future__ import annotations

import os
import sys
import math
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr
import matplotlib.tri as tri
from pyproj import Transformer
from scipy.interpolate import interp1d

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _parse_dt_utc_any(s) -> datetime:
    if isinstance(s, datetime):
        return s.replace(tzinfo=None)
    s = str(s).strip()
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s.replace("T", " ")).replace(tzinfo=None)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
                "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    # last resort
    import pandas as pd
    return pd.to_datetime(s).to_pydatetime().replace(tzinfo=None)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_open_local_nc(path: str) -> xr.Dataset:
    """Open a LOCAL NetCDF with a tolerant decode."""
    drop_vars = ["siglay", "siglev", "siglay_center", "siglev_center"]
    try:
        return xr.open_dataset(path, engine="netcdf4", decode_times=True, drop_variables=drop_vars)
    except Exception as e:
        print(f"[GLOFS] retrying without time decoding for {os.path.basename(path)} due to: {e}")
        return xr.open_dataset(path, engine="netcdf4", decode_times=False, drop_variables=drop_vars)


def _open_opendap(url: str) -> xr.Dataset:
    """Open a REMOTE OPeNDAP URL (no local write). Single attempt."""
    print(f"[GLOFS] OPeNDAP open: {url}")
    drop_vars = ["siglay", "siglev", "siglay_center", "siglev_center"]
    # We keep decode_times=True first; if it fails, try without time decoding.
    try:
        return xr.open_dataset(url, engine="netcdf4", decode_times=True, drop_variables=drop_vars)
    except Exception as e:
        print(f"[GLOFS] OPeNDAP decode_times=True failed: {e}; retry decode_times=False")
        return xr.open_dataset(url, engine="netcdf4", decode_times=False, drop_variables=drop_vars)


# -----------------------------------------------------------------------------
# Filename helpers
# -----------------------------------------------------------------------------

def _cycle_suffix_for(dt: datetime) -> Tuple[str, str]:
    """Return (cycle_str, n_suffix) like ('t00z', 'n000') for a specific hour."""
    h = dt.hour
    cycle_hour = (h // 6) * 6
    return f"t{cycle_hour:02d}z", f"n{h % 6:03d}"


def _local_basenames_for(model: str, dt: datetime) -> List[str]:
    """
    Return both name styles for local search, regardless of date cutoffs:
      A) model.t??z.YYYYMMDD.fields.nxxx.nc
      B) nos.model.fields.nxxx.YYYYMMDD.t??z.nc
    """
    datestr = dt.strftime("%Y%m%d")
    cycle, n = _cycle_suffix_for(dt)
    # A (new)
    a = f"{model}.{cycle}.{datestr}.fields.{n}.nc"
    # B (legacy NOS)
    b = f"nos.{model}.fields.{n}.{datestr}.{cycle}.nc"
    return [a, b]


def _find_local_file(search_dirs: List[str], basenames: List[str]) -> Optional[str]:
    """Return the first existing non-empty path among search_dirs × basenames."""
    for d in search_dirs:
        if not d:
            continue
        for name in basenames:
            cand = os.path.join(d, name)
            if os.path.exists(cand) and os.path.getsize(cand) > 0:
                print(f"[GLOFS] local hit: {cand}")
                return cand
    return None


def _opendap_hour_url(model: str, dt: datetime) -> str:
    """Requested OPeNDAP URL pattern (A-style) for a specific hour."""
    yyyy = f"{dt.year:04d}"
    mm = dt.strftime("%m")
    datestr = dt.strftime("%Y%m%d")
    cycle, n = _cycle_suffix_for(dt)
    return (
        f"https://www.ncei.noaa.gov/thredds/dodsC/model-{model}/{yyyy}/{mm}/"
        f"{model}.{cycle}.{datestr}.fields.{n}.nc"
    )


def _opendap_sample_url(model: str, dt: datetime) -> str:
    """Use the t00z n000 file of the start day for mesh (matches your legacy sample)."""
    yyyy = f"{dt.year:04d}"
    mm = dt.strftime("%m")
    datestr = dt.strftime("%Y%m%d")
    return (
        f"https://www.ncei.noaa.gov/thredds/dodsC/model-{model}/{yyyy}/{mm}/"
        f"{model}.t00z.{datestr}.fields.n000.nc"
    )


# -----------------------------------------------------------------------------
# Mesh + interpolation
# -----------------------------------------------------------------------------

def _build_triangulation(ds: xr.Dataset) -> Tuple[tri.Triangulation, np.ndarray, np.ndarray]:
    lon = ds["lon"].values
    lat = ds["lat"].values
    nv = ds["nv"].values.T - 1  # 1-based → 0-based
    triang = tri.Triangulation(lon, lat, nv)
    return triang, lon, lat


# -----------------------------------------------------------------------------
# Main extractor
# -----------------------------------------------------------------------------

def extract_glofs_timeseries(
    model: str,
    bnd_file: str,
    bzs_outfile: str,
    start_time: datetime,
    end_time: datetime,
    time_step: timedelta,
    utm_epsg: str,
    add_360_longitudes: bool = True,
    downloads_dir: Optional[str] = None,
    extra_search_dirs: Optional[List[str]] = None,
) -> None:
    """
    Generate SFINCS .bzs using local files if present; else OPeNDAP; else on-demand download.
    Now matches STOFS behavior: writes a raw hourly *and* a 10-minute-resampled final file.
    """
    # Where to look for already-downloaded files
    if downloads_dir is None:
        downloads_dir = os.path.join(os.path.dirname(os.path.abspath(bzs_outfile)), "glofs_nc")
    _ensure_dir(downloads_dir)

    # prepend our run folder first, then any extras (e.g., global cache)
    search_dirs: List[str] = [downloads_dir] + list(filter(None, extra_search_dirs or []))

    # Import the downloader
    try:
        from download_data.glofs_downloader import download_glofs_range
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, ".."))
        sys.path.insert(0, root)
        from download_data.glofs_downloader import download_glofs_range

    # Read SFINCS boundary in UTM
    coords: List[Tuple[float, float]] = []
    with open(bnd_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            if len(p) >= 2:
                coords.append((float(p[0]), float(p[1])))
    if not coords:
        raise RuntimeError(f"No valid boundary points in {bnd_file}")

    # Transform to lon/lat
    transformer = Transformer.from_crs(utm_epsg, "EPSG:4326", always_xy=True)
    xs, ys = zip(*coords)
    qlon, qlat = transformer.transform(xs, ys)
    qlon = np.asarray(qlon, float)
    qlat = np.asarray(qlat, float)
    if add_360_longitudes:
        qlon = np.where(qlon < 0, qlon + 360.0, qlon)

    print(f"[GLOFS] boundary points: {len(qlon)}")

    # ---- SAMPLE (mesh) ----
    sample_local = _find_local_file(search_dirs, _local_basenames_for(model, start_time.replace(hour=0)))
    ds0: Optional[xr.Dataset] = None

    if sample_local:
        ds0 = _safe_open_local_nc(sample_local)
    else:
        sample_url = _opendap_sample_url(model, start_time)
        try:
            ds0 = _open_opendap(sample_url)
        except Exception as e:
            print(f"[GLOFS] OPeNDAP sample failed: {e}")

        if ds0 is None:
            try:
                download_glofs_range(
                    start=start_time.replace(minute=0, second=0, microsecond=0),
                    end=start_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1),
                    step=timedelta(hours=1),
                    outdir=downloads_dir,
                    allow_cached=True,
                )
                sample_local = _find_local_file(search_dirs, _local_basenames_for(model, start_time.replace(hour=0)))
                if not sample_local:
                    raise RuntimeError("download_glofs_range finished but sample file still not found.")
                ds0 = _safe_open_local_nc(sample_local)
            except Exception as e:
                raise RuntimeError(f"Could not obtain a sample GLOFS file for mesh triangulation: {e}")

    triang, lon1d, lat1d = _build_triangulation(ds0)
    finder = triang.get_trifinder()
    tri_idx = finder(qlon, qlat)
    print(f"[GLOFS] mesh lon/lat bbox: "
          f"lon[{float(np.nanmin(lon1d)):.3f},{float(np.nanmax(lon1d)):.3f}], "
          f"lat[{float(np.nanmin(lat1d)):.3f},{float(np.nanmax(lat1d)):.3f}]")

    interp_data: List[Optional[Tuple[np.ndarray, List[float]]]] = []
    for i, t_idx in enumerate(tri_idx):
        if t_idx == -1:
            interp_data.append(None)
            continue
        nodes = triang.triangles[t_idx]
        x = triang.x[nodes]
        y = triang.y[nodes]
        A = np.array([[x[1] - x[0], x[2] - x[0]],
                      [y[1] - y[0], y[2] - y[0]]], dtype=float)
        b = np.array([qlon[i] - x[0], qlat[i] - y[0]], dtype=float)
        try:
            l1, l2 = np.linalg.solve(A, b)
            w0 = 1.0 - l1 - l2
            interp_data.append((nodes, [w0, l1, l2]))
        except np.linalg.LinAlgError:
            interp_data.append(None)

    node_list = sorted({n for it in interp_data if it for n in it[0]})
    if all(it is None for it in interp_data):
        print(f"[GLOFS][warn] all boundary points fall outside mesh; "
              f"values will be NaN. Check utm_epsg and add_360_longitudes.")

    # ---- TIME LOOP (collect; don't write yet) ----
    times_sec: List[int] = []
    rows_vals: List[List[float]] = []

    t = start_time
    while t <= end_time:
        basenames = _local_basenames_for(model, t)

        # 1) local?
        local_nc = _find_local_file(search_dirs, basenames)
        ds_hour: Optional[xr.Dataset] = None

        if local_nc:
            ds_hour = _safe_open_local_nc(local_nc)
        else:
            # 2) OPeNDAP (A-pattern)
            url = _opendap_hour_url(model, t)
            try:
                ds_hour = _open_opendap(url)
            except Exception as e:
                print(f"[GLOFS] OPeNDAP failed for {t}: {e}")

            if ds_hour is None:
                # 3) on-demand download
                try:
                    download_glofs_range(
                        start=t.replace(minute=0, second=0, microsecond=0),
                        end=(t + time_step).replace(minute=0, second=0, microsecond=0),
                        step=time_step,
                        outdir=downloads_dir,
                        allow_cached=True,
                    )
                    local_nc = _find_local_file(search_dirs, basenames)
                    if not local_nc:
                        raise RuntimeError("download_glofs_range finished but file still not found.")
                    ds_hour = _safe_open_local_nc(local_nc)
                except Exception as e:
                    print(f"[GLOFS] no data for {t} after on-demand download: {e}")

        sec = int((t - start_time).total_seconds())
        vals_for_time: List[float] = []

        if ds_hour is None:
            vals_for_time = [math.nan] * len(qlon)
            times_sec.append(sec)
            rows_vals.append(vals_for_time)
            t += time_step
            continue

        try:
            z0 = ds_hour["zeta"].isel(time=0)
            max_node = int(z0.sizes.get("node", ds_hour.sizes.get("node", 0))) - 1
            valid_nodes = [n for n in node_list if 0 <= n <= max_node]

            if not valid_nodes or not node_list:
                vals_for_time = [math.nan] * len(qlon)
            else:
                vals = z0.isel(node=valid_nodes).load().values
                idx_map = {n: i for i, n in enumerate(valid_nodes)}

                for it in interp_data:
                    if it is None:
                        vals_for_time.append(math.nan)
                    else:
                        nodes, w = it
                        v = []
                        for n in nodes:
                            j = idx_map.get(n)
                            v.append(float(vals[j]) if j is not None else math.nan)
                        vals_for_time.append(math.nan if any(math.isnan(x) for x in v) else float(np.dot(w, v)))
        except Exception as e:
            print(f"[GLOFS] failed to process hour {t}: {e}")
            vals_for_time = [math.nan] * len(qlon)

        times_sec.append(sec)
        rows_vals.append(vals_for_time)
        t += time_step

    # ---- SORT + DEDUP (keep last) ----
    times = np.asarray(times_sec, dtype=np.int64)
    vals = np.asarray(rows_vals, dtype=float)

    order = np.argsort(times)
    times = times[order]
    vals = vals[order, :]

    # keep LAST occurrence for duplicates
    _, last_indices_rev = np.unique(times[::-1], return_index=True)
    keep = (times.size - 1) - last_indices_rev
    keep = np.sort(keep)
    times = times[keep]
    vals = vals[keep, :]

    # ---- WRITE RAW HOURLY ----

    # ---- WRITE RAW HOURLY ----
    base, ext = os.path.splitext(os.path.abspath(bzs_outfile))
    raw_path = base + "_raw.bzs"
    with open(raw_path, "w", encoding="utf-8") as f:
        for tsec, row in zip(times, vals):
            row = row + 0.25  # <-- add offset to values only
            f.write(f"{int(tsec)} " + " ".join(f"{v:.4f}" if np.isfinite(v) else "nan" for v in row) + "\n")
    print(f"[GLOFS] wrote raw hourly: {raw_path} ({vals.shape[0]} rows, {vals.shape[1]} points)")

    '''
    base, ext = os.path.splitext(os.path.abspath(bzs_outfile))
    raw_path = base + "_raw.bzs"
    with open(raw_path, "w", encoding="utf-8") as f:
        for tsec, row in zip(times, vals):
            f.write(f"{int(tsec)} " + " ".join(f"{v:.4f}" if np.isfinite(v) else "nan" for v in row) + "\n")
    print(f"[GLOFS] wrote raw hourly: {raw_path} ({vals.shape[0]} rows, {vals.shape[1]} points)")
    '''

    # ---- RESAMPLE TO 10-MIN (600s), like STOFS ----
    if times.size < 2:
        print("[GLOFS][warn] not enough data points to interpolate; copying raw → final")
        final_times = times
        final_vals = vals
    else:
        t0, t1 = int(times[0]), int(times[-1])
        final_times = np.arange(t0, t1 + 1, 600, dtype=int)

        final_vals = np.zeros((final_times.size, vals.shape[1]), dtype=float)
        for j in range(vals.shape[1]):
            col = vals[:, j]
            mask = np.isfinite(col)
            nvalid = int(mask.sum())
            if nvalid >= 4:
                kind = "cubic"
            elif nvalid >= 2:
                kind = "linear"
            else:
                final_vals[:, j] = 0.0
                continue
            f = interp1d(times[mask], col[mask], kind=kind, fill_value="extrapolate", assume_sorted=True)
            final_vals[:, j] = f(final_times)

    # ---- WRITE FINAL 10-MIN FILE ----

        # ---- WRITE FINAL 10-MIN FILE ----
    _ensure_dir(os.path.dirname(os.path.abspath(bzs_outfile)))
    with open(bzs_outfile, "w", encoding="utf-8") as f:
        for tsec, row in zip(final_times, final_vals):
            row = row + 0.25  # <-- add offset to values only
            f.write(f"{int(tsec)} " + " ".join(f"{v:.4f}" if np.isfinite(v) else "0.0000" for v in row) + "\n")
    print(f"[GLOFS] wrote 10-min: {bzs_outfile} ({final_vals.shape[0]} rows @600s)")
    '''
    _ensure_dir(os.path.dirname(os.path.abspath(bzs_outfile)))
    with open(bzs_outfile, "w", encoding="utf-8") as f:
        for tsec, row in zip(final_times, final_vals):
            f.write(f"{int(tsec)} " + " ".join(f"{v:.4f}" if np.isfinite(v) else "0.0000" for v in row) + "\n")
    print(f"[GLOFS] wrote 10-min: {bzs_outfile} ({final_vals.shape[0]} rows @600s)")
    '''

# -----------------------------------------------------------------------------
# Wrapper for pipeline
# -----------------------------------------------------------------------------

def build_bzs_from_glofs_legacy(
    *,
    model: str,
    bnd_file: str,
    bzs_outfile: str,
    start_dt,
    end_dt,
    time_step_hours: int = 1,
    utm_epsg: str = "EPSG:32617",
    add_360_longitudes: bool = True,
    downloads_dir: Optional[str] = None,
    extra_search_dirs: Optional[List[str]] = None,
    base_dir: Optional[str] = None,  # ignored; kept for legacy signature
) -> str:
    """
    Public entry point compatible with your existing caller.
    """
    start = _parse_dt_utc_any(start_dt)
    end = _parse_dt_utc_any(end_dt)
    extract_glofs_timeseries(
        model=model,
        bnd_file=bnd_file,
        bzs_outfile=bzs_outfile,
        start_time=start,
        end_time=end,
        time_step=timedelta(hours=int(time_step_hours)),
        utm_epsg=utm_epsg,
        add_360_longitudes=add_360_longitudes,
        downloads_dir=downloads_dir,
        extra_search_dirs=extra_search_dirs,
    )
    return bzs_outfile

