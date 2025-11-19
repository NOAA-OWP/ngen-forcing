import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import numpy as np
import xarray as xr
import pyproj
from shapely.geometry import box
from shapely.affinity import rotate


# ----------------------------
# Helpers
# ----------------------------

def _open_file_writer(filename: str, nx: int, ny: int, dx: float, dy: float,
                      x0: float, y0: float, quantity: str, unit: str):
    f = open(filename, "w", encoding="utf-8")
    f.write("FileVersion      = 1.03\n")
    f.write("filetype         = meteo_on_equidistant_grid\n")
    f.write(f"n_cols           = {nx}\n")
    f.write(f"n_rows           = {ny}\n")
    f.write("grid_unit        = m\n")
    f.write(f"x_llcorner       = {x0:.0f}\n")
    f.write(f"y_llcorner       = {y0:.0f}\n")
    f.write(f"dx               = {dx:.0f}\n")
    f.write(f"dy               = {dy:.0f}\n")
    f.write("n_quantity       = 1\n")
    f.write(f"quantity1        = {quantity}\n")
    f.write(f"unit1            = {unit}\n")
    f.write("NODATA_value     = -999\n")
    return f


def _retro_candidates(base_dir: str, when: datetime):
    """
    RETRO naming patterns to try, in order:
      1) YYYYMMDDHH00.LDASIN_DOMAIN1.nc
      2) YYYYMMDDHH00.LDASIN_DOMAIN1
      3) nwm_forcing_YYYYMMDD_HH.nc
    """
    stamp = when.strftime("%Y%m%d%H") + "00"
    return [
        os.path.join(base_dir, f"{stamp}.LDASIN_DOMAIN1.nc"),
        os.path.join(base_dir, f"{stamp}.LDASIN_DOMAIN1"),
        os.path.join(base_dir, f"nwm_forcing_{when:%Y%m%d}_{when:%H}.nc"),
    ]


def _find_first_existing(paths) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _hours_since_unix_epoch(dt: datetime) -> float:
    """
    Return hours since 1970-01-01 00:00:00 UTC.
    - If dt is naive, treat as naive epoch (matching original intent).
    - If dt is aware, convert to UTC first.
    """
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        epoch = datetime(1970, 1, 1)  # naive
        delta = dt - epoch
    else:
        dt_utc = dt.astimezone(timezone.utc)
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        delta = dt_utc - epoch
    return delta.total_seconds() / 3600.0


def _decode_scaled(ds: xr.Dataset, varname: str, arr: np.ndarray) -> np.ndarray:
    """
    Apply _FillValue masking, then scale_factor/add_offset for packed integer variables.
    No-op if attrs are missing (scale=1, offset=0).
    """
    a = arr.astype(float, copy=False)
    v = ds[varname]
    fv = v.attrs.get("_FillValue", None)
    if fv is not None:
        a = np.where(a == fv, np.nan, a)
    scale = float(v.attrs.get("scale_factor", 1.0))
    offs = float(v.attrs.get("add_offset", 0.0))
    return a * scale + offs


# ----------------------------
# Main function
# ----------------------------

def write_sfincs_meteo_from_nwm(
    start_date: datetime,
    end_date: datetime,
    *,
    mode: str,                # "ana" or "retro"
    domain_nc_path: str,      # path to sfincs.nc
    raw_root: str,            # root containing meteo/{nwm_ana|nwm_retro}
    out_dir: str,             # where sfincs.amu/.amv/.ampr/.amp go
    target_epsg: str = "32614",
    buffer_m: float = 2000.0,
    flip_vertical: bool = True,
    decode_packed: bool = False,   # only used in retro; if True, decode U2D/V2D/PSFC with scale/offset
) -> dict:
    """
    Same processing as the original script (spacing from 1D x/y, crop with jmin:jmax+1,
    then flipud, TIME from loop), with input discovery switched by mode:

      mode="ana"  -> {raw_root}/meteo/nwm_ana/nwm_forcing_YYYYMMDD_HH.nc
      mode="retro"-> {raw_root}/meteo/nwm_retro/{YYYYMMDDHH00.LDASIN_DOMAIN1[.nc] | nwm_forcing_YYYYMMDD_HH.nc}

    If decode_packed=True (retro only), apply _FillValue + scale_factor/add_offset to U2D/V2D/PSFC.
    By default (decode_packed=False), behavior is identical to ANA (no decoding).
    """
    if mode not in ("ana", "retro"):
        raise ValueError("mode must be 'ana' or 'retro'")

    os.makedirs(out_dir, exist_ok=True)

    # --- SFINCS domain + buffered, rotated bbox ---
    sfgrid = xr.open_dataset(domain_nc_path)
    x0 = float(sfgrid.attrs["x0"])
    y0 = float(sfgrid.attrs["y0"])
    nmax = int(sfgrid.attrs["nmax"])
    mmax = int(sfgrid.attrs["mmax"])
    dx_sf = float(sfgrid.attrs["dx"])
    dy_sf = float(sfgrid.attrs["dy"])
    rotation = float(sfgrid.attrs.get("rotation", 0.0))
    width = mmax * dx_sf
    height = nmax * dy_sf
    domain_box = box(x0, y0, x0 + width, y0 + height)
    rotated_domain = rotate(domain_box, rotation, origin=(x0, y0), use_radians=False)
    xsf, ysf = rotated_domain.exterior.xy
    xmin, xmax = np.min(xsf) - buffer_m, np.max(xsf) + buffer_m
    ymin, ymax = np.min(ysf) - buffer_m, np.max(ysf) + buffer_m
    sfgrid.close()

    # Writers (lazy)
    u_writer = v_writer = rr_writer = p_writer = None
    crop_idx: Optional[Tuple[int, int, int, int]] = None
    nx = ny = None
    dxnwm = dynwm = None
    x0nwm = y0nwm = None

    base_dir = os.path.join(raw_root, "meteo", "nwm_ana" if mode == "ana" else "nwm_retro")

    cur_time = start_date
    written = 0
    first_ts = None
    last_ts = None

    try:
        while cur_time <= end_date:
            # --- Input discovery per mode ---
            if mode == "ana":
                date_str = cur_time.strftime("%Y%m%d")
                hour_str = f"{cur_time.hour:02d}"
                in_path = os.path.join(base_dir, f"nwm_forcing_{date_str}_{hour_str}.nc")
                if not os.path.exists(in_path):
                    print(f"[meteo:ana][miss] {in_path}")
                    cur_time += timedelta(hours=1)
                    continue
            else:
                in_path = _find_first_existing(_retro_candidates(base_dir, cur_time))
                if in_path is None:
                    print(f"[meteo:retro][miss] {cur_time:%Y-%m-%d %H}")
                    cur_time += timedelta(hours=1)
                    continue

            # EXACT var selection as original
            ds = xr.open_dataset(in_path)[["crs", "U2D", "V2D", "RAINRATE", "PSFC"]]

            # First timestep: crop + open writers
            if crop_idx is None:
                proj = pyproj.CRS.from_cf(ds["crs"].attrs)  # same CRS path as original
                transformer = pyproj.Transformer.from_crs(proj, f"EPSG:{target_epsg}", always_xy=True)

                x = ds["x"].values
                y = ds["y"].values
                X, Y = np.meshgrid(x, y)
                xutm, yutm = transformer.transform(X, Y)

                # spacing from 1D x/y (original behavior)
                dxnwm = float(x[1] - x[0]) if x.size > 1 else dx_sf
                dynwm = float(y[1] - y[0]) if y.size > 1 else dy_sf

                mask = (xutm >= xmin) & (xutm <= xmax) & (yutm >= ymin) & (yutm <= ymax)
                iy, ix = np.where(mask)
                if iy.size == 0:
                    ds.close()
                    raise RuntimeError("No overlap between NWM grid and SFINCS domain; check EPSG/buffer.")

                imin, imax = iy.min(), iy.max()
                jmin, jmax = ix.min(), ix.max()
                crop_idx = (imin, imax, jmin, jmax)

                x_crop = xutm[imin:imax+1, jmin:jmax+1]
                y_crop = yutm[imin:imax+1, jmin:jmax+1]
                ny, nx = x_crop.shape
                x0nwm = float(x_crop[0, 0])
                y0nwm = float(y_crop[0, 0])

                u_writer = _open_file_writer(os.path.join(out_dir, "sfincs.amu"), nx, ny, dxnwm, dynwm, x0nwm, y0nwm, "x_wind", "m s-1")
                v_writer = _open_file_writer(os.path.join(out_dir, "sfincs.amv"), nx, ny, dxnwm, dynwm, x0nwm, y0nwm, "y_wind", "m s-1")
                rr_writer = _open_file_writer(os.path.join(out_dir, "sfincs.ampr"), nx, ny, dxnwm, dynwm, x0nwm, y0nwm, "rainfall", "mm hr-1")
                p_writer = _open_file_writer(os.path.join(out_dir, "sfincs.amp"), nx, ny, dxnwm, dynwm, x0nwm, y0nwm, "air_pressure", "Pa")

            # Crop arrays (time=0)
            imin, imax, jmin, jmax = crop_idx
            u_raw  = ds["U2D"].values[0, imin:imax+1, jmin:jmax+1]
            v_raw  = ds["V2D"].values[0, imin:imax+1, jmin:jmax+1]
            rr_raw = ds["RAINRATE"].values[0, imin:imax+1, jmin:jmax+1]  # mm/s
            p_raw  = ds["PSFC"].values[0, imin:imax+1, jmin:jmax+1]

            # Retro optional decoding (kept off by default)
            if mode == "retro" and decode_packed:
                u = _decode_scaled(ds, "U2D", u_raw)
                v = _decode_scaled(ds, "V2D", v_raw)
                p = _decode_scaled(ds, "PSFC", p_raw)
            else:
                u, v, p = u_raw, v_raw, p_raw

            # Rain: always convert mm/s -> mm/hr; optional fill masking if decode_packed
            if mode == "retro" and decode_packed:
                rr_fill = ds["RAINRATE"].attrs.get("_FillValue", None)
                if rr_fill is not None:
                    rr_raw = np.where(rr_raw == rr_fill, np.nan, rr_raw)
            rr = rr_raw * 3600.0

            # Orientation identical to original
            if flip_vertical:
                u = np.flipud(u)
                v = np.flipud(v)
                rr = np.flipud(rr)
                p = np.flipud(p)

            # TIME header from loop dt (naive/aware-safe)
            offset = _hours_since_unix_epoch(cur_time)
            stamp = f"TIME = {offset:.6f} hours since 1970-01-01 00:00:00 +00:00  # {cur_time:%Y-%m-%d %H:%M:%S}\n"

            for f, data in ((u_writer, u), (v_writer, v), (rr_writer, rr), (p_writer, p)):
                f.write(stamp)
                for row in data:
                    f.write(" ".join(f"{val:.5g}" for val in row) + "\n")

            ds.close()
            written += 1
            if first_ts is None:
                first_ts = cur_time
            last_ts = cur_time
            cur_time += timedelta(hours=1)

    finally:
        for f in (u_writer, v_writer, rr_writer, p_writer):
            if f:
                f.close()

    return {
        "timesteps_written": written,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "output_dir": out_dir,
        "grid": {"nx": nx, "ny": ny, "dx": dxnwm, "dy": dynwm, "x0": x0nwm, "y0": y0nwm},
        "mode": mode,
        "decode_packed": decode_packed,
    }


