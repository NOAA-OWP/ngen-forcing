# coastalforcing/process_data/singlefile_dis.py

import os
import re
import shlex
import numpy as np
import xarray as xr
from datetime import datetime
from typing import List, Optional

def _parse_src(sfincs_src: str):
    """
    Read SFINCS .src lines like:
        <x> <y> "nex-2414735 poi-86738"
    Returns list of (x, y, feature_id_int, raw_label)
    """
    pts = []
    nex_pat = re.compile(r"nex-(\d+)")
    with open(sfincs_src, "r") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                parts = shlex.split(s)
            except ValueError:
                parts = s.split()
            if len(parts) < 3:
                continue
            try:
                x = float(parts[0]); y = float(parts[1])
            except Exception:
                continue
            label = parts[2]
            m = nex_pat.search(label)
            if not m:
                continue
            nexus = int(m.group(1))
            pts.append((x, y, nexus, label))
    if not pts:
        raise RuntimeError(f"No usable points parsed from {sfincs_src}")
    return pts

def _write_dis(output_path: str, times: np.ndarray, columns: List[np.ndarray]):
    """
    Write SFINCS .dis file (space-separated).
    First column: integer seconds.
    """
    T = len(times)
    for c in columns:
        if len(c) != T:
            raise RuntimeError("column length mismatch")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for i in range(T):
            row = [f"{int(times[i])}"]
            for col in columns:
                v = float(col[i])
                if not np.isfinite(v):
                    v = 0.0
                row.append(f"{v:.6f}")
            f.write(" ".join(row) + "\n")

def build_dis_from_ngen_netcdf(
    sfincs_src: str,
    troute_nc: str,
    output_dis: str,
    flow_var: str = "flow",
    time_var: str = "time",
    id_var: str = "feature_id",
    fill_missing: float = 0.0,
    start_time: Optional[datetime] = None,  # if provided, write seconds since this start
):
    """
    Reuses your previous single-file logic, with an optional start_time.
    - If start_time is None: time column is written as given by the netCDF (numeric seconds).
    - If start_time is provided: convert netCDF times to seconds since start_time.
    """
    # 1) parse src
    pts = _parse_src(sfincs_src)
    nexus_ids = [p[2] for p in pts]

    # 2) open netcdf (decode_times=False to keep numeric axis)
    ds = xr.open_dataset(troute_nc, engine="netcdf4", decode_times=False)

    # sanity checks
    if id_var not in ds or time_var not in ds or flow_var not in ds:
        raise KeyError(f"Missing '{id_var}' or '{time_var}' or '{flow_var}' in {troute_nc}")

    feature_ids = ds[id_var].values           # (feature,)
    times = np.asarray(ds[time_var].values)   # numeric vector (e.g., seconds since ref)
    flow = ds[flow_var]
    # make dims (feature, time)
    if flow.dims == (time_var, id_var):
        flow = flow.transpose(id_var, time_var)

    # map feature_id -> index
    id_index = {int(fid): int(i) for i, fid in enumerate(np.asarray(feature_ids).tolist())}

    # time handling
    if start_time is not None:
        # write seconds since start_time (assuming numeric netcdf time ≈ seconds since file ref)
        # First, try to interpret netcdf times as seconds since the file’s reference;
        # then compute absolute epoch by reading the units if present.
        # If units are missing, we fall back to (times - times[0]) and shift to start_time.
        units = ds[time_var].attrs.get("units", "")
        if "since" in units:
            # e.g. "seconds since 2023-04-01 00:00:00"
            try:
                ref_str = units.split("since", 1)[1].strip()
                ref_str = ref_str.replace("UTC", "").strip()
                # accept "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DDTHH:MM:SS"
                ref_str = ref_str.replace("T", " ")
                file_ref = datetime.strptime(ref_str, "%Y-%m-%d %H:%M:%S")
                # absolute seconds for each time stamp:
                abs_epoch = np.array([file_ref.timestamp() + float(t) for t in times])
                times_sec = abs_epoch - start_time.replace(tzinfo=None).timestamp()
                times_sec = times_sec.astype(np.int64)
            except Exception:
                # fallback: relative to first sample, then shift to start_time
                rel = (times - times[0]).astype(np.float64)
                times_sec = rel
        else:
            rel = (times - times[0]).astype(np.float64)
            times_sec = rel
    else:
        # write raw numeric time
        times_sec = times.astype(np.int64)

    # columns (in src order)
    cols = []
    fv = flow.encoding.get("_FillValue", flow.attrs.get("_FillValue", None))
    for fid in nexus_ids:
        j = id_index.get(fid)
        if j is None:
            cols.append(np.full(times.shape, float(fill_missing)))
        else:
            c = flow.isel({flow.dims[0]: j}).values.astype(float)
            if fv is not None:
                c = np.where(c == fv, float(fill_missing), c)
            cols.append(c)

    _write_dis(output_dis, np.asarray(times_sec), cols)
    return len(nexus_ids), sum(np.any(np.isfinite(c)) for c in cols), len(times_sec)

