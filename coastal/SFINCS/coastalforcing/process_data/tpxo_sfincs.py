import os
from pathlib import Path
from typing import Iterable, Optional, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyproj
import subprocess
import shutil
import re


# ---------- helpers ----------

def _project_root() -> Path:
    """
    Return the absolute path to the project root (the 'coastalforcing' directory),
    assuming this file lives under coastalforcing/process_data/.
    """
    return Path(__file__).resolve().parents[1]


def _parse_dt_any(dt_like: Union[str, datetime]) -> datetime:
    if isinstance(dt_like, datetime):
        return dt_like.replace(tzinfo=None)
    s = str(dt_like).strip().replace("T", " ")
    if s.endswith(("Z", "z")):
        s = s[:-1]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return pd.to_datetime(s).to_pydatetime().replace(tzinfo=None)


# ---------- step 1: make TPXO input (lat lon time) ----------

def _make_tpxo_input_from_sfincs_bnd(
    bnd_file_abs: Union[str, Path],
    start_time: Union[str, datetime],
    end_time: Union[str, datetime],
    utm_epsg: Union[int, str],
    out_path_abs: Union[str, Path],
    step_seconds: int = 600,
) -> None:
    """
    Create TPXO/OTPS station+time input from a SFINCS .bnd (x,y in given UTM).
    Writes file with lines: "<lat>  <lon>  YYYY MM DD HH mm ss"
    """
    bnd_file_abs = Path(bnd_file_abs).resolve()
    out_path_abs = Path(out_path_abs).resolve()
    out_path_abs.parent.mkdir(parents=True, exist_ok=True)

    t0 = _parse_dt_any(start_time)
    t1 = _parse_dt_any(end_time)
    if t1 < t0:
        raise ValueError("end_time must be >= start_time")

    # read .bnd coordinates (columns 0/1 = x/y)
    bnd = pd.read_csv(bnd_file_abs, sep=r"\s+", header=None, usecols=[0, 1],
                      names=["x", "y"], engine="python")
    if bnd.empty:
        raise RuntimeError(f"No points read from {bnd_file_abs}")

    # UTM -> WGS84
    utm_str = f"EPSG:{utm_epsg}" if isinstance(utm_epsg, int) else str(utm_epsg)
    to_wgs84 = pyproj.Transformer.from_crs(pyproj.CRS(utm_str), pyproj.CRS("EPSG:4326"), always_xy=True)
    lon, lat = to_wgs84.transform(bnd["x"].to_numpy(), bnd["y"].to_numpy())

    # times
    step = timedelta(seconds=int(step_seconds))
    times = []
    tt = t0
    while tt <= t1:
        times.append(tt)
        tt += step
    if not times:
        times = [t0]

    # write
    with open(out_path_abs, "w", encoding="utf-8") as f:
        for la, lo in zip(lat, lon):
            la = float(la)
            lo = float(lo)
            for dt in times:
                f.write(
                    f"{la:.10f}  {lo:.10f}  {dt.year:04d} {dt.month:02d} {dt.day:02d} "
                    f"{dt.hour:02d} {dt.minute:02d} {dt.second:02d}\n"
                )
    print(f"[tpxo] wrote lat/lon/time → {out_path_abs}")


# ---------- step 2: run predict_tide with relative paths ----------

def _run_predict_tide_relative(
    project_root: Path,
    *,
    tp_dir_rel: Path,                 # e.g., Path("process_data/TPXO")
    predict_tide_rel: Path,           # e.g., Path("process_data/TPXO/predict_tide")
    model_control_rel: Path,          # e.g., Path("process_data/TPXO/Model_tpxo10_atlas")
    latlon_time_rel: Path,            # e.g., Path("process_data/TPXO/tpxo_lat_lon_time")
    output_rel: Path,                 # e.g., Path("process_data/TPXO/tpxo_out.txt")
    setup_rel: Path = Path("process_data/TPXO/setup.tpxo_sfincs"),
    constituents: Optional[Iterable[str]] = None,  # blank → model default
    ap_or_ri: str = "AP",
    oce_or_geo: str = "oce",
    correct_minor: bool = True,
    env_extra: Optional[dict] = None,
) -> Path:
    """
    Write a short-path setup file (relative to project root) and run predict_tide.
    All file paths in the setup are *relative*, and the subprocess runs with cwd=project_root.
    """
    # Ensure folders exist
    (project_root / tp_dir_rel).mkdir(parents=True, exist_ok=True)

    # Build constituents line
    line4 = "" if not constituents else " ".join(str(c).strip() for c in constituents if str(c).strip())

    # Prepare setup text (8 lines, then some blank lines)
    setup_text = "\n".join([
        str(model_control_rel).replace("\\", "/"),       # 1) model control (relative)
        str(latlon_time_rel).replace("\\", "/"),         # 2) lat lon time (relative)
        "z",                                             # 3) predict elevations
        line4,                                           # 4) constituents or blank
        ap_or_ri,                                        # 5) AP or RI
        oce_or_geo,                                      # 6) oce or geo
        "1" if correct_minor else "0",                   # 7) minor constituents
        str(output_rel).replace("\\", "/"),              # 8) output (relative)
        "", "", "",
    ])

    # Write setup file inside process_data/TPXO
    setup_abs = (project_root / setup_rel).resolve()
    setup_abs.parent.mkdir(parents=True, exist_ok=True)
    with open(setup_abs, "w", encoding="utf-8") as f:
        f.write(setup_text)

    # Command uses *relative* exe and setup path, run at project_root
    cmd = f"{str(predict_tide_rel).replace(os.sep, '/')} < {str(setup_rel).replace(os.sep, '/')}"
    print("[tpxo] Running predict_tide:")
    print(f"  (cwd) {project_root}")
    print(f"  {cmd}")

    proc_env = os.environ.copy()
    proc_env.setdefault("LC_ALL", "C")
    if env_extra:
        proc_env.update({str(k): str(v) for k, v in env_extra.items()})

    result = subprocess.run(
        cmd,
        shell=True,
        cwd=str(project_root),
        env=proc_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    # decode for logs safely
    def _safe(b: bytes) -> str:
        try:
            return b.decode("utf-8", errors="replace")
        except Exception:
            return b.decode("latin-1", errors="replace")

    if result.stdout:
        s = _safe(result.stdout)
        print(f"[tpxo][stdout] ({len(result.stdout)} bytes)\n" + s[:2000] + ("...\n" if len(s) > 2000 else ""))
    if result.stderr:
        s = _safe(result.stderr)
        print(f"[tpxo][stderr] ({len(result.stderr)} bytes)\n" + s[:2000] + ("...\n" if len(s) > 2000 else ""))

    if result.returncode != 0:
        raise RuntimeError(f"predict_tide exited with code {result.returncode}")

    out_abs = (project_root / output_rel).resolve()
    if not out_abs.exists() or out_abs.stat().st_size == 0:
        raise RuntimeError(f"predict_tide finished but output missing/empty: {out_abs}")

    print(f"[tpxo] results → {out_abs}")
    return out_abs


def _prepend_tpxo_header(output_abs: Path,
                         model_label: Optional[str] = None,
                         constituents: Optional[Iterable[str]] = None) -> None:
    """
    Prepend a human-readable header to predict_tide ASCII output.
    """
    if not output_abs.exists() or output_abs.stat().st_size == 0:
        raise FileNotFoundError(f"Output not found or empty: {output_abs}")

    if model_label is None:
        # best-effort
        model_label = "tpxo_model"

    const_line = " ".join(constituents) if constituents else ""

    original = output_abs.read_text(encoding="utf-8", errors="replace")
    header_lines = [
        "------------------------------------------------------------",
        f" Model:        {model_label}",
        f" Constituents included: {const_line}".rstrip(),
        "",
        "     Lat       Lon        mm.dd.yyyy hh:mm:ss     z(m)   Depth(m)",
        "",
        "",
    ]
    tmp = output_abs.with_suffix(output_abs.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines))
        f.write(original if original.startswith("\n") else "\n" + original)
    os.replace(tmp, output_abs)


# ---------- step 3: convert TPXO output → sfincs.bzs ----------
def _tpxo_out_to_sfincs_bzs(
    tpxo_out_abs: Union[str, Path],
    latlon_time_abs: Union[str, Path],
    bnd_file_abs: Union[str, Path],
    start_time: Union[str, datetime],
    out_bzs_abs: Union[str, Path],
    tol_deg: float = 1e-4,
    verbose: bool = True,
) -> None:
    """
    Convert predict_tide ASCII output to SFINCS .bzs (time in seconds since start_time).
    Uses the *input* lat/lon/time file to define station order and shared time axis.
    """
    tpxo_out_abs = Path(tpxo_out_abs).resolve()
    latlon_time_abs = Path(latlon_time_abs).resolve()
    bnd_file_abs = Path(bnd_file_abs).resolve()
    out_bzs_abs = Path(out_bzs_abs).resolve()
    out_bzs_abs.parent.mkdir(parents=True, exist_ok=True)

    start_dt = _parse_dt_any(start_time)
    if verbose:
        print(f"[tpxo→bzs] start_time: {start_dt}")

    # check .bnd count
    bnd = pd.read_csv(bnd_file_abs, sep=r"\s+", header=None, usecols=[0, 1],
                      names=["x", "y"], engine="python")
    n_bnd = len(bnd)
    if n_bnd == 0:
        raise RuntimeError(f"No points in {bnd_file_abs}")

    # parse TPXO input (order = stations, first station's block = time axis)
    in_re = re.compile(
        r"^\s*([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)\s+(\d{4})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s*$"
    )
    in_lats, in_lons, in_dts = [], [], []
    with open(latlon_time_abs, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            m = in_re.match(raw)
            if not m:
                continue
            la = float(m.group(1)); lo = float(m.group(2))
            Y, M, D = int(m.group(3)), int(m.group(4)), int(m.group(5))
            h, m_, s_ = int(m.group(6)), int(m.group(7)), int(m.group(8))
            in_lats.append(la); in_lons.append(lo)
            in_dts.append(datetime(Y, M, D, h, m_, s_))
    if not in_lats:
        raise RuntimeError(f"No valid lines in {latlon_time_abs}")

    stations = []
    seen = set()
    for la, lo in zip(in_lats, in_lons):
        key = (round(la, 6), round(lo, 6))
        if key not in seen:
            stations.append((la, lo))
            seen.add(key)
    first_la, first_lo = stations[0]
    time_list = []
    for la, lo, dt in zip(in_lats, in_lons, in_dts):
        if abs(la - first_la) <= 1e-12 and abs(lo - first_lo) <= 1e-12:
            time_list.append(dt)
        else:
            break
    n_stn = len(stations)
    n_time = len(time_list)
    if verbose:
        print(f"[tpxo→bzs] stations={n_stn}, times per station={n_time}")

    # map station & time quickly
    stn_lat = np.array([s[0] for s in stations])
    stn_lon = np.array([s[1] for s in stations])

    def _find_station_index(la, lo):
        d = np.maximum(np.abs(stn_lat - la), np.abs(stn_lon - lo))
        j = int(np.argmin(d))
        return j if d[j] <= tol_deg else None

    time_index = {t: i for i, t in enumerate(time_list)}

    # parse predict_tide output
    line_re = re.compile(
        r"^\s*([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)\s+(\d{2}\.\d{2}\.\d{4})\s+(\d{2}:\d{2}:\d{2})\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)\s*$"
    )
    arr = np.full((n_time, n_stn), np.nan, dtype=float)
    n_parsed = n_matched = n_miss_stn = n_miss_time = 0
    with open(tpxo_out_abs, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            m = line_re.match(raw)
            if not m:
                continue
            n_parsed += 1
            la = float(m.group(1)); lo = float(m.group(2))
            mdY = m.group(3); HMS = m.group(4)
            z = float(m.group(5))
            dt = datetime.strptime(mdY + " " + HMS, "%m.%d.%Y %H:%M:%S")
            j = _find_station_index(la, lo)
            if j is None:
                n_miss_stn += 1
                continue
            i = time_index.get(dt)
            if i is None:
                n_miss_time += 1
                continue
            arr[i, j] = z
            n_matched += 1

    if verbose:
        print(f"[tpxo→bzs] matched={n_matched}, miss_stn={n_miss_stn}, miss_time={n_miss_time}, parsed={n_parsed}")

    # seconds since start
    sec_since_start = np.array([(t - start_dt).total_seconds() for t in time_list], dtype=int)
  
    with open(out_bzs_abs, "w", encoding="utf-8") as f:
        for i in range(n_time):
            vals = " ".join(f"{v:.4f}" for v in arr[i, :])
            f.write(f"{int(sec_since_start[i])} {vals}\n")
    '''

    with open(out_bzs_abs, "w", encoding="utf-8") as f:
        for i in range(n_time):
            row = arr[i, :] + 1.0  # <-- add offset to values only (time unchanged)
            print(f"{arr[i, :]} : {row}")
            vals = " ".join(f"{v:.4f}" for v in row)
            f.write(f"{int(sec_since_start[i])} {vals}\n")
    '''

    if verbose:
        print(f"[tpxo→bzs] wrote {out_bzs_abs} ({n_time} rows × {n_stn} stations + time)")


# ---------- one-call pipeline ----------

def run_tpxo_pipeline_for_sfincs(
    *,
    # required
    sfincs_bnd_file: Union[str, Path],   # absolute path to run-folder sfincs.bnd
    start_time: Union[str, datetime],
    end_time: Union[str, datetime],
    utm_epsg: Union[int, str],

    # optional overrides (defaults match your layout & constraints)
    predict_tide_exe_rel: Union[str, Path] = "process_data/TPXO/predict_tide",
    model_control_rel: Union[str, Path] = "process_data/TPXO/Model_tpxo10_atlas",
    tp_dir_rel: Union[str, Path] = "process_data/TPXO",
    lat_lon_time_rel: Union[str, Path] = "process_data/TPXO/tpxo_lat_lon_time",
    tpxo_out_rel: Union[str, Path] = "process_data/TPXO/tpxo_out.txt",
    setup_rel: Union[str, Path] = "process_data/TPXO/setup.tpxo_sfincs",
    out_bzs_path: Optional[Union[str, Path]] = None,   # default: <run-dir>/sfincs.bzs

    step_seconds: int = 600,
    prepend_header_block: bool = True,
    header_model_label: Optional[str] = "tpxo10_atlas",
    header_constituents: Optional[Iterable[str]] = None,
    env_extra: Optional[dict] = None,
) -> dict:
    """
    Full pipeline:
      1) make lat/lon/time from SFINCS .bnd (in UTM)  → process_data/TPXO/tpxo_lat_lon_time
      2) run predict_tide with *relative* paths      → process_data/TPXO/tpxo_out.txt
      3) (optional) prepend header to ascii output
      4) convert to <run-dir>/sfincs.bzs (time in seconds since start_time)

    Returns dict with key paths.
    """
    project = _project_root()                              # .../coastalforcing
    tp_dir_rel = Path(tp_dir_rel)
    lat_lon_time_rel = Path(lat_lon_time_rel)
    tpxo_out_rel = Path(tpxo_out_rel)
    model_control_rel = Path(model_control_rel)
    predict_tide_exe_rel = Path(predict_tide_exe_rel)
    setup_rel = Path(setup_rel)

    # Absolute paths for step 1 (we generate file), but we’ll reference them relatively in setup
    latlon_abs = (project / lat_lon_time_rel).resolve()
    tp_dir_abs = (project / tp_dir_rel).resolve()
    tp_dir_abs.mkdir(parents=True, exist_ok=True)

    # 1) write lat/lon/time
    _make_tpxo_input_from_sfincs_bnd(
        bnd_file_abs=sfincs_bnd_file,
        start_time=start_time,
        end_time=end_time,
        utm_epsg=utm_epsg,
        out_path_abs=latlon_abs,
        step_seconds=step_seconds,
    )

    # 2) run predict_tide with only relative paths inside setup & short setup path
    out_abs = _run_predict_tide_relative(
        project_root=project,
        tp_dir_rel=tp_dir_rel,
        predict_tide_rel=predict_tide_exe_rel,
        model_control_rel=model_control_rel,
        latlon_time_rel=lat_lon_time_rel,
        output_rel=tpxo_out_rel,
        setup_rel=setup_rel,
        constituents=None,          # or pass a list if you want explicit set
        ap_or_ri="AP",
        oce_or_geo="oce",
        correct_minor=True,
        env_extra=env_extra,
    )

    # 3) (optional) header
    if prepend_header_block:
        _prepend_tpxo_header(out_abs, model_label=header_model_label,
                             constituents=header_constituents)

    # 4) convert → sfincs.bzs in the run directory
    if out_bzs_path is None:
        # default to sibling of the .bnd
        out_bzs_path = Path(sfincs_bnd_file).resolve().with_name("sfincs.bzs")
    out_bzs_abs = Path(out_bzs_path).resolve()

    _tpxo_out_to_sfincs_bzs(
        tpxo_out_abs=out_abs,
        latlon_time_abs=latlon_abs,
        bnd_file_abs=sfincs_bnd_file,
        start_time=start_time,
        out_bzs_abs=out_bzs_abs,
        tol_deg=1e-4,
        verbose=True,
    )

    return {
        "project_root": str(project),
        "lat_lon_time": str(latlon_abs),
        "tpxo_out": str(out_abs),
        "sfincs_bzs": str(out_bzs_abs),
    }

