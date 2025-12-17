#!/usr/bin/env python3
"""
Compare SFINCS point output (sfincs_his.nc) vs NOAA observed water level JSON.

- NOAA observation: time vs water level (meters, NAVD)   <-- as provided in JSON
- SFINCS:           time vs point_zs for the station_name that contains (NOAAID)

Outputs (per station):
  <outdir>/<station>.png
  <outdir>/<station>_sim.txt
  <outdir>/<station>_obv.txt
  <outdir>/<station>.csv           # aligned table for convenience
  <outdir>/summary_metrics.csv     # RMSE, bias, corr, N
"""

import os
import re
import json
import argparse
from typing import List, Tuple, Dict

# Headless-safe backend
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker


# ------------------------- helpers -------------------------

def _parse_list(arg: str) -> List[str]:
    return [s.strip() for s in arg.split(",") if s.strip()]


def _decode_station_names(station_char: np.ndarray) -> List[str]:
    """
    Convert a CF-style char array (stations, str_len) to a list of Python strings.
    Robust to bytes/objects/char arrays.
    """
    # If we already received an array of “strings”, normalize to Python str
    if station_char.dtype.kind in ("U", "S", "O"):
        out = []
        for row in station_char:
            if isinstance(row, (bytes, bytearray)):
                out.append(row.decode("utf-8", "ignore").strip())
            elif isinstance(row, np.ndarray):
                # a row of bytes/chars
                out.append("".join(
                    (b.decode("utf-8", "ignore") if isinstance(b, (bytes, bytearray)) else str(b))
                    for b in row
                ).strip())
            else:
                out.append(str(row).strip())
        return out
    # Typical char array path
    return ["".join(ch.astype(str)).strip() for ch in station_char]


def _map_stationid_to_index(names: List[str]) -> Dict[str, int]:
    """
    Dict: NOAAID (digits) -> station index
    Extract '(NNNNNNN)' from station_name and keep pure-digit IDs.
    """
    id_to_idx: Dict[str, int] = {}
    for i, nm in enumerate(names):
        m = re.search(r"\(([^)]+)\)", nm)
        if not m:
            continue
        inside = m.group(1).strip()
        if inside.isdigit():
            id_to_idx[inside] = i
    return id_to_idx


def _read_noaa_json(path: str) -> Tuple[pd.DatetimeIndex, np.ndarray, Dict]:
    """
    Read NOAA JSON (as downloaded meters/NAVD).
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    meta = payload.get("metadata", {})
    rows = payload.get("data", [])

    # NOAA timestamps are "YYYY-MM-DD HH:MM" in UTC
    t = pd.to_datetime([r.get("t") for r in rows], format="%Y-%m-%d %H:%M", errors="coerce")
    v = np.array([float(r["v"]) if r.get("v") not in (None, "", "NaN") else np.nan for r in rows], float)

    mask = ~pd.isna(t)
    times = pd.DatetimeIndex(t[mask]).tz_localize(None)
    vals = v[mask]
    return times, vals, meta


def _interp_to(times_src: pd.DatetimeIndex, vals_src: np.ndarray,
               times_dst: pd.DatetimeIndex) -> np.ndarray:
    """
    Time-linear interpolate from src → dst with end fill.
    """
    if len(times_src) == 0 or len(times_dst) == 0:
        return np.full(times_dst.shape, np.nan)
    s = pd.Series(vals_src, index=pd.DatetimeIndex(times_src)).sort_index()
    if s.notna().sum() < 1:
        return np.full(times_dst.shape, np.nan)
    s = s.interpolate(method="time", limit_direction="both")
    return s.reindex(times_dst).interpolate(method="time", limit_direction="both").values


def _metrics(obs: np.ndarray, mod: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(obs) & np.isfinite(mod)
    if m.sum() == 0:
        return {"rmse": np.nan, "bias": np.nan, "corr": np.nan, "n": 0}
    o = obs[m]; md = mod[m]
    rmse = float(np.sqrt(np.mean((md - o) ** 2)))
    bias = float(np.mean(md - o))
    corr = float(np.corrcoef(o, md)[0, 1]) if len(o) > 1 else np.nan
    return {"rmse": rmse, "bias": bias, "corr": corr, "n": int(m.sum())}


# ------------------------- core -------------------------

def compare_sfincs_his_vs_noaa(
    sfincs_his_nc: str,
    noaa_dir: str,
    stations: List[str],
    outdir: str,
    *,
    datum_shift: float = 0.0,
    resample_to: str = "model",   # 'model' or 'obs' for CSV/metrics alignment
    fig_dpi: int = 140,
    verbose: bool = False,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Open sfincs_his.nc

    ds = xr.open_dataset(sfincs_his_nc, decode_cf=True)

    # Time → pandas
    if "time" not in ds.variables:
        raise KeyError("No 'time' variable in sfincs_his.nc")
    time_units = ds["time"].attrs.get("units", None)
    if time_units:
        tvals = nc.num2date(ds["time"].values, units=time_units)
        t_model = pd.to_datetime(tvals).tz_localize(None)
    else:
        t_model = pd.DatetimeIndex(pd.to_datetime(ds["time"].values)).tz_localize(None)

    # Station names
    if "station_name" not in ds.variables:
        raise KeyError("No 'station_name' in sfincs_his.nc")
    names = _decode_station_names(ds["station_name"].values)
    id_to_idx = _map_stationid_to_index(names)

    if verbose:
        found_ids = sorted(id_to_idx.keys())
        print(f"[info] SFINCS station IDs (digits extracted from station_name): {found_ids}")

    # SFINCS point water level
    if "point_zs" not in ds.variables:
        raise KeyError("No 'point_zs' in sfincs_his.nc")
    point_zs = np.asarray(ds["point_zs"].values, float)  # (time, stations)

    # Plot style
    major_fmt = mdates.DateFormatter("%b %d")
    minor_fmt = mdates.DateFormatter("%Hz")
    day_locator = mdates.DayLocator()
    hour_locator = mdates.HourLocator(interval=2)

    processed = 0
    summary = []

    for sid in stations:
        if sid not in id_to_idx:
            print(f"[skip] NOAA station {sid} not found in sfincs_his station_name list")
            continue

        jpath = os.path.join(noaa_dir, f"{sid}.json")
        if not os.path.exists(jpath):
            print(f"[skip] missing NOAA JSON: {jpath}")
            continue

        # OBS
        t_obs, v_obs, meta = _read_noaa_json(jpath)

        # MODEL @ station index
        sidx = id_to_idx[sid]
        v_model = point_zs[:, sidx].astype(float)
        if datum_shift:
            v_model = v_model + float(datum_shift)

        # Align for metrics/CSV
        if resample_to.lower() == "obs":
            model_on_obs = _interp_to(t_model, v_model, t_obs)
            times_aligned = t_obs
            obs_aligned = v_obs
            mod_aligned = model_on_obs
        else:
            obs_on_model = _interp_to(t_obs, v_obs, t_model)
            times_aligned = t_model
            obs_aligned = obs_on_model
            mod_aligned = v_model

        # Metrics
        met = _metrics(obs_aligned, mod_aligned)
        print(f"[{sid}] station_idx={sidx}  N={met['n']}  RMSE={met['rmse']:.3f}  "
              f"Bias={met['bias']:.3f}  Corr={met['corr']:.2f}")

        # ------- plot -------
        fig, ax = plt.subplots(figsize=(10, 4), dpi=fig_dpi)

        # Trim obs to model window for plotting
        tmin = t_model[0] - pd.Timedelta(hours=1)
        tmax = t_model[-1] + pd.Timedelta(hours=1)
        mask_plot_obs = (t_obs >= tmin) & (t_obs <= tmax)
        t_obs_plot = t_obs[mask_plot_obs]
        v_obs_plot = v_obs[mask_plot_obs]

        # NOAA observation: blue circles, solid line
        ax.plot_date(
                t_obs_plot,
                v_obs_plot,
                fmt="o-",
                markersize=5,
                markerfacecolor="None",
                color="blue",
                label="NOAA obs"
        )

        # SFINCS simulation: red crosses, dashed line
        ax.plot_date(
                t_model,
                v_model,
                fmt="x:",
                markersize=5,
                markerfacecolor="None",
                color="red",
                label="SFINCS sim"
        )

        ax.xaxis.set_major_formatter(major_fmt)
        ax.xaxis.set_minor_formatter(minor_fmt)
        ax.xaxis.set_major_locator(day_locator)
        ax.xaxis.set_minor_locator(hour_locator)
        ax.yaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis="x", rotation=90)

        ax.set_xlim(tmin, tmax)

        # Y limits spanning both
        yvals = []
        if len(v_model) > 0:
                yvals.extend(v_model.tolist())
        if len(v_obs_plot) > 0:
                yvals.extend(v_obs_plot.tolist())
        yvals = [y for y in yvals if np.isfinite(y)]
        if yvals:
                ax.set_ylim(bottom=min(yvals), top=max(yvals))

        ax.grid(True, which="minor", axis="x")
        ax.grid(True, which="major", axis="y")
        ax.set_xlabel("Time")
        ax.set_ylabel("Elevation (m)")
        ax.set_title(f"Station ID: {sid}  —  {names[sidx]}")

        lgd = ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{sid}.png"), bbox_inches="tight")
        plt.close(fig)

        # ------- text outputs -------
        with open(os.path.join(outdir, f"{sid}_sim.txt"), "w", encoding="utf-8") as ftxt:
                ftxt.write(f"#SFINCS station idx: {sidx} name: {names[sidx]}\n")
                ftxt.write("#time(YYYYmmdd_HH:%M)    water_level (m NAVD88)\n")
                for t, val in zip(t_model, v_model):
                        ftxt.write(f"{t.strftime('%Y%m%d_%H:%M')} {val}\n")

        with open(os.path.join(outdir, f"{sid}_obv.txt"), "w", encoding="utf-8") as ftxt:
                ftxt.write(f"#{meta}\n")
                ftxt.write("#time(YYYYmmdd_HH:%M)    water_level (m NAVD88)\n")
                for t, val in zip(t_obs, v_obs):
                        ftxt.write(f"{t.strftime('%Y%m%d_%H:%M')} {val}\n")

        # ------- CSV (aligned, with labels) -------
        pd.DataFrame({
                "time_utc": times_aligned,
                "NOAA_obs_m": obs_aligned,
                "SFINCS_sim_m": mod_aligned
        }).to_csv(os.path.join(outdir, f"{sid}.csv"), index=False)


        # CSV (aligned)
        csv_path = os.path.join(outdir, f"{sid}.csv")
        pd.DataFrame({
            "time_utc": times_aligned,
            "obs": obs_aligned,
            "model": mod_aligned
        }).to_csv(csv_path, index=False)
        print(f"[save] {csv_path}")

        summary.append({"station": sid, **met})
        processed += 1

    if summary:
        sum_path = os.path.join(outdir, "summary_metrics.csv")
        pd.DataFrame(summary).to_csv(sum_path, index=False)
        print(f"[done] wrote {sum_path} for {len(summary)} station(s)")
    else:
        print("[warn] no stations processed; nothing written.")

    ds.close()


# ------------------------- CLI -------------------------

def main():
    p = argparse.ArgumentParser(
        description="Compare sfincs_his.nc point_zs vs NOAA JSON by matching station_name (...(ID))."
    )
    p.add_argument("--sfincs_his", required=True, help="Path to sfincs_his.nc")
    p.add_argument("--noaa_dir", required=True, help="Directory containing NOAA *.json")
    p.add_argument("--stations", required=True, help="Comma-separated station IDs")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--datum_shift", type=float, default=0.0,
                   help="Additive shift (m) applied to model values")
    p.add_argument("--resample_to", choices=["model", "obs"], default="model",
                   help="Align times on 'model' (default) or 'obs' for CSV/metrics")
    p.add_argument("--verbose", action="store_true", help="Print parsed station IDs from sfincs_his")
    args = p.parse_args()

    compare_sfincs_his_vs_noaa(
        sfincs_his_nc=args.sfincs_his,
        noaa_dir=args.noaa_dir,
        stations=_parse_list(args.stations),
        outdir=args.outdir,
        datum_shift=args.datum_shift,
        resample_to=args.resample_to,
        fig_dpi=140,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

