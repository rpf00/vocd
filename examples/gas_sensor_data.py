"""
gas_sensor_data.py
==================
Download, parse, and prepare the Wörner et al. (2025) Electronic Nose
Long-Term Drift Dataset for use with the VOCD drift-correction pipeline.

Reference
---------
  Wörner, Eimler & Pein-Hackelbusch (2025).
  Long-term drift behavior in metal oxide gas sensor arrays: a one-year
  dataset from an electronic nose.  Scientific Data 12, 1628.
  DOI: 10.1038/s41597-025-05993-8
  Dataset: https://doi.org/10.5281/zenodo.15681119

Dataset structure
-----------------
  39 measurement days  (day 14 excluded by authors — protocol deviation)
  18 measurements per day in fixed order:
      3× EtOH 5%v/v
      3× Diacetyl 0.1 ppm
      3× Diacetyl 1 ppm
      3× EtOH 5%v/v  (repeated mid-day)
      3× 2-Phenylethanol 200 ppm
      3× 2-Phenylethanol 1000 ppm
  62 MOX sensors (R1–R62) + temperature (T) + humidity (H)
  Each file = one measurement, raw resistance time series at ~1 Hz
  Cycle stages stored per row:  1=baseline  2=equilibration  3=sample

Signals for drift analysis
--------------------------
  Baseline resistance B(t): mean of last 20 readings of stage 1 before
  sample injection.  Measured before EACH of the 18 daily measurements
  → 18 × 39 = ~700 baseline points per sensor, ordered chronologically.
  This is the continuous drift signal VOCD runs on.

  Response ΔR(t): |B(t) − mean(stage-3 readings)| for a given analyte.
  Tracks sensitivity drift separately from baseline drift.

Usage
-----
  python gas_sensor_data.py                   # download + all plots
  python gas_sensor_data.py --sensor 0        # sensor index 0-61
  python gas_sensor_data.py --analyte EtOH    # EtOH | Diacetyl_0.1ppm |
                                              # Diacetyl_1ppm | PEA_200ppm |
                                              # PEA_1000ppm

  from gas_sensor_data import load_baseline_series, load_response_series
  signal, timeslots, time_days = load_baseline_series(sensor_id=0)
  # pass directly to run_vocd(signal, timeslots)
"""

import argparse
import io
import os
import re
import zipfile

import numpy as np
import pandas as pd
import requests

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "gas_sensor_worner2025"
)

ZENODO_RECORD_ID = "15681119"
ZENODO_API_URL   = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

N_SENSORS   = 62
SENSOR_COLS = [f"R{i}" for i in range(1, N_SENSORS + 1)]

ANALYTE_LABELS = {
    "EtOH":            "Ethanol 5% v/v",
    "Diacetyl_0.1ppm": "Diacetyl 0.1 ppm",
    "Diacetyl_1ppm":   "Diacetyl 1 ppm",
    "PEA_200ppm":      "2-Phenylethanol 200 ppm",
    "PEA_1000ppm":     "2-Phenylethanol 1000 ppm",
}

BASELINE_N_ROWS = 20   # last N rows of stage 1 used for baseline (per paper)


# ══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_dataset(data_dir: str = DEFAULT_DATA_DIR,
                     force: bool = False) -> str:
    """
    Download the Zenodo archive and extract all daily CSV files.
    Uses the Zenodo REST API to discover the download URL.
    """
    os.makedirs(data_dir, exist_ok=True)

    day_dirs = [d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
                and re.search(r"\d", d)]
    if day_dirs and not force:
        print(f"  Dataset already cached ({len(day_dirs)} day folders) -> {data_dir}")
        return data_dir

    print(f"  Querying Zenodo record {ZENODO_RECORD_ID} ...")
    try:
        resp = requests.get(ZENODO_API_URL, timeout=30)
        resp.raise_for_status()
        meta  = resp.json()
        files = meta.get("files", [])
        if not files:
            raise ValueError("No files in Zenodo record metadata.")
    except Exception as e:
        raise RuntimeError(
            f"Could not reach Zenodo API: {e}\n"
            f"Download manually from https://doi.org/10.5281/zenodo.{ZENODO_RECORD_ID}"
            f"\nand extract day subfolders into: {data_dir}"
        )

    # Pick the largest ZIP file
    zips = [f for f in files if f.get("key", "").endswith(".zip")]
    if not zips:
        zips = files
    target   = sorted(zips, key=lambda f: f.get("size", 0), reverse=True)[0]
    url      = target["links"]["self"]
    size_mb  = target.get("size", 0) / 1e6

    print(f"  Downloading: {url}  ({size_mb:.1f} MB)")
    r = requests.get(url, timeout=300, stream=True)
    r.raise_for_status()
    raw = b"".join(r.iter_content(chunk_size=1 << 20))
    print(f"  Downloaded {len(raw)/1e6:.1f} MB")

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        # Detect ZIP structure: does it have a single top-level wrapper folder
        # (e.g. Dataset/day01/file.csv) or not (day01/file.csv)?
        # Count unique top-level components across all CSV members.
        csv_members = [m for m in zf.namelist() if m.endswith(".csv")]
        top_levels  = set(m.split("/")[0] for m in csv_members)
        # If there is exactly one top-level component and it contains no "."
        # it's a wrapper folder — strip it.  Otherwise keep all path components.
        strip_first = (len(top_levels) == 1 and "." not in list(top_levels)[0])

        extracted = 0
        for member in csv_members:
            parts = member.split("/")
            rel   = "/".join(parts[1:]) if strip_first else member
            if not rel:
                continue
            dest = os.path.join(data_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            extracted += 1

    print(f"  Extracted {extracted} CSV files -> {data_dir}")
    return data_dir


# ══════════════════════════════════════════════════════════════════════════════
#  PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _canonical_analyte(filename: str) -> str:
    """Map a raw filename to a ANALYTE_LABELS key."""
    name = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r"_\d+$", "", name)                         # strip replicate
    name = re.sub(r"2.?[Pp]henylethanol", "PEA", name)        # normalise PEA
    name = name.replace(" ", "_").replace("-", "_")
    return name


def parse_measurement(path: str) -> pd.DataFrame:
    """Read one measurement CSV. Returns DataFrame with normalised column names:
       R1..R62 (stripped of [Ohm]), T (temperature), H (humidity), cycle_stage.
    """
    with open(path, "r") as fh:
        first = fh.readline()
    sep = " " if (" " in first and "," not in first) else ","
    df  = pd.read_csv(path, sep=sep, decimal=".")
    df.columns = [c.strip() for c in df.columns]

    # Rename sensor columns: "R1[Ohm]" -> "R1", etc.
    rename = {}
    for col in df.columns:
        m = re.match(r"^(R\d+)\[.*\]$", col)
        if m:
            rename[col] = m.group(1)
    # Temperature / humidity
    for col in df.columns:
        if re.match(r"^T\d+\[", col):
            rename[col] = "T"
        elif re.match(r"^H\d+\[", col):
            rename[col] = "H"
    # Cycle stage — case-insensitive match
    for col in df.columns:
        if col.lower() in ("cycle_stage", "stage", "phase"):
            rename[col] = "cycle_stage"
            break
    if rename:
        df = df.rename(columns=rename)

    present = [c for c in SENSOR_COLS if c in df.columns]
    df[present] = df[present].apply(pd.to_numeric, errors="coerce")
    return df


def extract_baseline(df: pd.DataFrame, sensor_col: str) -> float:
    s1 = df[df["cycle_stage"] == 1][sensor_col].dropna()
    if len(s1) < 3:
        return np.nan
    return float(s1.iloc[-min(BASELINE_N_ROWS, len(s1)):].mean())


def extract_response(df: pd.DataFrame, sensor_col: str,
                     baseline: float) -> float:
    s3 = df[df["cycle_stage"] == 3][sensor_col].dropna()
    if len(s3) < 3 or np.isnan(baseline):
        return np.nan
    return abs(baseline - float(s3.mean()))


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE TABLE  —  one row per measurement
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_table(data_dir: str = DEFAULT_DATA_DIR,
                        force: bool = False) -> pd.DataFrame:
    """
    Walk all daily subfolders, extract per-sensor baseline + response,
    and cache as feature_table.parquet.

    Columns:
        day_idx, day_folder, analyte, replicate,
        R1_base..R62_base, R1_resp..R62_resp, T_mean, H_mean
    """
    cache = os.path.join(data_dir, "feature_table.parquet")
    if os.path.exists(cache) and not force:
        ft = pd.read_parquet(cache)
        # Invalidate if all-NaN (stale parse) or missing new columns
        probe = "R1_base"
        if (probe in ft.columns and ft[probe].isna().all()) or \
                "is_first_of_day" not in ft.columns:
            print("  Stale cache detected — rebuilding ...")
            os.remove(cache)
        else:
            print(f"  Loading cached feature table: {cache}")
            return ft

    # Natural sort: "Day 2" before "Day 10"
    def _day_key(name):
        nums = re.findall(r"\d+", name)
        return int(nums[0]) if nums else 0

    day_folders = sorted(
        [d for d in os.listdir(data_dir)
         if os.path.isdir(os.path.join(data_dir, d)) and re.search(r"\d", d)],
        key=_day_key
    )

    # Fallback: if CSVs were extracted flat (no subfolders), group them by the
    # day number embedded in the filename or treat each unique name prefix as a day.
    flat_csvs = []
    if not day_folders:
        flat_csvs = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
        if not flat_csvs:
            raise FileNotFoundError(
                f"No day subfolders or CSV files found in {data_dir}. "
                "Run download_dataset(force=True) to re-download."
            )
        print(f"  ⚠ Flat layout detected ({len(flat_csvs)} CSVs, no subfolders). "
              f"Re-extracting with corrected path logic ...")
        # Trigger a forced re-download/re-extract now that the fix is in place
        for f in flat_csvs:
            os.remove(os.path.join(data_dir, f))
        download_dataset(data_dir, force=True)
        day_folders = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d)) and re.search(r"\d", d)
        ])
        if not day_folders:
            raise FileNotFoundError(
                f"Still no day subfolders after re-extract. "
                f"Please inspect {data_dir} manually."
            )

    print(f"  Parsing {len(day_folders)} day folders ...")
    rows = []
    for day_idx, folder in enumerate(day_folders):
        folder_path = os.path.join(data_dir, folder)
        csv_files   = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

        # Sort by the actual start timestamp in each file — not filename
        def _first_timestamp(fname):
            path = os.path.join(folder_path, fname)
            try:
                with open(path, "r") as fh:
                    fh.readline()          # header
                    second = fh.readline() # first data row
                # Timestamp is first token: "HH:MM:SS"
                ts = second.split()[0]
                return ts
            except Exception:
                return "99:99:99"         # sort last if unreadable

        csv_files = sorted(csv_files, key=_first_timestamp)

        day_meas_idx = 0
        for csv_file in csv_files:
            analyte   = _canonical_analyte(csv_file)
            m         = re.search(r"_(\d+)\.csv$", csv_file)
            replicate = int(m.group(1)) if m else 1
            path      = os.path.join(folder_path, csv_file)

            try:
                df = parse_measurement(path)
            except Exception as e:
                print(f"    skip {csv_file}: {e}")
                continue

            row = {"day_idx":        day_idx,
                   "day_folder":     folder,
                   "analyte":        analyte,
                   "replicate":      replicate,
                   "day_meas_idx":   day_meas_idx,
                   "is_first_of_day": day_meas_idx == 0,
                   "start_time":     _first_timestamp(csv_file)}
            day_meas_idx += 1

            for scol in SENSOR_COLS:
                if scol not in df.columns:
                    row[f"{scol}_base"] = np.nan
                    row[f"{scol}_resp"] = np.nan
                    continue
                b = extract_baseline(df, scol)
                row[f"{scol}_base"] = b
                row[f"{scol}_resp"] = extract_response(df, scol, b)

            for aux in ["T", "H"]:
                if aux in df.columns:
                    row[f"{aux}_mean"] = float(
                        df[df["cycle_stage"] == 1][aux].dropna().mean()
                    )
                else:
                    row[f"{aux}_mean"] = np.nan

            rows.append(row)

    ft = pd.DataFrame(rows)

    # Build a continuous elapsed-time axis: each measurement gets a fractional
    # day value — day_idx + (day_meas_idx / 18) so within-day spacing is visible
    ft["elapsed"] = ft["day_idx"] + ft["day_meas_idx"] / 18.0

    ft.to_parquet(cache, index=False)
    print(f"  Feature table: {len(ft):,} measurements -> {cache}")
    return ft


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_baseline_series(sensor_id: int = 0,
                         analyte: str = None,
                         first_of_day: bool = False,
                         data_dir: str = DEFAULT_DATA_DIR,
                         auto_download: bool = True
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Baseline resistance time series for a given sensor.

    first_of_day=False (default): all ~700 measurements in chronological
    order — the densest possible drift signal for VOCD.

    first_of_day=True: one point per day (post-24h purge only) — 39 points,
    cleanest inter-day signal with within-day carryover removed.
    """
    if auto_download:
        download_dataset(data_dir)
    ft   = build_feature_table(data_dir)
    scol = f"R{sensor_id + 1}_base"
    if scol not in ft.columns:
        raise ValueError(f"sensor_id={sensor_id} out of range 0-{N_SENSORS-1}")

    sub = ft.copy()
    if analyte:
        sub = sub[sub["analyte"] == analyte]
    if first_of_day:
        sub = sub[sub["is_first_of_day"]]

    sub = sub.dropna(subset=[scol]).sort_values(
        ["day_idx", "day_meas_idx"]).reset_index(drop=True)

    signal    = sub[scol].values.astype(np.float64)
    time_days = sub["elapsed"].values.astype(np.float64)
    timeslots = np.arange(len(signal), dtype=np.float64)
    return signal, timeslots, time_days


def load_response_series(sensor_id: int = 0,
                         analyte: str = "Diacetyl_1ppm",
                         data_dir: str = DEFAULT_DATA_DIR,
                         auto_download: bool = True
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Response magnitude (delta-R) time series for a fixed (sensor, analyte)."""
    if auto_download:
        download_dataset(data_dir)
    ft   = build_feature_table(data_dir)
    scol = f"R{sensor_id + 1}_resp"
    sub  = ft[ft["analyte"] == analyte].dropna(subset=[scol]).sort_values(
        ["day_idx", "replicate"]).reset_index(drop=True)

    if sub.empty:
        raise ValueError(f"No data for analyte='{analyte}', sensor={sensor_id}")

    signal    = sub[scol].values.astype(np.float64)
    time_days = sub["day_idx"].values.astype(np.float64)
    timeslots = np.arange(len(signal), dtype=np.float64)
    return signal, timeslots, time_days


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_single_measurement(data_dir: str = DEFAULT_DATA_DIR,
                            day: int = 0,
                            analyte: str = "Diacetyl_1ppm",
                            replicate: int = 1,
                            sensors: list = None,
                            save: bool = True) -> None:
    """
    Fig A — Raw resistance time series for one measurement.
    Shows all 3 cycle stages: baseline / equilibration / sample.
    """
    import matplotlib.pyplot as plt

    if sensors is None:
        sensors = list(range(5))

    ft          = build_feature_table(data_dir)
    day_folders = sorted(ft["day_folder"].unique())
    folder      = os.path.join(data_dir, day_folders[min(day, len(day_folders)-1)])

    candidates = sorted(f for f in os.listdir(folder)
                        if f.endswith(".csv") and _canonical_analyte(f) == analyte)
    if not candidates:
        print(f"  No file for analyte={analyte} in day {day}")
        return
    path = os.path.join(folder, candidates[min(replicate-1, len(candidates)-1)])
    df   = parse_measurement(path)

    stage_colors = {1: "#d4e6f1", 2: "#d5f5e3", 3: "#fdebd0"}
    stage_labels = {1: "Stage 1: Baseline", 2: "Stage 2: Equilibration",
                    3: "Stage 3: Sample"}

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.subplots_adjust(top=0.87, bottom=0.12, left=0.09, right=0.97)

    for stage, color in stage_colors.items():
        idx = df.index[df["cycle_stage"] == stage]
        if len(idx):
            ax.axvspan(idx[0], idx[-1], color=color, alpha=0.5,
                       label=stage_labels[stage])

    cmap = plt.cm.tab10(np.linspace(0, 1, len(sensors)))
    for sid, color in zip(sensors, cmap):
        scol = f"R{sid+1}"
        if scol in df.columns:
            ax.plot(df.index, df[scol], color=color, lw=1.2, label=f"R{sid+1}")

    ax.set_xlabel("Sample index (~1 Hz)", fontsize=10)
    ax.set_ylabel("Resistance (Ohm)", fontsize=10)
    ax.set_title(
        f"Day {day+1} | {ANALYTE_LABELS.get(analyte, analyte)} | Replicate {replicate}\n"
        f"Raw sensor resistance — 3 cycle stages  (Worner et al. 2025)",
        fontsize=10
    )
    ax.legend(fontsize=8, loc="upper right", ncol=4, framealpha=0.85)
    ax.tick_params(labelsize=9)

    if save:
        out = os.path.join(data_dir, f"figA_raw_day{day}_{analyte}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Fig A saved -> {out}")


def plot_baseline_drift(data_dir: str = DEFAULT_DATA_DIR,
                        sensors: list = None,
                        analyte: str = None,
                        save: bool = True) -> None:
    """
    Fig B — Baseline resistance over time, all 700 measurements in
    chronological order.  x-axis = elapsed days (fractional, so within-day
    spacing is visible).  First-of-day points highlighted in a darker marker.

    One panel per sensor (2x5 grid for first 10 sensors).
    """
    import matplotlib.pyplot as plt

    ft = build_feature_table(data_dir)
    if sensors is None:
        sensors = list(range(10))

    sub = ft.copy()
    if analyte:
        sub = sub[sub["analyte"] == analyte]
    sub = sub.sort_values(["day_idx", "day_meas_idx"])

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=False)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.06, right=0.98,
                        hspace=0.45, wspace=0.35)
    tag = ANALYTE_LABELS.get(analyte, "all analytes") if analyte else "all analytes"
    fig.suptitle(
        f"Baseline resistance — all 700 measurements in chronological order | {tag}\n"
        f"Dots = all measurements  |  Red markers = first-of-day (post-24h purge)  "
        f"(Worner et al. 2025)",
        fontsize=10
    )

    cmap = plt.cm.tab10(np.linspace(0, 1, len(sensors)))
    for ax, sid, color in zip(axes.flat, sensors, cmap):
        scol = f"R{sid+1}_base"
        if scol not in sub.columns:
            continue

        all_meas = sub.dropna(subset=[scol])
        fod      = all_meas[all_meas["is_first_of_day"]]

        # All measurements as small connected dots
        ax.plot(all_meas["elapsed"], all_meas[scol],
                color=color, lw=0.7, alpha=0.6, zorder=1)
        ax.scatter(all_meas["elapsed"], all_meas[scol],
                   color=color, s=6, alpha=0.5, zorder=2, rasterized=True)

        # First-of-day highlighted in red
        ax.scatter(fod["elapsed"], fod[scol],
                   color="red", s=30, zorder=4, marker="D",
                   label="First-of-day")

        ax.set_title(f"R{sid+1}", fontsize=9)
        ax.set_xlabel("Elapsed days", fontsize=7)
        ax.set_ylabel("Baseline (Ohm)", fontsize=7)
        ax.tick_params(labelsize=7)

    # Single shared legend
    axes.flat[0].legend(fontsize=7, framealpha=0.85)

    if save:
        out = os.path.join(data_dir, f"figB_baseline_drift_{analyte or 'all'}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Fig B saved -> {out}")


def plot_response_drift(data_dir: str = DEFAULT_DATA_DIR,
                        sensors: list = None,
                        analyte: str = "Diacetyl_1ppm",
                        save: bool = True) -> None:
    """
    Fig C — Response magnitude (delta-R) over time.
    Same concentration, same sensor, changing response = sensitivity drift.
    """
    import matplotlib.pyplot as plt

    ft = build_feature_table(data_dir)
    if sensors is None:
        sensors = list(range(10))

    sub = ft[ft["analyte"] == analyte].copy()

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.subplots_adjust(top=0.87, bottom=0.12, left=0.09, right=0.97)

    cmap = plt.cm.tab10(np.linspace(0, 1, len(sensors)))
    for sid, color in zip(sensors, cmap):
        scol = f"R{sid+1}_resp"
        if scol not in sub.columns:
            continue
        s = sub.dropna(subset=[scol]).sort_values(["day_idx", "replicate"])
        ax.scatter(s["day_idx"], s[scol], color=color, s=8, alpha=0.4,
                   rasterized=True)
        daily = s.groupby("day_idx")[scol].median()
        ax.plot(daily.index, daily.values, color=color, lw=1.8, label=f"R{sid+1}")

    ax.set_xlabel("Measurement day index", fontsize=10)
    ax.set_ylabel("Delta-R response (Ohm)", fontsize=10)
    ax.set_title(
        f"Response magnitude (delta-R) over 12 months\n"
        f"Analyte: {ANALYTE_LABELS.get(analyte, analyte)}  |  "
        f"Sensitivity drift — same concentration, changing response  "
        f"(Worner et al. 2025)",
        fontsize=10
    )
    ax.legend(fontsize=7, loc="upper right", ncol=5, framealpha=0.85,
              markerscale=2)
    ax.tick_params(labelsize=9)

    if save:
        out = os.path.join(data_dir, f"figC_response_drift_{analyte}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Fig C saved -> {out}")


def plot_sensor_heatmap(data_dir: str = DEFAULT_DATA_DIR,
                        analyte: str = None,
                        save: bool = True) -> None:
    """
    Fig D — All 62 sensors x all days as a heatmap of daily median
    baseline resistance.  Reveals which sensors drift and which are stable.
    """
    import matplotlib.pyplot as plt

    ft  = build_feature_table(data_dir)
    sub = ft.copy()
    if analyte:
        sub = sub[sub["analyte"] == analyte]

    days   = sorted(sub["day_idx"].unique())
    matrix = np.full((N_SENSORS, len(days)), np.nan)

    for di, day in enumerate(days):
        d = sub[sub["day_idx"] == day]
        for sid in range(N_SENSORS):
            scol = f"R{sid+1}_base"
            if scol in d.columns:
                vals = d[scol].dropna()
                if len(vals):
                    matrix[sid, di] = vals.median()

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(top=0.90, bottom=0.09, left=0.07, right=0.97)

    vmin = np.nanpercentile(matrix, 2)
    vmax = np.nanpercentile(matrix, 98)
    im = ax.imshow(matrix, aspect="auto", origin="upper",
                   extent=[-0.5, len(days)-0.5, N_SENSORS-0.5, -0.5],
                   vmin=vmin, vmax=vmax, cmap="viridis")

    ax.set_xlabel("Measurement day index", fontsize=10)
    ax.set_ylabel("Sensor (R1-R62)", fontsize=10)
    ax.set_yticks(range(0, N_SENSORS, 5))
    ax.set_yticklabels([f"R{i+1}" for i in range(0, N_SENSORS, 5)], fontsize=7)
    tag = ANALYTE_LABELS.get(analyte, "all analytes") if analyte else "all analytes"
    ax.set_title(
        f"All 62 sensors — daily median baseline resistance (Ohm)\n"
        f"{tag}  |  Worner et al. 2025",
        fontsize=10
    )
    plt.colorbar(im, ax=ax, label="Baseline resistance (Ohm)",
                 fraction=0.015, pad=0.01)

    if save:
        out = os.path.join(data_dir,
                           f"figD_heatmap_{analyte or 'all'}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Fig D saved -> {out}")


def plot_single_sensor_closeup(data_dir: str = DEFAULT_DATA_DIR,
                               sensor_id: int = 0,
                               save: bool = True) -> None:
    """
    Fig E — One sensor: baseline drift (left) + response drift (right).
    Left panel shows all measurements (grey) vs first-of-day (colour) to
    illustrate within-day carryover noise vs the true drift signal.
    """
    import matplotlib.pyplot as plt

    ft      = build_feature_table(data_dir)
    scol_b  = f"R{sensor_id+1}_base"
    scol_r  = f"R{sensor_id+1}_resp"
    analytes = [a for a in ANALYTE_LABELS
                if not ft[ft["analyte"] == a][scol_b].dropna().empty]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.subplots_adjust(top=0.87, bottom=0.12, left=0.07, right=0.97, wspace=0.28)
    fig.suptitle(
        f"Sensor R{sensor_id+1} — baseline drift (left) & response drift (right)\n"
        f"Grey = all daily measurements  |  Colour lines = first-of-day baseline  "
        f"(Worner et al. 2025)",
        fontsize=10
    )

    cmap = plt.cm.Set1(np.linspace(0, 0.9, len(analytes)))

    # LEFT: all measurements chronologically, first-of-day in red
    ax = axes[0]
    all_b = ft.dropna(subset=[scol_b]).sort_values(["day_idx", "day_meas_idx"])
    fod   = all_b[all_b["is_first_of_day"]]

    ax.plot(all_b["elapsed"], all_b[scol_b],
            color="#aaaaaa", lw=0.7, alpha=0.6, zorder=1)
    ax.scatter(all_b["elapsed"], all_b[scol_b],
               color="#aaaaaa", s=6, alpha=0.5, zorder=2, rasterized=True,
               label="All measurements")
    ax.scatter(fod["elapsed"], fod[scol_b],
               color="red", s=35, zorder=4, marker="D",
               label="First-of-day (post-purge)")
    ax.set_xlabel("Elapsed days", fontsize=10)
    ax.set_ylabel("Baseline resistance (Ohm)", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.tick_params(labelsize=9)

    # RIGHT: response per analyte (first-of-day only — carryover affects ΔR too)
    ax = axes[1]
    for analyte, color in zip(analytes, cmap):
        sub = (ft[(ft["analyte"] == analyte) & (ft["is_first_of_day"])]
               .dropna(subset=[scol_r])
               .sort_values("day_idx"))
        if sub.empty:
            continue
        ax.plot(sub["day_idx"], sub[scol_r], color=color, lw=2.0,
                marker="o", ms=5, label=ANALYTE_LABELS.get(analyte, analyte))
    ax.set_xlabel("Measurement day index", fontsize=10)
    ax.set_ylabel("Delta-R response (Ohm)", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.tick_params(labelsize=9)

    if save:
        out = os.path.join(data_dir, f"figE_sensor{sensor_id+1}_closeup.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Fig E saved -> {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Worner et al. 2025 e-nose drift dataset loader & plotter"
    )
    parser.add_argument("--sensor",  type=int, default=0,
                        help="Sensor index 0-61 (default: 0)")
    parser.add_argument("--analyte", default=None,
                        help=f"Analyte key; one of: {list(ANALYTE_LABELS)}")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    print("Worner et al. 2025 -- E-nose Drift Dataset")
    print("=" * 50)

    download_dataset(args.data_dir, force=args.force_download)
    ft = build_feature_table(args.data_dir)

    print(f"\n  Measurements : {len(ft):,}")
    print(f"  Days         : {ft['day_idx'].nunique()}")
    print(f"  Analytes     : {sorted(ft['analyte'].unique())}")
    print(f"  Sensors      : {N_SENSORS}")

    scol = f"R{args.sensor+1}_base"
    if scol in ft.columns:
        s = ft[scol].dropna()
        x = ft.loc[s.index, "day_idx"].values.astype(float)
        rate = float(np.polyfit(x, s.values, 1)[0]) if len(s) > 1 else np.nan
        print(f"\n  R{args.sensor+1} baseline: "
              f"[{s.min():.1f}, {s.max():.1f}] Ohm  "
              f"mean={s.mean():.1f}  sigma={s.std():.1f}")
        print(f"  Overall drift rate: {rate:+.2f} Ohm/day")

    print("\nGenerating figures ...")
    analyte_plots = args.analyte or "Diacetyl_1ppm"

    print("  [A] Raw single measurement ...")
    plot_single_measurement(args.data_dir, day=0,
                            analyte=analyte_plots, save=True)

    print("  [B] Baseline drift over time ...")
    plot_baseline_drift(args.data_dir, sensors=list(range(10)),
                        analyte=args.analyte, save=True)

    print("  [C] Response drift over time ...")
    plot_response_drift(args.data_dir, sensors=list(range(10)),
                        analyte=analyte_plots, save=True)

    print("  [D] All-sensor heatmap ...")
    plot_sensor_heatmap(args.data_dir, analyte=args.analyte, save=True)

    print("  [E] Single-sensor close-up ...")
    plot_single_sensor_closeup(args.data_dir, sensor_id=args.sensor, save=True)

    print("\nDone.  Figures saved to:", args.data_dir)
    print("\nIntegration tip:")
    print("  from gas_sensor_data import load_baseline_series")
    print("  signal, timeslots, time_days = load_baseline_series(sensor_id=0)")
    print("  # pass to run_vocd(signal, timeslots)")


if __name__ == "__main__":
    main()