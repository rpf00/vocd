"""
Download the datasets of the VOCD manuscript from Zenodo and unpack them here.

    python data/fetch_data.py                          # all archives
    python data/fetch_data.py isfet_data env_monitoring   # only some of them
    python data/fetch_data.py --keep-zips

Record: https://doi.org/10.5281/zenodo.23388462  (CC BY 4.0)

Standard library only.  Archives are unpacked into this folder (``data/``);
see ``data/README.md`` for the layout the experiment scripts expect.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile

RECORD = "23388462"
BASE = f"https://zenodo.org/records/{RECORD}/files"
ARCHIVES = ["isfet_data", "env_monitoring", "gas_sensor_worner2025",
            "flashcrash", "bond", "well-log"]
HERE = os.path.dirname(os.path.abspath(__file__))
# folder inside the archive -> folder the experiment scripts expect
RENAME = {"isfet_data": "isfet"}


def _wanted(member: str) -> bool:
    """Skip macOS archive clutter."""
    parts = member.split("/")
    return parts[0] != "__MACOSX" and parts[-1] != ".DS_Store"


def download(name: str, dest: str) -> None:
    url = f"{BASE}/{name}.zip?download=1"
    print(f"downloading {name}.zip ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "vocd-fetch-data"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    if total:
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"archives to fetch (default: all of {ARCHIVES})")
    ap.add_argument("--keep-zips", action="store_true", help="do not delete the .zip files")
    args = ap.parse_args()

    names = args.names or ARCHIVES
    unknown = [n for n in names if n not in ARCHIVES]
    if unknown:
        print(f"unknown archive(s) {unknown}; choose from {ARCHIVES}", file=sys.stderr)
        return 2

    for name in names:
        zpath = os.path.join(HERE, f"{name}.zip")
        try:
            download(name, zpath)
        except Exception as e:
            print(f"! could not download {name}.zip ({e}).\n"
                  f"  Download it by hand from https://doi.org/10.5281/zenodo.{RECORD} "
                  f"and unzip it into {HERE}", file=sys.stderr)
            return 1
        with zipfile.ZipFile(zpath) as zf:
            members = [m for m in zf.namelist() if _wanted(m)]
            zf.extractall(HERE, members=members)
            top = sorted({m.split("/")[0] for m in members})
        for i, t in enumerate(top):
            new = RENAME.get(t)
            if new:
                src, dst = os.path.join(HERE, t), os.path.join(HERE, new)
                shutil.copytree(src, dst, dirs_exist_ok=True)
                shutil.rmtree(src)
                top[i] = new
        print(f"  unpacked into data/: {', '.join(top)}")
        if not args.keep_zips:
            os.remove(zpath)

    print("\ndone. Expected layout: see data/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
