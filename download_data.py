"""
Download CMD 2D maps for a given field and suite.

Usage:
  python download_data.py --field Mcdm --suite IllustrisTNG
  python download_data.py --field Mcdm --suite IllustrisTNG --suite SIMBA  # for cross-suite
  python download_data.py --all_fields --suite IllustrisTNG

Data source: https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/
"""

import argparse
import os
import subprocess
import sys

BASE_URL = (
    "https://users.flatironinstitute.org/~fvillaescusa/priv/"
    "DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data"
)

ALL_FIELDS = ["Mgas", "Vgas", "T", "P", "Z", "HI", "ne", "B", "MgFe", "Mcdm", "Vcdm", "Mstar", "Mtot"]
NBODY_FIELDS = ["Mtot"]


def download_file(url, dest):
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading: {url}")
    print(f"  -> {dest}")
    result = subprocess.run(
        ["curl", "-L", "--progress-bar", "-o", dest, url],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  FAILED: {url}")
        if os.path.exists(dest):
            os.remove(dest)


def main():
    parser = argparse.ArgumentParser(description="Download CMD 2D maps")
    parser.add_argument("--field", nargs="+", default=["Mcdm"],
                        help="Fields to download (e.g. Mcdm T Mgas)")
    parser.add_argument("--all_fields", action="store_true")
    parser.add_argument("--suite", nargs="+", default=["IllustrisTNG"],
                        help="Simulation suites (e.g. IllustrisTNG SIMBA)")
    parser.add_argument("--set_name", default="LH", help="Set name (LH, CV, 1P)")
    parser.add_argument("--output_dir", default="./cmd_data")
    args = parser.parse_args()

    fields = ALL_FIELDS if args.all_fields else args.field
    os.makedirs(args.output_dir, exist_ok=True)

    for suite in args.suite:
        print(f"\n{'=' * 60}")
        print(f"Suite: {suite}")
        print(f"{'=' * 60}")

        suite_subdir = "Nbody" if suite.startswith("Nbody") else suite

        params_name = f"params_{args.set_name}_{suite}.txt"
        params_url = f"{BASE_URL}/{suite_subdir}/{params_name}"
        download_file(params_url, os.path.join(args.output_dir, params_name))

        available = NBODY_FIELDS if suite.startswith("Nbody") else fields
        for field in available:
            if field == "B" and suite != "IllustrisTNG":
                print(f"  Skipping {field} (only available for IllustrisTNG)")
                continue
            fname = f"Maps_{field}_{suite}_{args.set_name}_z=0.00.npy"
            url = f"{BASE_URL}/{suite_subdir}/{fname}"
            download_file(url, os.path.join(args.output_dir, fname))

    print(f"\nDone. Data saved to {args.output_dir}/")
    print("\nTo start training:")
    print(f"  python train.py --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()
