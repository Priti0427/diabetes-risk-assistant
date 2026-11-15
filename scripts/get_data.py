"""Download the Pima Indians Diabetes dataset into data/diabetes.csv.

The data is not committed to Git (see .gitignore); run this once after cloning.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "diabetes.csv"


def download(dest: Path = DEST, url: str = URL, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"{dest} already exists; use --force to re-download.")
        return dest
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    rows = sum(1 for _ in dest.open())
    print(f"Done: {rows} rows.")
    return dest


if __name__ == "__main__":
    download(force="--force" in sys.argv)
