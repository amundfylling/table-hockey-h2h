#!/usr/bin/env python3
"""Convert all parquet files in data/ to JSON files in public/data/.

Usage:
    python scripts/convert_parquet_to_json.py

The script reads each `.parquet` file in the `data` directory and writes
an identically named `.json` file to `public/data` using `orient='records'`.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "public" / "data"


def convert_parquet_to_json() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for parquet_path in DATA_DIR.glob("*.parquet"):
        df = pd.read_parquet(parquet_path)
        json_path = OUT_DIR / f"{parquet_path.stem}.json"
        df.to_json(json_path, orient="records", date_format="iso")
        print(f"Wrote {json_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    convert_parquet_to_json()
