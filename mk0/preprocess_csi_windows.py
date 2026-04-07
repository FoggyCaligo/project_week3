"""
preprocess_csi_windows.py

Purpose
-------
Convert raw CSI CSV logs (from esp-csi / csi_recv_router style output)
into fixed-size .npy windows + a manifest CSV for training.

Expected input
--------------
CSV files that contain at least:
- a `data` column: string like "[67,48,4,0,...]"
- optional `local_timestamp` column for sorting

The official esp-csi example prints CSI rows whose last `data` field stores
subcarrier CSI values. In the example README, each subcarrier is stored as:
[Imaginary part, Real part, Imaginary part, Real part, ...].

This script converts each packet's I/Q list into amplitude features:
amp = sqrt(real^2 + imag^2)

Typical usage
-------------
python preprocess_csi_windows.py ^
  --input-glob "raw_csi/empty/*.csv" ^
  --label 0 ^
  --output-dir "dataset/empty" ^
  --manifest-path "manifest.csv" ^
  --time-steps 120 ^
  --target-subcarriers 64 ^
  --stride 30

python preprocess_csi_windows.py ^
  --input-glob "raw_csi/human/*.csv" ^
  --label 1 ^
  --output-dir "dataset/human" ^
  --manifest-path "manifest.csv" ^
  --time-steps 120 ^
  --target-subcarriers 64 ^
  --stride 30
"""

import argparse
import ast
import csv
import glob
import math
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True, help="Glob for raw CSI CSV files")
    parser.add_argument("--label", required=True, type=int, choices=[0, 1], help="0=empty_normal, 1=human_motion")
    parser.add_argument("--output-dir", required=True, help="Directory to save .npy windows")
    parser.add_argument("--manifest-path", required=True, help="CSV file to append [path,label]")
    parser.add_argument("--time-steps", type=int, default=120, help="Packets per window")
    parser.add_argument("--target-subcarriers", type=int, default=64, help="Amplitude features per packet after pad/truncate")
    parser.add_argument("--stride", type=int, default=30, help="Window stride in packets")
    parser.add_argument("--max-files", type=int, default=0, help="Optional limit; 0 = no limit")
    parser.add_argument("--prefix", type=str, default="", help="Optional filename prefix")
    return parser.parse_args()


def parse_data_list(value: str) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, float) and math.isnan(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        arr = ast.literal_eval(text)
    except Exception:
        return None

    if not isinstance(arr, list):
        return None

    out = []
    for x in arr:
        try:
            out.append(int(x))
        except Exception:
            return None
    return out


def iq_to_amplitude(iq_list: List[int]) -> np.ndarray:
    """
    Official esp-csi README states each subcarrier is stored as:
    [imag0, real0, imag1, real1, ...]
    """
    if len(iq_list) < 2:
        return np.array([], dtype=np.float32)

    if len(iq_list) % 2 == 1:
        iq_list = iq_list[:-1]

    imag = np.asarray(iq_list[0::2], dtype=np.float32)
    real = np.asarray(iq_list[1::2], dtype=np.float32)
    amp = np.sqrt(real * real + imag * imag)
    return amp.astype(np.float32)


def fix_subcarrier_length(x: np.ndarray, target_subcarriers: int) -> np.ndarray:
    if x.shape[0] > target_subcarriers:
        return x[:target_subcarriers]
    if x.shape[0] < target_subcarriers:
        pad = np.zeros((target_subcarriers - x.shape[0],), dtype=np.float32)
        return np.concatenate([x, pad], axis=0)
    return x


def load_one_csv(csv_path: str, target_subcarriers: int) -> np.ndarray:
    df = pd.read_csv(csv_path)

    if "data" not in df.columns:
        raise ValueError(f"'data' column not found: {csv_path}")

    if "type" in df.columns:
        df = df[df["type"] == "CSI_DATA"].copy()

    if "local_timestamp" in df.columns:
        df = df.sort_values("local_timestamp").reset_index(drop=True)

    packets = []
    dropped = 0

    for _, row in df.iterrows():
        iq_list = parse_data_list(row["data"])
        if iq_list is None:
            dropped += 1
            continue

        amp = iq_to_amplitude(iq_list)
        if amp.size == 0:
            dropped += 1
            continue

        amp = fix_subcarrier_length(amp, target_subcarriers)
        packets.append(amp)

    if len(packets) == 0:
        raise ValueError(f"No usable CSI packets found in {csv_path}")

    packets = np.stack(packets, axis=0)  # (num_packets, subcarriers)
    print(f"[load] {csv_path} -> packets={packets.shape[0]}, subcarriers={packets.shape[1]}, dropped={dropped}")
    return packets


def sliding_windows(arr: np.ndarray, time_steps: int, stride: int) -> List[np.ndarray]:
    """
    arr shape: (num_packets, num_subcarriers)
    returns list of windows with shape: (time_steps, num_subcarriers)
    """
    n = arr.shape[0]
    out = []

    if n < time_steps:
        return out

    for start in range(0, n - time_steps + 1, stride):
        end = start + time_steps
        out.append(arr[start:end].copy())
    return out


def append_manifest(manifest_path: Path, rows: List[List[str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = manifest_path.exists()

    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["path", "label"])
        writer.writerows(rows)


def main():
    args = parse_args()

    input_files = sorted(glob.glob(args.input_glob))
    if not input_files:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    if args.max_files > 0:
        input_files = input_files[:args.max_files]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest_path)
    rows_to_append = []
    saved_count = 0

    for file_idx, csv_path in enumerate(input_files):
        packet_arr = load_one_csv(csv_path, args.target_subcarriers)
        windows = sliding_windows(packet_arr, args.time_steps, args.stride)

        stem = Path(csv_path).stem
        if not windows:
            print(f"[skip] not enough packets for a full window: {csv_path}")
            continue

        for win_idx, win in enumerate(windows):
            # win shape: (time_steps, target_subcarriers)
            name_parts = []
            if args.prefix:
                name_parts.append(args.prefix)
            name_parts.extend([stem, f"{file_idx:03d}", f"{win_idx:04d}"])
            out_name = "_".join(name_parts) + ".npy"

            out_path = (output_dir / out_name).resolve()
            np.save(out_path, win.astype(np.float32))
            rows_to_append.append([str(out_path), str(args.label)])
            saved_count += 1

        print(f"[window] {csv_path} -> {len(windows)} samples")

    append_manifest(manifest_path, rows_to_append)
    print(f"[done] saved {saved_count} windows")
    print(f"[done] manifest appended: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
