#!/usr/bin/env python3
"""Simple NPY/NPZ viewer.

Usage:
  python npyzviewer.py data.npy
  python npyzviewer.py data.npz --array table
  python npyzviewer.py data.npy x_key y_key
  python npyzviewer.py data.npz x_key y_key --array table

If only input is given, this prints header-like metadata.
If `x_key` and `y_key` are also given, it plots y vs x.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    epilog = """Examples:
  npyzviewer.py example.npy
  npyzviewer.py example.npy index 0
  npyzviewer.py example.npz --list-arrays
  npyzviewer.py example.npz --array arr0
  npyzviewer.py example.npz index real --array arr0
"""
    parser = argparse.ArgumentParser(
        description="Show NPY/NPZ header information and optionally plot two keys.",
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input_file", type=Path, help="Path to .npy or .npz file")
    parser.add_argument("x_key", nargs="?", help="X-axis key (field name or column index)")
    parser.add_argument("y_key", nargs="?", help="Y-axis key (field name or column index)")
    parser.add_argument(
        "--array",
        default=None,
        help="Array key for .npz input.",
    )
    parser.add_argument(
        "--list-arrays",
        action="store_true",
        help="List arrays in .npz and exit.",
    )
    return parser


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = _build_parser()
    if hasattr(parser, "parse_intermixed_args"):
        return parser.parse_intermixed_args(argv)
    return parser.parse_args(argv)


def _print_summary(path: Path, arr: np.ndarray, selected_array: str | None) -> None:
    print(f"file: {path}")
    if selected_array is not None:
        print(f"array: {selected_array}")
    print(f"shape: {arr.shape}")
    print(f"dtype: {arr.dtype}")
    print(f"ndim: {arr.ndim}")
    print(f"size: {arr.size}")
    print(f"itemsize: {arr.itemsize}")
    print(f"nbytes: {arr.nbytes}")

    if arr.dtype.names:
        print("keys:")
        for name in arr.dtype.names:
            print(f"  - {name}: {arr.dtype[name]}")
    elif arr.ndim == 1:
        print("keys:")
        print("  - index")
        print("  - 0")
    elif arr.ndim >= 2:
        print("keys (column index):")
        for i in range(arr.shape[1]):
            print(f"  - {i}")
        print("  - index")


def _extract_1d(arr: np.ndarray, key: str) -> np.ndarray:
    if arr.dtype.names:
        if key == "index":
            return np.arange(arr.shape[0], dtype=np.float64)
        if key not in arr.dtype.names:
            raise KeyError(
                f"Unknown key '{key}'. Available keys: {', '.join(arr.dtype.names)}"
            )
        out = np.asarray(arr[key])
        if out.ndim != 1:
            raise ValueError(
                f"Key '{key}' is not 1D (shape={out.shape}). This tool expects 1D fields."
            )
        return out

    if key == "index":
        return np.arange(arr.shape[0], dtype=np.float64)

    try:
        col = int(key)
    except ValueError as exc:
        raise KeyError(
            f"Key '{key}' is invalid for non-structured array. Use 'index' or column number."
        ) from exc

    if arr.ndim == 1:
        if col != 0:
            raise IndexError("1D array only has column 0.")
        return arr

    if arr.ndim == 2:
        if not (0 <= col < arr.shape[1]):
            raise IndexError(f"Column {col} out of range [0, {arr.shape[1] - 1}].")
        return arr[:, col]

    raise ValueError(
        f"Unsupported ndim={arr.ndim} for non-structured array. Expected 1D or 2D."
    )


def _plot_xy(arr: np.ndarray, x_key: str, y_key: str) -> None:
    x = _extract_1d(arr, x_key)
    y = _extract_1d(arr, y_key)

    if len(x) != len(y):
        raise ValueError(f"Length mismatch: len({x_key})={len(x)}, len({y_key})={len(y)}")

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, lw=1)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(f"{y_key} vs {x_key}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _load_array(
    path: Path, array_key: str | None, list_arrays: bool
) -> tuple[np.ndarray, str | None]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        return arr, None
    if suffix != ".npz":
        raise ValueError(f"not a .npy/.npz file: {path}")

    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.files)
        if not keys:
            raise ValueError("npz archive has no arrays")
        print("npz keys:")
        for key in keys:
            v = np.asarray(npz[key])
            print(f"  - {key}: shape={v.shape}, dtype={v.dtype}")
        if list_arrays:
            raise ValueError("__LIST_ONLY__")
        if array_key is None:
            if len(keys) == 1:
                array_key = keys[0]
            else:
                raise ValueError("__MULTI_NO_ARRAY__")
        if array_key not in keys:
            joined = ", ".join(keys)
            raise KeyError(f"array '{array_key}' not found. available: {joined}")
        arr = np.asarray(npz[array_key])
        return arr, array_key


def main() -> int:
    if len(sys.argv) == 1:
        parser = _build_parser()
        parser.print_help()
        return 0

    args = _parse_args(sys.argv[1:])
    path = args.input_file

    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 1

    try:
        arr, selected_array = _load_array(path, args.array, args.list_arrays)
    except Exception as exc:
        msg = str(exc)
        if msg == "__LIST_ONLY__":
            return 0
        if msg == "__MULTI_NO_ARRAY__":
            print("Multiple arrays detected. Use --array <key> to select one.", file=sys.stderr)
            return 0
        print(f"ERROR: failed to load input: {exc}", file=sys.stderr)
        return 1

    _print_summary(path, arr, selected_array)

    has_x = args.x_key is not None
    has_y = args.y_key is not None
    if has_x != has_y:
        print("ERROR: specify both x_key and y_key, or neither.", file=sys.stderr)
        return 1

    if has_x and has_y:
        try:
            _plot_xy(arr, args.x_key, args.y_key)
        except Exception as exc:
            print(f"ERROR: failed to plot: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
