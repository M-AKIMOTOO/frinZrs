#!/usr/bin/env python3
"""Convert .npy/.npz file to plain text.

Examples:
  python npyz2txt.py data.npy
  python npyz2txt.py data.npy out.txt
  python npyz2txt.py data.npz out.txt --array table
  python npyz2txt.py data.npy out.txt --index --delimiter ","
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    epilog = """Examples:
  npyz2txt.py example.npy
  npyz2txt.py example.npy out.txt
  npyz2txt.py example.npy out.csv --delimiter ","
  npyz2txt.py example.npz out.txt --array arr0
  npyz2txt.py example.npz --array arr0 --index
"""
    p = argparse.ArgumentParser(
        description="Convert NPY/NPZ to text table.",
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("input_file", type=Path, help="Input .npy or .npz path")
    p.add_argument(
        "txt_file",
        nargs="?",
        type=Path,
        default=None,
        help="Output text path (default: stdout)",
    )
    p.add_argument(
        "--array",
        default=None,
        help="Array key for .npz input (required when multiple arrays exist)",
    )
    p.add_argument(
        "--delimiter",
        default="\t",
        help=r"Column delimiter (default: tab '\t')",
    )
    p.add_argument(
        "--index",
        action="store_true",
        help="Add leading index column",
    )
    p.add_argument(
        "--no-header",
        action="store_true",
        help="Do not write header line",
    )
    return p


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def scalar_to_text(v: object) -> str:
    x = np.asarray(v).item() if np.asarray(v).shape == () else v
    if isinstance(x, float):
        return f"{x:.8g}"
    if isinstance(x, complex):
        return f"{x.real:.8g}{x.imag:+.8g}j"
    return str(x)


def write_lines(path: Path | None, lines: Iterable[str]) -> None:
    if path is None:
        for line in lines:
            print(line)
        return
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def structured_rows(
    arr: np.ndarray, delimiter: str, use_index: bool, no_header: bool
) -> Sequence[str]:
    names = list(arr.dtype.names or [])
    lines: List[str] = []
    if not no_header:
        header_cols = (["index"] if use_index else []) + names
        lines.append("# " + delimiter.join(header_cols))
    for i in range(arr.shape[0]):
        row = [scalar_to_text(arr[name][i]) for name in names]
        if use_index:
            row = [str(i)] + row
        lines.append(delimiter.join(row))
    return lines


def regular_rows(
    arr: np.ndarray, delimiter: str, use_index: bool, no_header: bool
) -> Sequence[str]:
    lines: List[str] = []
    if arr.ndim == 1:
        if not no_header:
            cols = ["index", "value"] if use_index else ["value"]
            lines.append("# " + delimiter.join(cols))
        for i, v in enumerate(arr):
            row = [scalar_to_text(v)]
            if use_index:
                row = [str(i)] + row
            lines.append(delimiter.join(row))
        return lines

    if arr.ndim == 2:
        ncol = arr.shape[1]
        if not no_header:
            cols = [f"col{j}" for j in range(ncol)]
            if use_index:
                cols = ["index"] + cols
            lines.append("# " + delimiter.join(cols))
        for i, row_v in enumerate(arr):
            row = [scalar_to_text(v) for v in row_v]
            if use_index:
                row = [str(i)] + row
            lines.append(delimiter.join(row))
        return lines

    raise ValueError(
        f"Unsupported ndim={arr.ndim}. This tool supports 1D, 2D, or structured 1D arrays."
    )


def load_array(path: Path, array_key: str | None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix != ".npz":
        raise ValueError(f"not a .npy/.npz file: {path}")

    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.files)
        if not keys:
            raise ValueError("npz archive has no arrays")
        if array_key is None:
            if len(keys) == 1:
                return np.asarray(npz[keys[0]])
            joined = ", ".join(keys)
            raise ValueError(
                f"npz has multiple arrays ({joined}); specify one with --array"
            )
        if array_key not in keys:
            joined = ", ".join(keys)
            raise KeyError(f"array '{array_key}' not found. available: {joined}")
        return np.asarray(npz[array_key])


def main() -> int:
    if len(sys.argv) == 1:
        parser = build_parser()
        parser.print_help()
        return 0

    args = parse_args(sys.argv[1:])
    if not args.input_file.exists():
        print(f"ERROR: file not found: {args.input_file}", file=sys.stderr)
        return 1
    try:
        arr = load_array(args.input_file, args.array)
    except Exception as e:
        print(f"ERROR: failed to load input: {e}", file=sys.stderr)
        return 1

    try:
        if arr.dtype.names:
            lines = structured_rows(arr, args.delimiter, args.index, args.no_header)
        else:
            lines = regular_rows(arr, args.delimiter, args.index, args.no_header)
        write_lines(args.txt_file, lines)
    except Exception as e:
        print(f"ERROR: conversion failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
