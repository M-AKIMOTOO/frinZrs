#!/usr/bin/env python3
"""Convert one or more .npy files to a single .npz archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    epilog = """Examples:
  npy2npz.py example.npy
  npy2npz.py example.npy -o example.npz --key data
  npy2npz.py example1.npy example2.npy -o bundle.npz
  npy2npz.py example1.npy example2.npy --no-compress
"""
    p = argparse.ArgumentParser(
        description="Convert one or more NPY files to NPZ.",
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("npy_files", nargs="+", type=Path, help="Input .npy path(s)")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .npz path (default: input name with .npz for single input, *_bundle.npz for multiple)",
    )
    p.add_argument(
        "--key",
        default="data",
        help="Array key when single input is given (default: data)",
    )
    p.add_argument(
        "--no-compress",
        action="store_true",
        help="Use np.savez (uncompressed) instead of np.savez_compressed",
    )
    return p


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def field_key_signature(arr: np.ndarray) -> tuple[str, ...]:
    if arr.dtype.names:
        return tuple(arr.dtype.names)
    return tuple()


def signature_text(sig: tuple[str, ...]) -> str:
    if not sig:
        return "(non-structured array: no field keys)"
    return ", ".join(sig)


def unique_name(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 2
    while True:
        cand = f"{base}_{i}"
        if cand not in used:
            used.add(cand)
            return cand
        i += 1


def main() -> int:
    if len(sys.argv) == 1:
        parser = build_parser()
        parser.print_help()
        return 0

    args = parse_args(sys.argv[1:])

    npy_files: list[Path] = args.npy_files
    if not npy_files:
        print("ERROR: no input files", file=sys.stderr)
        return 1

    for path in npy_files:
        if path.suffix.lower() != ".npy":
            print(f"ERROR: not a .npy file: {path}", file=sys.stderr)
            return 1
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            return 1

    if args.output is not None:
        out = args.output
    elif len(npy_files) == 1:
        out = npy_files[0].with_suffix(".npz")
    else:
        out = npy_files[0].with_name(f"{npy_files[0].stem}_bundle.npz")

    if out.suffix.lower() != ".npz":
        print(f"ERROR: output must end with .npz: {out}", file=sys.stderr)
        return 1

    loaded: list[tuple[Path, np.ndarray, tuple[str, ...]]] = []
    for path in npy_files:
        try:
            arr = np.load(path, allow_pickle=False)
        except Exception as exc:
            print(f"ERROR: failed to load npy: {path}: {exc}", file=sys.stderr)
            return 1
        loaded.append((path, arr, field_key_signature(arr)))

    ref_sig = loaded[0][2]
    mismatches = [(p, s) for (p, _, s) in loaded[1:] if s != ref_sig]
    if mismatches:
        print("ERROR: field keys are not consistent across input npy files.", file=sys.stderr)
        print(
            f"reference ({loaded[0][0]}): {signature_text(ref_sig)}",
            file=sys.stderr,
        )
        print("mismatched files:", file=sys.stderr)
        for p, sig in mismatches:
            print(f"  - {p}: {signature_text(sig)}", file=sys.stderr)
        return 1

    arrays_for_npz: dict[str, np.ndarray] = {}
    if len(loaded) == 1:
        arrays_for_npz[args.key] = loaded[0][1]
    else:
        used: set[str] = set()
        for p, arr, _ in loaded:
            k = unique_name(p.stem, used)
            arrays_for_npz[k] = arr

    try:
        if args.no_compress:
            np.savez(out, **arrays_for_npz)
        else:
            np.savez_compressed(out, **arrays_for_npz)
    except Exception as exc:
        print(f"ERROR: failed to write npz: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote: {out} (arrays={len(arrays_for_npz)})")
    for k, arr in arrays_for_npz.items():
        print(f"  - {k}: shape={arr.shape}, dtype={arr.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
