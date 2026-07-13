"""Training and evaluation script for GTSRB subset classifiers."""

from __future__ import annotations

import argparse

from data import load_prepared, summarize_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load GTSRB 5-class subset (training comes in later commits)."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download GTSRB Final Training Images into ./data if missing",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset root (default: ./data)",
    )
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, names = load_prepared(
        data_dir=args.data_dir,
        download=args.download,
    )

    print(summarize_split(y_train, y_val, names))
    print()
    print(f"X_train shape: {X_train.shape}  dtype={X_train.dtype}")
    print(f"X_val   shape: {X_val.shape}  dtype={X_val.dtype}")
    print(f"Pixel range: [{X_train.min():.3f}, {X_train.max():.3f}]")


if __name__ == "__main__":
    main()
