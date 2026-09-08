#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POC_CHANNELS = {2, 4, 6, 8}


def read_lg_values(filename):
    """
    Read a CAEN Janus Spect_Timing list file and return LG values
    for PoC channels 2, 4, 6, and 8.
    """
    values = []

    with open(filename, "r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()

            # Skip comments, header, and blank lines
            if not line or line.startswith("//") or line.startswith("Brd"):
                continue

            fields = line.split()
            if len(fields) < 3:
                continue

            try:
                channel = int(fields[1])
                lg = int(fields[2])
            except ValueError:
                continue

            if channel in POC_CHANNELS:
                values.append((channel, lg))

    return values


def print_values(label, filename, values):
    print(f"\n{label}: {filename}")
    print("-" * (len(label) + len(str(filename)) + 2))

    for i, (channel, lg) in enumerate(values, start=1):
        print(f"{i:4d}  Ch {channel:02d}  LG = {lg}")

    print(f"\nFilled {len(values)} LG values from channels 2, 4, 6, 8.")


def main():
    parser = argparse.ArgumentParser(
        description="Compare PoC SiPM LG distributions for signal and background."
    )
    parser.add_argument("signal_file", help='Signal file, e.g. SIG.TXT')
    parser.add_argument("background_file", help='Background file, e.g. BKG.TXT')
    parser.add_argument(
        "-o", "--output",
        default="signal_background_LG.png",
        help="Output plot filename (default: signal_background_LG.png)"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of histogram bins (default: 60)"
    )
    args = parser.parse_args()

    sig_entries = read_lg_values(args.signal_file)
    bkg_entries = read_lg_values(args.background_file)

    if not sig_entries:
        raise RuntimeError(f"No PoC-channel LG values found in {args.signal_file}")
    if not bkg_entries:
        raise RuntimeError(f"No PoC-channel LG values found in {args.background_file}")

    print_values("SIG — Both CW fire", args.signal_file, sig_entries)
    print_values("BKG — Either CW fires", args.background_file, bkg_entries)

    sig = np.array([lg for _, lg in sig_entries])
    bkg = np.array([lg for _, lg in bkg_entries])

    # Common binning so the two distributions can be compared directly.
    xmin = min(sig.min(), bkg.min())
    xmax = max(sig.max(), bkg.max())

    # Add a small margin unless the range already spans the ADC limits.
    span = max(xmax - xmin, 1)
    lo = max(0, xmin - 0.02 * span)
    hi = min(4096, xmax + 0.02 * span)
    bins = np.linspace(lo, hi, args.bins + 1)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(
        bkg,
        bins=bins,
        histtype="stepfilled",
        alpha=0.45,
        linewidth=1.8,
        label="Either CW fires",
    )
    ax.hist(
        sig,
        bins=bins,
        histtype="stepfilled",
        alpha=0.55,
        linewidth=1.8,
        label="Both CW fire",
    )

    ax.set_xlabel("Counts", fontsize=14)
    ax.set_ylabel("SiPMs", fontsize=14)
    ax.set_title("PoC SiPM Low-Gain Response", fontsize=16, pad=12)

    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")

    print(f"\nSaved plot to: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
