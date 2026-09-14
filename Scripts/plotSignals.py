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
        default=200,
        help="Number of histogram bins (default: 200)"
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

    # Plot range: 0-1000 counts, with all values >=1000 folded into
    # the final visible bin as overflow.
    plot_xmax = 1000.0
    eps = 1e-6

    sig_plot = np.minimum(sig.astype(float), plot_xmax - eps)
    bkg_plot = np.minimum(bkg.astype(float), plot_xmax - eps)

    # Common binning for direct comparison.
    bins = np.linspace(0, plot_xmax, args.bins + 1)

    # Normalize each distribution to unit area.
    sig_weights = np.ones_like(sig_plot, dtype=float) / len(sig_plot)
    bkg_weights = np.ones_like(bkg_plot, dtype=float) / len(bkg_plot)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(
        bkg_plot,
        bins=bins,
        weights=bkg_weights,
        histtype="step",
        linewidth=2.0,
        label="Either CW fires",
    )
    ax.hist(
        sig_plot,
        bins=bins,
        weights=sig_weights,
        histtype="step",
        linewidth=2.0,
        label="Both CW fire",
    )

    ax.set_xlabel("Counts", fontsize=14)
    ax.set_ylabel("Normalized SiPMs", fontsize=14)
    ax.set_xlim(0, 1000)
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
