#!/usr/bin/env python3
"""
Compare total event energies from two Janus DT5202 Spect_Timing files.

Usage:
    python compareEnergies.py BKG.txt SIG.txt

For each event:
    LGtot = sum of LG counts over channels 6-63
    HGtot = sum of HG counts over channels 6-63

Produces two side-by-side histograms:
    LEFT:  LGtot for SIG vs BKG
    RIGHT: HGtot for SIG vs BKG
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def read_event_totals(filename):
    """Return arrays of LGtot and HGtot, one entry per event."""
    lg_totals = []
    hg_totals = []

    current_lg = 0.0
    current_hg = 0.0
    have_event = False

    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()

            if not line or line.startswith("//") or line.startswith("Brd"):
                continue

            fields = line.split()
            if len(fields) < 4:
                continue

            try:
                ch = int(fields[1])
                lg = float(fields[2])
                hg = float(fields[3])
            except (ValueError, IndexError):
                continue

            # First row of each event has timestamps, TrgID, and NChs.
            is_event_start = len(fields) >= 10

            if is_event_start:
                if have_event:
                    lg_totals.append(current_lg)
                    hg_totals.append(current_hg)

                current_lg = 0.0
                current_hg = 0.0
                have_event = True

            if have_event and 6 <= ch <= 63:
                current_lg += lg
                current_hg += hg

    if have_event:
        lg_totals.append(current_lg)
        hg_totals.append(current_hg)

    return np.asarray(lg_totals), np.asarray(hg_totals)


def common_bins(a, b, nbins=50):
    """Common histogram bins spanning both samples."""
    combined = np.concatenate([a, b])
    xmin = float(np.min(combined))
    xmax = float(np.max(combined))

    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5

    return np.linspace(xmin, xmax, nbins + 1)


def main():
    if len(sys.argv) != 3:
        print("Usage: python compareEnergies.py BKG.txt SIG.txt")
        sys.exit(1)

    bkg_file = sys.argv[1]
    sig_file = sys.argv[2]

    for filename in (bkg_file, sig_file):
        if not Path(filename).is_file():
            print(f"ERROR: file not found: {filename}")
            sys.exit(1)

    bkg_lg, bkg_hg = read_event_totals(bkg_file)
    sig_lg, sig_hg = read_event_totals(sig_file)

    if len(bkg_lg) == 0 or len(sig_lg) == 0:
        print("ERROR: no events found in one or both files")
        sys.exit(1)

    print(f"BKG: {len(bkg_lg)} events")
    print(f"SIG: {len(sig_lg)} events")
    print()
    print(f"BKG LGtot mean = {np.mean(bkg_lg):.1f}")
    print(f"SIG LGtot mean = {np.mean(sig_lg):.1f}")
    print(f"BKG HGtot mean = {np.mean(bkg_hg):.1f}")
    print(f"SIG HGtot mean = {np.mean(sig_hg):.1f}")

    lg_bins = common_bins(bkg_lg, sig_lg)
    hg_bins = common_bins(bkg_hg, sig_hg)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].hist(bkg_lg, bins=lg_bins, histtype="step", linewidth=2, label="BKG")
    axes[0].hist(sig_lg, bins=lg_bins, histtype="step", linewidth=2, label="SIG")
    axes[0].set_xlabel("LGtot = sum of LG counts, channels 6-63")
    axes[0].set_ylabel("Events")
    axes[0].set_title("Total low-gain energy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].hist(bkg_hg, bins=hg_bins, histtype="step", linewidth=2, label="BKG")
    axes[1].hist(sig_hg, bins=hg_bins, histtype="step", linewidth=2, label="SIG")
    axes[1].set_xlabel("HGtot = sum of HG counts, channels 6-63")
    axes[1].set_ylabel("Events")
    axes[1].set_title("Total high-gain energy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle("Signal vs. background event energy", fontsize=14)
    fig.tight_layout()

    outfile = "compareEnergies.png"
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    print(f"\nSaved {outfile}")

    plt.show()


if __name__ == "__main__":
    main()
