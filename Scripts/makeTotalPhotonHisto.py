#!/usr/bin/env python3
"""
makeTotalPhotonHisto.py

Total photon yield per event from a simulated SiPM hit file.

Usage
-----
    python makeTotalPhotonHisto.py TEXTFILE [-n 100000]

Input format
------------
One event is 9 non-blank lines, events separated by a blank line:

    line 1      8 numbers    event header (truth / summary quantities)
    lines 2-5   16 numbers   per-SiPM times, rows A B C D, -999999 = no hit
    lines 6-9   16 numbers   per-SiPM photon counts, rows A B C D

Within a row the 16 columns are SiPM 0 .. 15, so the four rows give A0..A15,
B0..B15, C0..C15, D0..D15 -- 64 SiPMs in total.

What is summed
--------------
All 64 SiPMs except

    columns 0, 1, 2 of every row   ->  A0 A1 A2 B0 B1 B2 C0 C1 C2 D0 D1 D2
    C4 and C12

i.e. 64 - 12 - 2 = 50 SiPMs.

Output
------
    TEXTFILE_totalPhotons.png   (and the matching .pdf; --no-pdf skips it)
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
ROWS = ["A", "B", "C", "D"]          # the four rows, in file order
NCOL = 16                            # SiPMs 0 .. 15 in each row
NSIPM = len(ROWS) * NCOL             # 64

NEVENTS = 100000                     # events read by default

# SiPMs left out of the sum
EXCLUDE_COLS = [0, 1, 2]             # these columns in every row
EXCLUDE_CELLS = [("C", 4), ("C", 12)]

NHEAD = 8                            # numbers on the event header line
NTIME_ROWS = 4                       # rows of per-SiPM times
NCOUNT_ROWS = 4                      # rows of per-SiPM photon counts

# thinnest line that still renders solidly: 0.5 pt is one pixel at 150 dpi
LINEWIDTH = 0.8
DPI = 150


# ----------------------------------------------------------------------------
def build_mask():
    """(4, 16) boolean array: True where the SiPM is summed."""
    keep = np.ones((len(ROWS), NCOL), dtype=bool)
    for c in EXCLUDE_COLS:
        keep[:, c] = False
    for row, c in EXCLUDE_CELLS:
        keep[ROWS.index(row), c] = False
    return keep


KEEP = build_mask()


def sipm_names(keep, wanted=True):
    """['A3', 'A4', ...] for the cells where keep == wanted."""
    return ["%s%d" % (ROWS[r], c)
            for r in range(len(ROWS)) for c in range(NCOL)
            if bool(keep[r, c]) == wanted]


# ----------------------------------------------------------------------------
def read_events(path, max_events=None):
    """Stream the file and return the per-event totals and per-SiPM sums.

    An event starts at a line with NHEAD numbers; the four count rows are the
    last four of the eight 16-number rows that follow it.  Blank lines and
    anything unparsable are skipped, so a trailing partial event at the end of
    the file is simply dropped.

    Returns (totals, per_sipm, nevents, nneg) where totals is one number per
    event, per_sipm is a (4, 16) array of the summed counts over all events
    read, and nneg counts negative entries seen in the count rows.
    """
    totals = []
    per_sipm = np.zeros((len(ROWS), NCOL), dtype=np.int64)
    nneg = 0

    rows = []          # the 16-number rows collected since the header line
    in_event = False

    def finish():
        """Close the event in progress.  True = enough events read."""
        nonlocal rows, in_event, nneg
        if in_event and len(rows) >= NTIME_ROWS + NCOUNT_ROWS:
            counts = np.array(rows[NTIME_ROWS:NTIME_ROWS + NCOUNT_ROWS],
                              dtype=np.float64)
            bad = counts < 0                       # e.g. a -999999 sentinel
            nneg += int(bad.sum())
            counts = np.where(bad, 0.0, counts)
            totals.append(float(counts[KEEP].sum()))
            per_sipm[...] += counts.astype(np.int64)
        rows = []
        in_event = False
        return max_events is not None and len(totals) >= max_events

    with open(path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue                           # not a data line

            if len(vals) == NHEAD:                 # starts a new event
                if finish():
                    break
                in_event = True
            elif len(vals) == NCOL and in_event:
                rows.append(vals)
            # any other line length is ignored

    if not (max_events is not None and len(totals) >= max_events):
        finish()                                   # the last event in the file

    return np.asarray(totals, dtype=float), per_sipm, len(totals), nneg


# ----------------------------------------------------------------------------
def make_plot(totals, out_paths, title, subtitle, lo, hi, nbins, logy=False):
    edges = np.linspace(lo, hi, nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    h, _ = np.histogram(totals, bins=edges)

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    fig.subplots_adjust(left=0.10, right=0.975, top=0.86, bottom=0.10)

    ax.step(centres, h, where="mid", color="black", lw=LINEWIDTH,
            solid_joinstyle="miter")
    ax.fill_between(centres, h, step="mid", color="black", alpha=0.08)

    ax.set_xlabel("total photons in the 50 summed SiPMs", fontsize=11)
    ax.set_ylabel("events / bin", fontsize=11)
    ax.set_xlim(lo, hi)
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 1.10 * h.max() if h.size and h.max() else 1)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=9)

    # the numbers that matter, on the plot itself
    q = np.percentile(totals, [50, 90, 99]) if totals.size else [0, 0, 0]
    stats = ("entries  %d\nmean     %.1f\nRMS      %.1f\nmedian   %.1f\n"
             "90 %%      %.1f\n99 %%      %.1f\nmax      %.0f"
             % (totals.size, totals.mean(), totals.std(ddof=1)
                if totals.size > 1 else 0.0, q[0], q[1], q[2], totals.max()
                if totals.size else 0))
    ax.text(0.985, 0.97, stats, transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, family="monospace", color="0.15",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8", lw=0.6))

    fig.suptitle(title, fontsize=14, y=0.975)
    fig.text(0.5, 0.935, subtitle, ha="center", va="top", fontsize=9,
             color="0.25", linespacing=1.5)
    for path in out_paths:
        fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("textfile", help="SiPM hit text file")
    p.add_argument("-n", "--nevents", type=int, default=NEVENTS,
                   help="read this many events (default %d; 0 = the whole "
                        "file)" % NEVENTS)
    p.add_argument("-o", "--out", default=None,
                   help="output png (default: TEXTFILE_totalPhotons.png next "
                        "to the input file)")
    p.add_argument("--bins", type=int, default=100,
                   help="number of histogram bins (default 100)")
    p.add_argument("--xmin", type=float, default=0.0,
                   help="low edge of the x axis (default 0)")
    p.add_argument("--xmax", type=float, default=None,
                   help="high edge of the x axis (default: the 99.5th "
                        "percentile of the data, rounded up)")
    p.add_argument("--logy", action="store_true",
                   help="logarithmic y axis")
    p.add_argument("--no-pdf", action="store_true",
                   help="only write the png, not the matching pdf")
    args = p.parse_args(argv)

    if not os.path.exists(args.textfile):
        sys.exit("no such file: %s" % args.textfile)

    max_events = None if args.nevents in (0, None) else args.nevents
    totals, per_sipm, nev, nneg = read_events(args.textfile, max_events)

    if nev == 0:
        sys.exit("no events parsed from %s" % args.textfile)

    summed = sipm_names(KEEP, True)
    dropped = sipm_names(KEEP, False)

    stem = os.path.splitext(os.path.abspath(args.textfile))[0]
    out_png = args.out or "%s_totalPhotons.png" % stem
    paths = [out_png] if args.no_pdf else [out_png,
                                           os.path.splitext(out_png)[0] + ".pdf"]

    if args.xmax is not None:
        hi = args.xmax
    else:
        hi = float(np.percentile(totals, 99.5)) if totals.size else 1.0
        hi = max(np.ceil(hi / 100.0) * 100.0, args.xmin + 1.0)
    lo = args.xmin

    # ---- what was read ----
    print("=" * 78)
    print("file        %s" % args.textfile)
    print("events      %d read%s" % (nev, "" if max_events is None
                                     else " (limit %d)" % max_events))
    print("summed      %d SiPMs: %s" % (len(summed), " ".join(summed)))
    print("excluded    %d SiPMs: %s" % (len(dropped), " ".join(dropped)))
    if nneg:
        print("            %d negative entries in the count rows treated as 0"
              % nneg)
    print("-" * 78)
    print("mean photons per SiPM over these events (blank = excluded):")
    print("      " + "".join("%8d" % c for c in range(NCOL)))
    for r, name in enumerate(ROWS):
        cells = []
        for c in range(NCOL):
            cells.append("%8.2f" % (per_sipm[r, c] / float(nev))
                         if KEEP[r, c] else "%8s" % "--")
        print("  %-4s" % name + "".join(cells))
    print("-" * 78)
    print("total photons per event over the %d summed SiPMs:" % len(summed))
    print("  entries %d   mean %.2f   RMS %.2f   min %.0f   max %.0f"
          % (totals.size, totals.mean(),
             totals.std(ddof=1) if totals.size > 1 else 0.0,
             totals.min(), totals.max()))
    print("  percentiles  1%% %.0f   25%% %.0f   50%% %.0f   75%% %.0f   "
          "90%% %.0f   99%% %.0f"
          % tuple(np.percentile(totals, [1, 25, 50, 75, 90, 99])))
    over = int(np.sum(totals > hi))
    print("  %d event(s) above the x range (%.1f %%) land in no bin"
          % (over, 100.0 * over / totals.size))
    print("histogram   %g to %g in %d bins (%.3g photons/bin)"
          % (lo, hi, args.bins, (hi - lo) / args.bins))
    print("=" * 78)

    sub = ("%d events  |  sum over %d SiPMs (all except columns 0-2 of A-D, "
           "C4 and C12)\n%s" % (nev, len(summed),
                                os.path.basename(args.textfile)))

    make_plot(totals, paths, "Total photons per event", sub,
              lo, hi, args.bins, logy=args.logy)
    for path in paths:
        print("wrote %s" % path)


if __name__ == "__main__":
    main()
