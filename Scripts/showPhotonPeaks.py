#!/usr/bin/env python3
"""
showPhotonPeaks.py

Single-photoelectron spectra for the 64 channels of a CAEN Janus / DT5202 run.

Usage
-----
    python showPhotonPeaks.py RUNFILE.txt [options]

Output
------
    A 4x4 grid of panels.  Each panel is one detector row; the rows run
    15, 14, 13, 12  /  11, 10, 9, 8  /  7, 6, 5, 4  /  3, 2, 1, 0, so row 15
    is upper left and row 0 is lower right.  Within a panel the four columns
    A, B, C, D are overlaid as black, red, blue and green step histograms of
    the raw HG ADC value, 0-100 in 20 bins.

    RunX_photonPeaks.png                        all four columns overlaid
    RunX_photonPeaks_A.png .. _D.png            one detector column each

    All next to the input file; -o renames the combined plot and the per-
    column ones follow it.  --no-per-column writes only the combined plot.

Channel map (Sept 9 assignment)
-------------------------------
    ch 0        -> CW top      (CosmicWatch trigger, not plotted)
    ch 1        -> CW bottom   (CosmicWatch trigger, not plotted)
    ch 2..5     -> A1, C1, A2, C2  -- unconnected, not plotted
    ch 6..31    -> even = A(n/2),      odd = C((n-1)/2)      row = 3..15
    ch 32..63   -> even = B((n-32)/2), odd = D((n-33)/2)     row = 0..15
"""

import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
NCH = 64
COLS = ["A", "B", "C", "D"]
NROW = 16
UNCONNECTED = {2, 3, 4, 5}          # real channels, nothing plugged in
TRIG_CH = {0, 1}                    # CosmicWatch trigger channels

COLCOLOUR = {"A": "black", "B": "red", "C": "blue", "D": "green"}

HG_LO = 20.0       # x axis of every panel
HG_HI = 100.0
NBINS = 80

# panel order: 15 at the upper left, then decreasing left to right, top to
# bottom, so row 0 ends up at the lower right
PANEL_ROWS = list(range(NROW - 1, -1, -1))


# ----------------------------------------------------------------------------
def build_channel_map():
    """ch -> (name, column letter or None, row or None)."""
    cmap = {0: ("CW top", None, None), 1: ("CW bottom", None, None)}
    for ch in range(2, 32):
        col, row = ("A", ch // 2) if ch % 2 == 0 else ("C", (ch - 1) // 2)
        cmap[ch] = ("%s%d" % (col, row), col, row)
    for ch in range(32, 64):
        col, row = ("B", (ch - 32) // 2) if ch % 2 == 0 \
            else ("D", (ch - 33) // 2)
        cmap[ch] = ("%s%d" % (col, row), col, row)
    return cmap


CHMAP = build_channel_map()

# (row, column letter) -> channel, for the channels that are actually read out
CELL2CH = {}
for _ch, (_nm, _col, _row) in CHMAP.items():
    if _col is not None and _ch not in UNCONNECTED:
        CELL2CH[(_row, _col)] = _ch


def run_tag(path):
    """'Run125_list.txt' -> 'Run125'; falls back to the bare file name."""
    base = os.path.basename(path)
    m = re.search(r"run[_\-\s]*0*(\d+)", base, re.IGNORECASE)
    if m:
        return "Run%s" % m.group(1)
    return os.path.splitext(base)[0]


# ----------------------------------------------------------------------------
# reading the run
# ----------------------------------------------------------------------------
def read_hg(path, skip_events=1, max_events=None, column="HG"):
    """Stream a Janus list file and collect the per-channel ADC values.

    Only the first line of an event carries the TStamp/TrgID columns; the other
    63 lines are just Brd Ch LG HG ToA ToT.  A line longer than the column
    header therefore starts a new event.

    Returns (header dict, list of 64 numpy arrays, n_events_used, n_seen).
    """
    header = {}
    chan_cols = None
    i_val = None
    vals = [[] for _ in range(NCH)]
    nseen = 0           # events encountered in the file
    nused = 0           # events actually filled
    keep = False

    with open(path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s.startswith("//"):
                body = s.lstrip("/").strip()
                if ":" in body:
                    k, v = body.split(":", 1)
                    if k.strip() and not k.strip().startswith("*"):
                        header[k.strip()] = v.strip()
                continue

            parts = s.split()

            if parts[0] == "Brd":                       # column header line
                split_at = len(parts)
                for i, n in enumerate(parts):
                    if n.lower().startswith("tstamp"):
                        split_at = i
                        break
                chan_cols = parts[:split_at]
                if column not in chan_cols:
                    sys.exit("no '%s' column in %s (found %s)"
                             % (column, path, " ".join(chan_cols)))
                i_val = chan_cols.index(column)
                continue

            if not parts[0].lstrip("-").isdigit():
                continue

            if chan_cols is None:                       # no header line seen
                chan_cols = ["Brd", "Ch", "LG", "HG", "ToA_ns", "ToT_ns"]
                i_val = chan_cols.index(column)

            nchan = len(chan_cols)
            if len(parts) < i_val + 1:
                continue                                # truncated final line

            if len(parts) > nchan:                      # starts a new event
                if max_events is not None and nused >= max_events:
                    break
                keep = nseen >= skip_events
                nseen += 1
                if keep:
                    nused += 1

            if not keep:
                continue

            try:
                ch = int(parts[1])
                v = float(parts[i_val])
            except ValueError:
                continue                                # e.g. a '-' entry
            if 0 <= ch < NCH:
                vals[ch].append(v)

    return header, [np.asarray(v, dtype=float) for v in vals], nused, nseen


# ----------------------------------------------------------------------------
# the plot
# ----------------------------------------------------------------------------
def make_plot(vals, out_png, title, subtitle, lo, hi, nbins,
              logy=False, density=False, column="HG", cols=None):
    """One 4x4 grid.  cols selects which detector columns to overlay."""
    cols = list(COLS) if cols is None else list(cols)
    edges = np.linspace(lo, hi, nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(4, 4, figsize=(14.5, 11.0),
                             sharex=True, sharey=not density)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.885, bottom=0.065,
                        hspace=0.20, wspace=0.26)

    ymax = 0.0
    for k, row in enumerate(PANEL_ROWS):
        ax = axes[k // 4][k % 4]
        nplotted = 0
        for col in cols:
            ch = CELL2CH.get((row, col))
            if ch is None:
                continue
            x = vals[ch]
            if x.size == 0:
                continue
            h, _ = np.histogram(x, bins=edges, density=density)
            ax.step(centres, h, where="mid", color=COLCOLOUR[col], lw=1.0,
                    label="%s%d  (ch%02d)" % (col, row, ch))
            ymax = max(ymax, float(h.max()) if h.size else 0.0)
            nplotted += 1

        ax.set_title("Row %d" % row, fontsize=10, pad=3)
        if nplotted:
            ax.legend(fontsize=6.5, loc="upper right", frameon=False,
                      handlelength=1.4, borderpad=0.2, labelspacing=0.25)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlim(lo, hi)
        if logy:
            ax.set_yscale("log")
        if nplotted == 0:
            ax.text(0.5, 0.5, "no channels", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="0.5")
        if k // 4 == 3:
            ax.set_xlabel("%s ADC counts" % column, fontsize=9)
        ax.set_ylabel("events / bin" if not density else "normalised",
                      fontsize=9)
        # y tick labels on every panel, not just the left column
        ax.tick_params(labelsize=8, labelleft=True)

    if not density and ymax > 0 and not logy:
        axes[0][0].set_ylim(0, 1.10 * ymax)

    fig.suptitle(title, fontsize=14, y=0.982)
    fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=9,
             color="0.25", linespacing=1.5)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("listfile", help="Janus list text file")
    p.add_argument("-o", "--out", default=None,
                   help="output png (default: RunX_photonPeaks.png next to "
                        "the input file)")
    p.add_argument("-n", "--nevents", type=int, default=None,
                   help="stop after this many events (default: whole file)")
    p.add_argument("--skip-events", type=int, default=1,
                   help="drop this many events from the start of the run "
                        "(default 1: event 0 is a start-of-run artifact)")
    p.add_argument("--column", default="HG", choices=["HG", "LG"],
                   help="which ADC column to histogram (default HG)")
    p.add_argument("--xmin", type=float, default=HG_LO,
                   help="low edge of the x axis (default %g)" % HG_LO)
    p.add_argument("--xmax", type=float, default=HG_HI,
                   help="high edge of the x axis (default %g)" % HG_HI)
    p.add_argument("--nbins", type=int, default=NBINS,
                   help="number of bins (default %d)" % NBINS)
    p.add_argument("--logy", action="store_true",
                   help="logarithmic y axis")
    p.add_argument("--density", action="store_true",
                   help="normalise each channel to unit area instead of "
                        "plotting raw counts")
    p.add_argument("--no-per-column", action="store_true",
                   help="only write the combined A-D plot, not the one-plot-"
                        "per-column versions")
    args = p.parse_args(argv)

    if not os.path.exists(args.listfile):
        sys.exit("no such list file: %s" % args.listfile)

    header, vals, nused, nseen = read_hg(
        args.listfile, skip_events=args.skip_events,
        max_events=args.nevents, column=args.column)

    if nused == 0:
        sys.exit("no events read from %s (%d seen, %d skipped)"
                 % (args.listfile, nseen, args.skip_events))

    tag = run_tag(args.listfile)
    out_png = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.listfile)),
        "%s_photonPeaks.png" % tag)

    # ---- what was read ----
    print("=" * 78)
    print("file        %s" % args.listfile)
    for k in ("Board", "Acquisition Mode", "Janus Release",
              "Energy Histogram NBins", "Run start time"):
        if k in header:
            print("  %-22s %s" % (k, header[k]))
    print("events      %d used of %d seen (first %d skipped)"
          % (nused, nseen, args.skip_events))
    print("histogram   %s ADC, %g to %g in %d bins (%.3g counts/bin)"
          % (args.column, args.xmin, args.xmax, args.nbins,
             (args.xmax - args.xmin) / args.nbins))
    print("-" * 78)
    print("  %-5s %-6s %-8s %8s %8s %8s %8s"
          % ("ch", "name", "colour", "entries", "median", "in range", "frac"))
    inrange_tot = 0
    ntot = 0
    for row in PANEL_ROWS:
        for col in COLS:
            ch = CELL2CH.get((row, col))
            if ch is None:
                continue
            x = vals[ch]
            if x.size == 0:
                print("  %-5d %-6s %-8s %8d %8s %8s %8s"
                      % (ch, CHMAP[ch][0], COLCOLOUR[col], 0, "-", "-", "-"))
                continue
            inr = int(np.sum((x >= args.xmin) & (x <= args.xmax)))
            inrange_tot += inr
            ntot += x.size
            print("  %-5d %-6s %-8s %8d %8.1f %8d %7.1f%%"
                  % (ch, CHMAP[ch][0], COLCOLOUR[col], x.size,
                     float(np.median(x)), inr, 100.0 * inr / x.size))
    for ch in sorted(TRIG_CH | UNCONNECTED):
        x = vals[ch]
        print("  %-5d %-6s %-8s %8d %8s   (not plotted)"
              % (ch, CHMAP[ch][0].replace(" ", ""), "-", x.size,
                 "%.1f" % np.median(x) if x.size else "-"))
    print("-" * 78)
    if ntot:
        print("plotted channels: %d of %d entries inside the x range (%.1f %%)"
              % (inrange_tot, ntot, 100.0 * inrange_tot / ntot))
    print("=" * 78)

    base = ("%d events  |  first %d skipped  |  %s ADC, %g-%g in %d bins"
            % (nused, args.skip_events, args.column, args.xmin, args.xmax,
               args.nbins))
    stamp = ("\n%s" % header["Run start time"]) if "Run start time" in header \
        else ""

    # ---- the combined A-D plot, then one plot per detector column ----
    jobs = [(list(COLS), out_png,
             "%s   photon peaks" % tag,
             base + "  |  A black, B red, C blue, D green" + stamp)]

    if not args.no_per_column:
        stem, ext = os.path.splitext(out_png)
        for col in COLS:
            jobs.append(([col], "%s_%s%s" % (stem, col, ext),
                         "%s   photon peaks   column %s" % (tag, col),
                         base + "  |  column %s only (%s)"
                         % (col, COLCOLOUR[col]) + stamp))

    for cols, png, title, sub in jobs:
        make_plot(vals, png, title, sub, args.xmin, args.xmax, args.nbins,
                  logy=args.logy, density=args.density, column=args.column,
                  cols=cols)
        print("wrote %s" % png)


if __name__ == "__main__":
    main()
