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
    the raw HG ADC value.

    RunX_photonPeaks.png                        all four columns overlaid
    RunX_photonPeaks_A.png .. _D.png            one detector column each
    RunX_photonPeaks_all50.png                  the 50 active SiPMs together

    A matching .pdf is written beside each png; --no-pdf skips them.
    All next to the input file; -o renames the combined plot and the others
    follow it.  --no-per-column writes only the combined plot.

The 50 active SiPMs
-------------------
Everything except rows 0, 1, 2 of all four columns and C4 and C12 -- the same
set summed by makeTotalPhotonHisto.py and kept by pickPoCRows.py, i.e. the
SiPMs that are connected and alive to bias:

    A3-A15, B3-B15, C3 C5-C11 C13-C15, D3-D15

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

# the 50 active SiPMs: everything except these
EXCLUDE_ROWS = [0, 1, 2]            # these rows in every column
EXCLUDE_CELLS = [("C", 4), ("C", 12)]

# thinnest line that still renders solidly: 0.5 pt is one pixel at 150 dpi and
# a true hairline in the pdf
LINEWIDTH = 0.5
YHEADROOM = 1.10                    # y axis top = this x the highest bin
DPI = 150

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


def is_active(ch):
    """True for the 50 connected, alive SiPMs."""
    name, col, row = CHMAP[ch]
    if col is None or ch in UNCONNECTED:
        return False
    if row in EXCLUDE_ROWS:
        return False
    return (col, row) not in EXCLUDE_CELLS


ACTIVE_CH = [ch for ch in range(NCH) if is_active(ch)]


def run_tag(path):
    """'Run125_list.txt' -> 'Run125'; falls back to the bare file name."""
    base = os.path.basename(path)
    m = re.search(r"run[_\-\s]*0*(\d+)", base, re.IGNORECASE)
    if m:
        return "Run%s" % m.group(1)
    return os.path.splitext(base)[0]


def outputs(png, no_pdf):
    """The png plus its matching pdf, unless pdfs are switched off."""
    return [png] if no_pdf else [png, os.path.splitext(png)[0] + ".pdf"]


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
# the 4 x 4 grid
# ----------------------------------------------------------------------------
def make_plot(vals, out_paths, title, subtitle, lo, hi, nbins,
              logy=False, density=False, column="HG", cols=None,
              per_panel_y=False):
    """One 4x4 grid.  cols selects which detector columns to overlay."""
    cols = list(COLS) if cols is None else list(cols)
    edges = np.linspace(lo, hi, nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    share_y = (not density) and (not per_panel_y)
    fig, axes = plt.subplots(4, 4, figsize=(14.5, 11.0),
                             sharex=True, sharey=share_y)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.885, bottom=0.055,
                        hspace=0.38, wspace=0.26)

    ymax = 0.0                      # highest bin anywhere in the figure
    panel_max = {}                  # highest bin in each panel
    for k, row in enumerate(PANEL_ROWS):
        ax = axes[k // 4][k % 4]
        nplotted = 0
        pmax = 0.0
        for col in cols:
            ch = CELL2CH.get((row, col))
            if ch is None:
                continue
            x = vals[ch]
            if x.size == 0:
                continue
            h, _ = np.histogram(x, bins=edges, density=density)
            ax.step(centres, h, where="mid", color=COLCOLOUR[col],
                    lw=LINEWIDTH, solid_joinstyle="miter",
                    label="%s%d  (ch%02d)" % (col, row, ch))
            pmax = max(pmax, float(h.max()) if h.size else 0.0)
            nplotted += 1
        ymax = max(ymax, pmax)
        panel_max[k] = pmax

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
        ax.set_xlabel("%s ADC counts" % column, fontsize=9, labelpad=1.5)
        ax.set_ylabel("events / bin" if not density else "normalised",
                      fontsize=9)
        # tick labels on every panel, not just the outer row and column
        ax.tick_params(labelsize=8, labelleft=True, labelbottom=True)

    # y axis top: YHEADROOM x the highest bin -- of the whole figure when the
    # panels share an axis, of each panel on its own with --per-panel-y
    if not density and not logy:
        if per_panel_y:
            for k in range(len(PANEL_ROWS)):
                m = panel_max.get(k, 0.0)
                axes[k // 4][k % 4].set_ylim(0, YHEADROOM * m if m > 0 else 1)
        elif ymax > 0:
            axes[0][0].set_ylim(0, YHEADROOM * ymax)

    fig.suptitle(title, fontsize=14, y=0.982)
    fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=9,
             color="0.25", linespacing=1.5)
    for path in out_paths:
        fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return ymax


# ----------------------------------------------------------------------------
# the 50 active SiPMs in one panel
# ----------------------------------------------------------------------------
def make_all50(vals, out_paths, title, subtitle, lo, hi, nbins,
               logy=False, column="HG", show_individual=False):
    """One axis: the mean spectrum of the active SiPMs.

    All 50 channels have the same number of entries -- one per event -- so the
    pooled spectrum divided by 50 is the average channel.  With
    show_individual the 50 channels are drawn faintly behind it, on the same
    scale; the y axis then covers them too.
    """
    edges = np.linspace(lo, hi, nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    fig.subplots_adjust(left=0.085, right=0.98, top=0.855, bottom=0.095)

    pooled = np.zeros(nbins, dtype=float)
    nch = 0
    ntot = 0
    ymax = 0.0
    seen_cols = []
    for ch in ACTIVE_CH:
        x = vals[ch]
        if x.size == 0:
            continue
        h, _ = np.histogram(x, bins=edges)
        pooled += h
        nch += 1
        ntot += x.size
        col = CHMAP[ch][1]
        if show_individual:
            ax.step(centres, h, where="mid", color=COLCOLOUR[col],
                    lw=LINEWIDTH, solid_joinstyle="miter", alpha=0.30,
                    zorder=2)
            ymax = max(ymax, float(h.max()) if h.size else 0.0)
        if col not in seen_cols:
            seen_cols.append(col)

    if nch == 0:
        sys.exit("no active channels had any entries")

    mean = pooled / float(nch)
    ymax = max(ymax, float(mean.max()))
    ax.fill_between(centres, mean, step="mid", color="0.2", alpha=0.10,
                    zorder=3)
    ax.step(centres, mean, where="mid", color="0.1", lw=1.6,
            solid_joinstyle="miter", zorder=4,
            label="mean of the %d SiPMs  (%d entries / %d)" % (nch, ntot, nch))

    if show_individual:
        handles = [plt.Line2D([], [], color=COLCOLOUR[c], lw=1.2, alpha=0.6,
                              label="column %s" % c) for c in COLS
                   if c in seen_cols]
        handles.append(plt.Line2D([], [], color="0.1", lw=1.6,
                                  label="mean of all %d" % nch))
        ax.legend(handles=handles, fontsize=9, frameon=False,
                  loc="upper right")
    else:
        ax.legend(fontsize=9, frameon=False, loc="upper right")

    ax.set_xlabel("%s ADC counts" % column, fontsize=11)
    ax.set_ylabel("events / bin, per SiPM", fontsize=11)
    ax.set_xlim(lo, hi)
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, YHEADROOM * ymax if ymax > 0 else 1)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=9)

    fig.suptitle(title, fontsize=14, y=0.975)
    fig.text(0.5, 0.935, subtitle, ha="center", va="top", fontsize=9,
             color="0.25", linespacing=1.5)
    for path in out_paths:
        fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return nch, ntot, ymax


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
    p.add_argument("--per-panel-y", action="store_true",
                   help="scale every panel of the grid to its own highest "
                        "bin instead of sharing one y axis set by the highest "
                        "bin in the figure")
    p.add_argument("--density", action="store_true",
                   help="normalise each channel to unit area instead of "
                        "plotting raw counts")
    p.add_argument("--no-per-column", action="store_true",
                   help="only write the combined A-D plot, not the one-plot-"
                        "per-column versions")
    p.add_argument("--no-all50", action="store_true",
                   help="skip the plot with all 50 active SiPMs together")
    p.add_argument("--all50-individual", action="store_true",
                   help="in the all-50 plot also draw the 50 individual "
                        "channels faintly behind the mean (default: the mean "
                        "alone, and the y axis then covers only the mean)")
    p.add_argument("--no-pdf", action="store_true",
                   help="only write the pngs, not the matching pdfs")
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

    active_names = [CHMAP[ch][0] for ch in ACTIVE_CH]

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
    print("active      %d SiPMs in the all-50 plot: %s"
          % (len(ACTIVE_CH), " ".join(active_names)))
    print("-" * 78)
    print("  %-5s %-6s %-8s %-7s %8s %8s %8s %8s"
          % ("ch", "name", "colour", "active", "entries", "median",
             "in range", "frac"))
    inrange_tot = 0
    ntot = 0
    for row in PANEL_ROWS:
        for col in COLS:
            ch = CELL2CH.get((row, col))
            if ch is None:
                continue
            act = "yes" if is_active(ch) else "no"
            x = vals[ch]
            if x.size == 0:
                print("  %-5d %-6s %-8s %-7s %8d %8s %8s %8s"
                      % (ch, CHMAP[ch][0], COLCOLOUR[col], act, 0, "-", "-",
                         "-"))
                continue
            inr = int(np.sum((x >= args.xmin) & (x <= args.xmax)))
            inrange_tot += inr
            ntot += x.size
            print("  %-5d %-6s %-8s %-7s %8d %8.1f %8d %7.1f%%"
                  % (ch, CHMAP[ch][0], COLCOLOUR[col], act, x.size,
                     float(np.median(x)), inr, 100.0 * inr / x.size))
    for ch in sorted(TRIG_CH | UNCONNECTED):
        x = vals[ch]
        print("  %-5d %-6s %-8s %-7s %8d %8s   (not plotted)"
              % (ch, CHMAP[ch][0].replace(" ", ""), "-", "no", x.size,
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

    stem, ext = os.path.splitext(out_png)
    if not args.no_per_column:
        for col in COLS:
            jobs.append(([col], "%s_%s%s" % (stem, col, ext),
                         "%s   photon peaks   column %s" % (tag, col),
                         base + "  |  column %s only (%s)"
                         % (col, COLCOLOUR[col]) + stamp))

    written = []
    for cols, png, title, sub in jobs:
        paths = outputs(png, args.no_pdf)
        top = make_plot(vals, paths, title, sub, args.xmin, args.xmax,
                        args.nbins, logy=args.logy, density=args.density,
                        column=args.column, cols=cols,
                        per_panel_y=args.per_panel_y)
        if not args.density and not args.logy:
            print("y axis     %s: highest bin %.0f -> top %.0f"
                  % (os.path.basename(png), top, YHEADROOM * top)
                  if not args.per_panel_y else
                  "y axis     %s: each panel scaled to its own highest bin"
                  % os.path.basename(png))
        written += paths

    # ---- all 50 active SiPMs in one panel ----
    if not args.no_all50:
        png = "%s_all50%s" % (stem, ext)
        paths = outputs(png, args.no_pdf)
        nch, nent, top50 = make_all50(
            vals, paths, "%s   photon peaks   all %d active SiPMs"
            % (tag, len(ACTIVE_CH)),
            base + ("  |  every active SiPM faint, their mean in black"
                    if args.all50_individual
                    else "  |  mean of the active SiPMs") + stamp,
            args.xmin, args.xmax, args.nbins, logy=args.logy,
            column=args.column, show_individual=args.all50_individual)
        print("all50      %d SiPMs, %d entries pooled (%d per SiPM)"
              % (nch, nent, nent // nch if nch else 0))
        if not args.logy:
            print("           highest bin %.1f -> y axis top %.1f"
                  % (top50, YHEADROOM * top50))
        written += paths

    print("-" * 78)
    for path in written:
        print("wrote %s" % path)


if __name__ == "__main__":
    main()
