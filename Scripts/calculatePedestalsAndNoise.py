#!/usr/bin/env python3
"""
calculatePedestalsAndNoise.py

Read a CAEN Janus / DT5202 "list" text file, compute the per-channel pedestal
(mean) and noise (RMS) of the LG and HG ADC distributions, draw them as four
2-D detector maps, and write a 64-row summary text file.

Usage
-----
    python calculatePedestalsAndNoise.py RUNFILE.txt [options]

Outputs
-------
    RUNFILE_pedestalsAndNoise.txt   64 rows: ch, name, HG mean, HG RMS, LG mean, LG RMS
    RUNFILE_pedestalsAndNoise.png   2x2 figure of the four maps

Channel map (Sept 9 assignment)
-------------------------------
    ch 0        -> CW top      (CosmicWatch trigger, drawn above the grid)
    ch 1        -> CW bottom   (CosmicWatch trigger, drawn below the grid)
    ch 2..5     -> A1, C1, A2, C2  -- unconnected: greyed out in the maps
                                      (values are still written to the table)
    ch 6..31    -> even = A(n/2),      odd = C((n-1)/2)      n = 3..15
    ch 32..63   -> even = B((n-32)/2), odd = D((n-33)/2)     n = 0..15
"""

import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

NCH = 64
COLS = ["A", "B", "C", "D"]          # left to right
NROW = 16                            # 0 at the bottom, 15 at the top
UNCONNECTED = {2, 3, 4, 5}           # real channels, but nothing plugged in


# ----------------------------------------------------------------------------
# channel map
# ----------------------------------------------------------------------------
def build_channel_map():
    """ch -> (name, column letter or None, row or None).

    Trigger channels get a name but no grid cell; they are drawn separately.
    """
    cmap = {}
    cmap[0] = ("CW top", None, None)
    cmap[1] = ("CW bottom", None, None)
    for ch in range(2, 32):
        if ch % 2 == 0:
            col, row = "A", ch // 2
        else:
            col, row = "C", (ch - 1) // 2
        cmap[ch] = ("%s%d" % (col, row), col, row)
    for ch in range(32, 64):
        if ch % 2 == 0:
            col, row = "B", (ch - 32) // 2
        else:
            col, row = "D", (ch - 33) // 2
        cmap[ch] = ("%s%d" % (col, row), col, row)
    return cmap


CHMAP = build_channel_map()


def run_label(path):
    """'Run124_list.txt' -> 'Run 124'; falls back to the bare file name."""
    base = os.path.basename(path)
    m = re.search(r"run[_\-\s]*0*(\d+)", base, re.IGNORECASE)
    if m:
        return "Run %s" % m.group(1)
    return os.path.splitext(base)[0]


# ----------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------
def parse_list_file(path, max_events=None):
    """Parse a Janus list file.

    Returns (header_dict, lg, hg, n_events) where lg/hg are lists of
    numpy arrays, one per channel, holding every sample seen for that channel,
    tagged with the event index so events can be skipped downstream.

    The column layout is taken from the "Brd Ch ..." header line, so the same
    code works for Spect and Spect_Timing files.
    """
    header = {}
    chan_cols = None       # column names belonging to a channel row
    evt_cols = None        # column names appended on the first row of an event

    lg_v, hg_v, ch_v, ev_v = [], [], [], []
    ievt = -1
    i_lg = i_hg = None

    with open(path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s.startswith("//"):
                body = s.lstrip("/").strip()
                if ":" in body:
                    k, v = body.split(":", 1)
                    # keep "Run start time: Fri Sep 11 14:23:36 2026 UTC" intact
                    if k.strip() and not k.strip().startswith("*"):
                        header[k.strip()] = v.strip()
                continue

            parts = s.split()

            # column header line
            if parts[0] == "Brd":
                names = parts
                # first column that looks like the event-level timestamp
                split_at = len(names)
                for i, n in enumerate(names):
                    if n.lower().startswith("tstamp"):
                        split_at = i
                        break
                chan_cols = names[:split_at]
                evt_cols = names[split_at:]
                i_lg = chan_cols.index("LG") if "LG" in chan_cols else 2
                i_hg = chan_cols.index("HG") if "HG" in chan_cols else 3
                continue

            if not parts[0].lstrip("-").isdigit():
                continue

            if chan_cols is None:                     # no header line seen
                chan_cols = ["Brd", "Ch", "LG", "HG", "ToA_ns", "ToT_ns"]
                evt_cols = ["TStamp_us", "Tref_TStamp_us", "TrgID", "NChs"]
                i_lg, i_hg = 2, 3

            nchan = len(chan_cols)
            if len(parts) < i_hg + 1:
                continue                              # truncated final line

            # a row carrying the event-level columns starts a new event
            if len(parts) > nchan:
                ievt += 1
                if max_events is not None and ievt >= max_events:
                    break

            if ievt < 0:
                ievt = 0

            try:
                ch = int(parts[1])
                lg = float(parts[i_lg])
                hg = float(parts[i_hg])
            except ValueError:
                continue

            if not 0 <= ch < NCH:
                continue

            ch_v.append(ch)
            lg_v.append(lg)
            hg_v.append(hg)
            ev_v.append(ievt)

    return (header,
            np.asarray(ch_v, dtype=np.int16),
            np.asarray(lg_v, dtype=np.float32),
            np.asarray(hg_v, dtype=np.float32),
            np.asarray(ev_v, dtype=np.int64),
            ievt + 1)


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def channel_stats(ch, lg, hg, ev, skip_events=0, robust=False):
    """Per-channel mean/RMS of LG and HG. Returns dict of 64-length arrays."""
    keep = ev >= skip_events
    ch, lg, hg = ch[keep], lg[keep], hg[keep]

    out = {k: np.full(NCH, np.nan) for k in
           ("hg_mean", "hg_rms", "lg_mean", "lg_rms")}
    out["n"] = np.zeros(NCH, dtype=int)

    for c in range(NCH):
        m = ch == c
        n = int(m.sum())
        out["n"][c] = n
        if n == 0:
            continue
        for tag, arr in (("hg", hg[m]), ("lg", lg[m])):
            if robust:
                med = np.median(arr)
                # MAD scaled to a Gaussian sigma
                rms = 1.4826 * np.median(np.abs(arr - med))
                out[tag + "_mean"][c] = med
                out[tag + "_rms"][c] = rms
            else:
                out[tag + "_mean"][c] = arr.mean()
                out[tag + "_rms"][c] = arr.std(ddof=1) if n > 1 else 0.0
    return out


# ----------------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------------
def to_grid(values):
    """64 channel values -> (16, 4) array indexed [row, col] with col A..D.

    Unconnected channels are left as NaN so they are greyed out.
    """
    g = np.full((NROW, len(COLS)), np.nan)
    for ch, (name, col, row) in CHMAP.items():
        if col is None or ch in UNCONNECTED:
            continue
        g[row, COLS.index(col)] = values[ch]
    return g


def grid_cells_of(channels):
    """set of (row, col index) cells occupied by the given channels."""
    cells = set()
    for ch in channels:
        _, col, row = CHMAP[ch]
        if col is not None:
            cells.add((row, COLS.index(col)))
    return cells


UNCONNECTED_CELLS = grid_cells_of(UNCONNECTED)


def fmt(v, vmax):
    if not np.isfinite(v):
        return ""
    if vmax < 10:
        return "%.2f" % v
    if vmax < 100:
        return "%.1f" % v
    return "%.0f" % v


def text_colour(rgba):
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "black" if lum > 0.55 else "white"


def draw_panel(fig, gs_cell, values, title, cmap_name, label):
    """One panel: CW-top strip, 16x4 grid, CW-bottom strip, shared colourbar."""
    # row 1 is an empty spacer that keeps the CW-top box off the A/B/C/D labels
    inner = gs_cell.subgridspec(4, 2, height_ratios=[1.0, 0.7, 16.0, 1.0],
                                width_ratios=[24, 1], hspace=0.08, wspace=0.04)
    ax_top = fig.add_subplot(inner[0, 0])
    ax_mid = fig.add_subplot(inner[2, 0])
    ax_bot = fig.add_subplot(inner[3, 0])
    ax_cb = fig.add_subplot(inner[:, 1])

    grid = to_grid(values)
    # colour scale from the connected channels only; z axis always starts at 0
    conn = np.array([values[c] for c in range(NCH)
                     if c not in UNCONNECTED and np.isfinite(values[c])])
    finite = conn
    vmin = 0.0
    if finite.size:
        vmax = float(np.percentile(finite, 98))
        if vmax <= vmin:
            vmax = float(max(finite.max(), vmin + 1.0))
    else:
        vmax = 1.0

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#e8e8e8")
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    im = ax_mid.imshow(np.ma.masked_invalid(grid), origin="lower",
                       extent=[0, len(COLS), 0, NROW], aspect="auto",
                       cmap=cmap, norm=norm, interpolation="nearest")

    scale = np.nanmax(np.abs(finite)) if finite.size else 1.0
    for r in range(NROW):
        for c in range(len(COLS)):
            v = grid[r, c]
            x, y = c + 0.5, r + 0.5
            if not np.isfinite(v):
                lab = "n/c" if (r, c) in UNCONNECTED_CELLS else "--"
                ax_mid.text(x, y, lab, ha="center", va="center",
                            fontsize=6, color="#8a8a8a")
                continue
            ax_mid.text(x, y, fmt(v, scale), ha="center", va="center",
                        fontsize=6.5, color=text_colour(cmap(norm(v))))

    ax_mid.set_xticks(np.arange(len(COLS)) + 0.5)
    ax_mid.set_xticklabels(COLS, fontsize=9)
    ax_mid.xaxis.tick_top()          # keep the A/B/C/D labels clear of the
    ax_mid.xaxis.set_label_position("top")   # CW-bottom strip
    ax_mid.set_yticks(np.arange(NROW) + 0.5)
    ax_mid.set_yticklabels([str(i) for i in range(NROW)], fontsize=6)
    ax_mid.set_xticks(np.arange(len(COLS) + 1), minor=True)
    ax_mid.set_yticks(np.arange(NROW + 1), minor=True)
    ax_mid.grid(which="minor", color="white", lw=0.6)
    ax_mid.tick_params(which="minor", length=0)
    ax_mid.tick_params(which="major", length=2)

    # CosmicWatch trigger channels, above and below the grid
    for ax, ch in ((ax_top, 0), (ax_bot, 1)):
        name = CHMAP[ch][0]
        v = values[ch]
        ax.set_xlim(0, len(COLS))
        ax.set_ylim(0, 1)
        if np.isfinite(v):
            fc = cmap(norm(v))
            tc = text_colour(fc)
            txt = "%s   %s" % (name, fmt(v, scale))
        else:
            fc, tc, txt = "#e8e8e8", "#555555", "%s   --" % name
        ax.add_patch(Rectangle((1.0, 0.06), 2.0, 0.88, facecolor=fc,
                               edgecolor="0.35", lw=0.8))
        ax.text(2.0, 0.5, txt, ha="center", va="center", fontsize=7, color=tc)
        ax.text(0.06, 0.5, "ch%d" % ch, ha="left", va="center",
                fontsize=6, color="0.4")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    ax_top.set_title(title, fontsize=10, pad=6)
    lo = bool(finite.size and finite.min() < vmin)
    hi = bool(finite.size and finite.max() > vmax)
    extend = ("both" if lo and hi else "min" if lo else "max" if hi else "neither")
    cb = fig.colorbar(im, cax=ax_cb, extend=extend)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return im


def make_figure(stats, out_png, suptitle, robust=False):
    fig = plt.figure(figsize=(11.5, 13.0))
    gs = fig.add_gridspec(2, 2, hspace=0.13, wspace=0.16,
                          left=0.055, right=0.955, top=0.925, bottom=0.035)

    centre = "median" if robust else "mean"
    width = "MAD-sigma" if robust else "RMS"

    panels = [
        (gs[0, 0], stats["hg_mean"], "HG pedestal (%s)" % centre, "viridis"),
        (gs[0, 1], stats["hg_rms"], "HG noise (%s)" % width, "magma"),
        (gs[1, 0], stats["lg_mean"], "LG pedestal (%s)" % centre, "viridis"),
        (gs[1, 1], stats["lg_rms"], "LG noise (%s)" % width, "magma"),
    ]
    for cell, vals, title, cname in panels:
        draw_panel(fig, cell, vals, title, cname, "ADC")

    fig.suptitle(suptitle, fontsize=12, y=0.975)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------------
# text output
# ----------------------------------------------------------------------------
def write_table(stats, out_txt, src, header, n_events, skip, robust):
    centre = "median" if robust else "mean"
    width = "1.4826*MAD" if robust else "RMS (sample std)"
    with open(out_txt, "w") as f:
        f.write("# Pedestals and noise from %s\n" % os.path.basename(src))
        for k in ("Board", "Acquisition Mode", "Run start time",
                  "Janus Release", "File Format Version"):
            if k in header:
                f.write("# %s: %s\n" % (k, header[k]))
        f.write("# events parsed: %d   events skipped at start: %d\n"
                % (n_events, skip))
        f.write("# pedestal = %s of the ADC distribution, "
                "noise = %s\n" % (centre, width))
        f.write("# channel map: Sept 9 assignment "
                "(ch2-5 unconnected, ch0/1 CosmicWatch)\n")
        f.write("#%4s %10s %12s %12s %12s %12s %9s\n"
                % ("ch", "name", "HG_ped", "HG_noise",
                   "LG_ped", "LG_noise", "N"))
        for c in range(NCH):
            name = CHMAP[c][0]
            if c in UNCONNECTED:
                name += "*"
            f.write("%5d %10s %12.4f %12.4f %12.4f %12.4f %9d\n"
                    % (c, name,
                       stats["hg_mean"][c], stats["hg_rms"][c],
                       stats["lg_mean"][c], stats["lg_rms"][c],
                       stats["n"][c]))
        f.write("# * = unconnected channel\n")


# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("listfile", help="Janus list text file")
    p.add_argument("--skip-events", type=int, default=1,
                   help="drop this many events from the start of the run "
                        "(default 1: event 0 is a full-scale start-of-run "
                        "artifact); use 0 to keep everything")
    p.add_argument("--max-events", type=int, default=None,
                   help="stop after this many events")
    p.add_argument("--robust", action="store_true",
                   help="use median and 1.4826*MAD instead of mean and RMS "
                        "(better with HV on, where HG has a dark-count tail)")
    p.add_argument("--runlabel", default=None,
                   help="title text for the figure "
                        "(default: 'Run N' taken from the file name)")
    p.add_argument("-o", "--outprefix", default=None,
                   help="output prefix (default: input file without .txt)")
    p.add_argument("--no-plot", action="store_true", help="skip the figure")
    args = p.parse_args(argv)

    src = args.listfile
    if not os.path.exists(src):
        sys.exit("no such file: %s" % src)

    prefix = args.outprefix
    if prefix is None:
        prefix = src[:-4] if src.lower().endswith(".txt") else src
    out_txt = prefix + "_pedestalsAndNoise.txt"
    out_png = prefix + "_pedestalsAndNoise.png"

    header, ch, lg, hg, ev, n_events = parse_list_file(src, args.max_events)
    if ch.size == 0:
        sys.exit("no data rows found in %s" % src)

    stats = channel_stats(ch, lg, hg, ev,
                          skip_events=args.skip_events, robust=args.robust)

    write_table(stats, out_txt, src, header, n_events, args.skip_events,
                args.robust)

    if not args.no_plot:
        bits = [args.runlabel or run_label(src)]
        if "Run start time" in header:
            bits.append(header["Run start time"])
        bits.append("%d events" % (n_events - args.skip_events))
        if args.robust:
            bits.append("median / MAD")
        make_figure(stats, out_png, "Pedestals and noise   -   " +
                    "   |   ".join(bits), robust=args.robust)

    # console summary
    good = np.array([c for c in range(NCH) if stats["n"][c] > 0])
    print("parsed %d events, %d channels with data" % (n_events, good.size))
    for tag, lab in (("hg", "HG"), ("lg", "LG")):
        m = stats[tag + "_mean"][good]
        r = stats[tag + "_rms"][good]
        print("  %s  pedestal %7.2f  (%6.2f - %6.2f)   noise %6.2f  "
              "(%5.2f - %5.2f)"
              % (lab, np.nanmean(m), np.nanmin(m), np.nanmax(m),
                 np.nanmean(r), np.nanmin(r), np.nanmax(r)))
    print("wrote %s" % out_txt)
    if not args.no_plot:
        print("wrote %s" % out_png)


if __name__ == "__main__":
    main()
