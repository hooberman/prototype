#!/usr/bin/env python3
"""
makeEventDisplays.py

Draw the 2-D detector image for individual events of a CAEN Janus / DT5202 run.

Per channel and event:

    HGcorr = HG - HG_pedestal
    LGcorr = LG - LG_pedestal
    signal = HGcorr                       normally
           = LGcorr * LGscale             when HG is saturated

The pedestal file and the LG scale file are hardcoded below.  Everything read
from them is echoed to the screen before any plotting.

Usage
-----
    python makeEventDisplays.py RUNFILE.txt [-n 100] [--normalize]

--normalize switches the image from ADC to the share of the event: every cell
is divided by the sum over the 58 cells of the image, the colour scale runs
0-100 %, the cell labels carry two significant figures, and a vertical bar at
the lower left shows the total that was divided out, on a fixed axis so the
bars are comparable between events.

Output
------
    RunX_plots/RunX_evtNNNNN.png    one image per event, X parsed from the
                                    input file name.  The folder is created
                                    next to the input file.

Channel map (Sept 9 assignment)
-------------------------------
    ch 0        -> CW top      (CosmicWatch trigger, drawn above the grid)
    ch 1        -> CW bottom   (CosmicWatch trigger, drawn below the grid)
    ch 2..5     -> A1, C1, A2, C2  -- unconnected: greyed out
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

# ----------------------------------------------------------------------------
PedestalAndNoiseFile = "Data/Run124_list_pedestalsAndNoise.txt"
LGScaleFactor = "Data/Run125_list_LGscale.txt"
# ----------------------------------------------------------------------------

NCH = 64
COLS = ["A", "B", "C", "D"]          # left to right
NROW = 16                            # 0 at the bottom, 15 at the top
UNCONNECTED = {2, 3, 4, 5}           # real channels, but nothing plugged in

HG_SAT = 4000.0     # raw HG at or above this counts as saturated
LG_SAT = 4000.0     # raw LG at or above this: even the LG branch is saturated
NEVENTS = 100       # events drawn by default

# --normalize: the vertical bar at the lower left runs 0 .. TOTAL_MAX ADC.
# 250000 is the round number for Run126 -- its per-event sums over the 58
# image cells peak just under 250 k (median ~30 k, 99th percentile ~155 k).
# Fixed rather than per-run so the bars stay comparable between plots;
# --total-max 0 recomputes it from the events actually drawn.
TOTAL_MAX = 250000.0


# ----------------------------------------------------------------------------
# channel map
# ----------------------------------------------------------------------------
def build_channel_map():
    """ch -> (name, column letter or None, row or None)."""
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
UNCONNECTED_CELLS = set()
for _ch in UNCONNECTED:
    _, _col, _row = CHMAP[_ch]
    if _col is not None:
        UNCONNECTED_CELLS.add((_row, COLS.index(_col)))

# the channels that make up the 4 x 16 image, i.e. what "sum over cells" means
GRID_CH = [ch for ch, (_n, _c, _r) in sorted(CHMAP.items())
           if _c is not None and ch not in UNCONNECTED]


def grid_total(sig):
    """Sum of the signal over the cells of the image (CW channels excluded)."""
    return float(np.nansum(sig[GRID_CH]))


def round_up_nice(x):
    """Smallest 1/1.5/2/2.5/3/4/5/7.5 x 10^n that is >= x."""
    if not np.isfinite(x) or x <= 0:
        return 1.0
    e = int(np.floor(np.log10(x)))
    f = x / 10.0 ** e
    for s in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        if f <= s * (1.0 + 1e-9):
            return s * 10.0 ** e
    return 10.0 ** (e + 1)


def two_sig_figs(v):
    """'2.8', '34', '0.34', '100' -- two significant figures, no exponent."""
    if not np.isfinite(v):
        return "--"
    if v == 0:
        return "0"
    d = 1 - int(np.floor(np.log10(abs(v))))
    d = min(max(d, 0), 6)
    s = "%.*f" % (d, v)
    return "0" if s.strip("-0.") == "" else s


def run_tag(path):
    """'Run125_list.txt' -> 'Run125'; falls back to the bare file name."""
    base = os.path.basename(path)
    m = re.search(r"run[_\-\s]*0*(\d+)", base, re.IGNORECASE)
    if m:
        return "Run%s" % m.group(1)
    return os.path.splitext(base)[0]


# ----------------------------------------------------------------------------
# calibration inputs
# ----------------------------------------------------------------------------
def read_pedestals(path):
    """calculatePedestalsAndNoise.py output -> dict of 64-length arrays."""
    v = {k: np.full(NCH, np.nan) for k in
         ("hg_ped", "hg_rms", "lg_ped", "lg_rms")}
    names = [None] * NCH
    comments = []
    with open(path) as f:
        for line in f:
            s = line.rstrip("\n")
            if not s.strip():
                continue
            if s.lstrip().startswith("#"):
                comments.append(s.strip())
                continue
            p = s.split()
            if len(p) < 7:
                continue
            try:
                c = int(p[0])
                if not 0 <= c < NCH:
                    continue
                # read from the right so a name with a space still parses
                v["hg_ped"][c], v["hg_rms"][c] = float(p[-5]), float(p[-4])
                v["lg_ped"][c], v["lg_rms"][c] = float(p[-3]), float(p[-2])
                names[c] = " ".join(p[1:-5])
            except (ValueError, IndexError):
                continue
    if not np.isfinite(v["hg_ped"]).any():
        sys.exit("no pedestal rows parsed from %s" % path)
    v["name"] = names
    return v, comments


def read_lgscale(path, column="k"):
    """calculateLGScale.py output -> 64-length arrays of the scale and offset.

    Row layout: ch name k k_err c k_gm k_orig r N   (7 numbers after the name)
    """
    idx = {"k": -7, "k_gm": -4, "k_orig": -3}
    if column not in idx:
        sys.exit("unknown scale column %r (use k, k_gm or k_orig)" % column)
    k = np.full(NCH, np.nan)
    c0 = np.full(NCH, np.nan)
    nfit = np.zeros(NCH, dtype=int)
    comments = []
    with open(path) as f:
        for line in f:
            s = line.rstrip("\n")
            if not s.strip():
                continue
            if s.lstrip().startswith("#"):
                comments.append(s.strip())
                continue
            p = s.split()
            if len(p) < 9:
                continue
            try:
                ch = int(p[0])
                if not 0 <= ch < NCH:
                    continue
                k[ch] = float(p[idx[column]])
                c0[ch] = float(p[-5])
                nfit[ch] = int(float(p[-1]))
            except (ValueError, IndexError):
                continue
    if not np.isfinite(k).any():
        sys.exit("no LG scale rows parsed from %s" % path)
    return {"k": k, "c": c0, "n": nfit, "column": column}, comments


def print_inputs(ped_path, ped, ped_comments,
                 sc_path, sc, sc_comments, kfill):
    """Echo both calibration files and every number taken from them."""
    print("=" * 92)
    print("PedestalAndNoiseFile = %s" % ped_path)
    print("  (absolute: %s)" % os.path.abspath(ped_path))
    print("LGScaleFactor        = %s   [using column '%s']"
          % (sc_path, sc["column"]))
    print("  (absolute: %s)" % os.path.abspath(sc_path))
    print("-" * 92)
    for c in ped_comments:
        print("  ped | %s" % c)
    for c in sc_comments:
        print("  scl | %s" % c)
    print("-" * 92)
    print("  %4s %10s %11s %11s %11s %11s %11s %8s %8s"
          % ("ch", "name", "HG_ped", "HG_noise", "LG_ped", "LG_noise",
             "LGscale", "fit_N", "note"))
    for c in range(NCH):
        nm = ped["name"][c] if ped["name"][c] else CHMAP[c][0].replace(" ", "")
        if not np.isfinite(ped["hg_ped"][c]):
            print("  %4d %10s %11s %11s %11s %11s %11s %8s %8s"
                  % (c, nm, "-", "-", "-", "-", "-", "-", "NO PEDESTAL"))
            continue
        if np.isfinite(sc["k"][c]):
            ks, note = "%11.4f" % sc["k"][c], ""
        else:
            ks, note = "%11.4f" % kfill, "filled"
        print("  %4d %10s %11.4f %11.4f %11.4f %11.4f %s %8d %8s"
              % (c, nm, ped["hg_ped"][c], ped["hg_rms"][c],
                 ped["lg_ped"][c], ped["lg_rms"][c], ks, sc["n"][c], note))
    nfit = int(np.isfinite(sc["k"]).sum())
    print("-" * 92)
    print("  pedestals for %d/%d channels, LG scale fitted for %d/%d "
          "(median %.4f; unfitted channels use %.4f)"
          % (int(np.isfinite(ped["hg_ped"]).sum()), NCH, nfit, NCH,
             np.nanmedian(sc["k"]), kfill))
    print("=" * 92)


# ----------------------------------------------------------------------------
# reading the run
# ----------------------------------------------------------------------------
def parse_events(path, max_events=None):
    """Parse a Janus list file into per-event 64-channel LG/HG arrays.

    Returns (header, events).  Each event is a dict with evt, trgid,
    tstamp_us, tref_us, and lg/hg arrays of length 64 (NaN where the channel
    did not appear).
    """
    header = {}
    chan_cols = None
    i_lg = i_hg = None
    events = []
    cur = None

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
                i_lg = chan_cols.index("LG") if "LG" in chan_cols else 2
                i_hg = chan_cols.index("HG") if "HG" in chan_cols else 3
                continue

            if not parts[0].lstrip("-").isdigit():
                continue

            if chan_cols is None:                       # no header line seen
                chan_cols = ["Brd", "Ch", "LG", "HG", "ToA_ns", "ToT_ns"]
                i_lg, i_hg = 2, 3

            nchan = len(chan_cols)
            if len(parts) < i_hg + 1:
                continue                                # truncated final line

            if len(parts) > nchan:                      # starts a new event
                if max_events is not None and len(events) >= max_events:
                    break
                ev = {"evt": len(events),
                      "lg": np.full(NCH, np.nan),
                      "hg": np.full(NCH, np.nan),
                      "tstamp_us": np.nan, "tref_us": np.nan, "trgid": -1}
                extra = parts[nchan:]
                try:
                    ev["tstamp_us"] = float(extra[0])
                    ev["tref_us"] = float(extra[1])
                    ev["trgid"] = int(extra[2])
                except (ValueError, IndexError):
                    pass
                events.append(ev)
                cur = ev

            if cur is None:
                continue

            try:
                ch = int(parts[1])
                lg = float(parts[i_lg])
                hg = float(parts[i_hg])
            except ValueError:
                continue
            if not 0 <= ch < NCH:
                continue

            cur["lg"][ch] = lg
            cur["hg"][ch] = hg

    return header, events


# ----------------------------------------------------------------------------
# signal
# ----------------------------------------------------------------------------
def event_signal(ev, ped, k, hg_sat, lg_sat, use_intercept, cfit):
    """signal, saturated flag and dead flag, per channel, for one event.

    signal = HG - HG_ped, or (LG - LG_ped) * LGscale when HG is saturated.
    """
    hg, lg = ev["hg"], ev["lg"]
    hgc = hg - ped["hg_ped"]
    lgc = lg - ped["lg_ped"]

    lg_equiv = lgc * k
    if use_intercept:
        lg_equiv = lg_equiv + cfit

    sat = np.isfinite(hg) & (hg >= hg_sat)
    sig = np.where(sat, lg_equiv, hgc)
    sig = np.where(np.isfinite(sig), sig, np.nan)
    # no pedestal for that channel, or the channel was not read out
    sig[~np.isfinite(ped["hg_ped"])] = np.nan
    both = sat & np.isfinite(lg) & (lg >= lg_sat)
    return sig, sat, both


# ----------------------------------------------------------------------------
# the 2-D image
# ----------------------------------------------------------------------------
def _text_colour(rgba):
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "black" if lum > 0.55 else "white"


def draw_event(sig, sat, both, out_png, title, subtitle,
               vmin, vmax, cmap_name="inferno", logscale=False,
               normalize=False, total=None, total_max=None, pct_max=None):
    """4 columns A-D, rows 0-15 bottom to top, CW strips above and below.

    With normalize=True every value is divided by `total`, the sum over the
    cells of the image, and shown as a percentage; the colour scale then runs
    0-100 % and a vertical bar at the lower left gives `total` itself against
    a fixed 0 .. total_max axis, so the bar heights are comparable between
    events.
    """
    if normalize:
        fig = plt.figure(figsize=(6.6, 11.1))
        outer = fig.add_gridspec(5, 2,
                                 height_ratios=[1.0, 0.7, 16.0, 1.0, 3.4],
                                 width_ratios=[24, 1], hspace=0.08, wspace=0.05,
                                 left=0.10, right=0.88, top=0.912, bottom=0.035)
        ax_cb = fig.add_subplot(outer[:4, 1])
        inner = outer[4, 0].subgridspec(1, 6, wspace=0.0)
        ax_tot = fig.add_subplot(inner[0, 0])
    else:
        fig = plt.figure(figsize=(6.6, 9.6))
        outer = fig.add_gridspec(4, 2, height_ratios=[1.0, 0.7, 16.0, 1.0],
                                 width_ratios=[24, 1], hspace=0.08, wspace=0.05,
                                 left=0.10, right=0.88, top=0.895, bottom=0.04)
        ax_cb = fig.add_subplot(outer[:, 1])
        ax_tot = None
    ax_top = fig.add_subplot(outer[0, 0])
    ax_mid = fig.add_subplot(outer[2, 0])
    ax_bot = fig.add_subplot(outer[3, 0])

    if normalize:
        # every cell as a percentage of the sum over the cells
        scale = 100.0 / total if (total and np.isfinite(total) and total != 0) \
            else np.nan
        sig = sig * scale
        # 0-100 % unless the caller asked for a tighter ceiling with --zmax;
        # with 58 cells the largest share is typically only a few per cent, so
        # --zmax 10 is often the readable choice
        vmin = 0.0
        vmax = 10.0 if pct_max is None else pct_max

    grid = np.full((NROW, len(COLS)), np.nan)
    satgrid = np.zeros((NROW, len(COLS)), dtype=bool)
    bothgrid = np.zeros((NROW, len(COLS)), dtype=bool)
    for ch, (name, col, row) in CHMAP.items():
        if col is None or ch in UNCONNECTED:
            continue
        j = COLS.index(col)
        grid[row, j] = sig[ch]
        satgrid[row, j] = sat[ch]
        bothgrid[row, j] = both[ch]

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#e8e8e8")
    if logscale:
        norm = matplotlib.colors.LogNorm(vmin=max(vmin, 1.0), vmax=vmax)
        shown = np.clip(grid, max(vmin, 1.0), None)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        shown = grid

    im = ax_mid.imshow(np.ma.masked_invalid(shown), origin="lower",
                       extent=[0, len(COLS), 0, NROW], aspect="auto",
                       cmap=cmap, norm=norm, interpolation="nearest")

    for r in range(NROW):
        for c in range(len(COLS)):
            v = grid[r, c]
            x, y = c + 0.5, r + 0.5
            if not np.isfinite(v):
                lab = "n/c" if (r, c) in UNCONNECTED_CELLS else "--"
                ax_mid.text(x, y, lab, ha="center", va="center",
                            fontsize=6, color="#8a8a8a")
                continue
            col = _text_colour(cmap(norm(max(v, norm.vmin))))
            lab = two_sig_figs(v) if normalize else "%.0f" % v
            if lab == "-0":
                lab = "0"
            ax_mid.text(x, y, lab, ha="center", va="center",
                        fontsize=6.5, color=col)
            if satgrid[r, c]:
                # HG saturated: this cell came from LG * scale
                ax_mid.add_patch(Rectangle((c, r), 1, 1, fill=False, lw=1.6,
                                           ec="#00e5ff", zorder=3))
                ax_mid.text(c + 0.95, r + 0.93,
                            "S*" if bothgrid[r, c] else "S",
                            ha="right", va="top", fontsize=5.5,
                            color="#00e5ff", zorder=4)

    ax_mid.set_xticks(np.arange(len(COLS)) + 0.5)
    ax_mid.set_xticklabels(COLS, fontsize=9)
    ax_mid.xaxis.tick_top()
    ax_mid.xaxis.set_label_position("top")
    ax_mid.set_yticks(np.arange(NROW) + 0.5)
    ax_mid.set_yticklabels([str(i) for i in range(NROW)], fontsize=6)
    ax_mid.set_xticks(np.arange(len(COLS) + 1), minor=True)
    ax_mid.set_yticks(np.arange(NROW + 1), minor=True)
    ax_mid.grid(which="minor", color="white", lw=0.6)
    ax_mid.tick_params(which="minor", length=0)
    ax_mid.tick_params(which="major", length=2)

    for ax, ch in ((ax_top, 0), (ax_bot, 1)):
        v = sig[ch]
        ax.set_xlim(0, len(COLS))
        ax.set_ylim(0, 1)
        if np.isfinite(v):
            fc = cmap(norm(max(v, norm.vmin)))
            tc = _text_colour(fc)
            txt = "%s   %s%s" % (CHMAP[ch][0],
                                 two_sig_figs(v) if normalize else "%.0f" % v,
                                 "  S" if sat[ch] else "")
        else:
            fc, tc, txt = "#e8e8e8", "#555555", "%s   --" % CHMAP[ch][0]
        ax.add_patch(Rectangle((1.0, 0.06), 2.0, 0.88, facecolor=fc,
                               edgecolor="#00e5ff" if sat[ch] else "0.35",
                               lw=1.6 if sat[ch] else 0.8))
        ax.text(2.0, 0.5, txt, ha="center", va="center", fontsize=7, color=tc)
        ax.text(0.06, 0.5, "ch%d" % ch, ha="left", va="center",
                fontsize=6, color="0.4")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    cb = fig.colorbar(im, cax=ax_cb, extend="max")
    if normalize:
        cb.set_label("share of the event total [%]", fontsize=7)
        cb.set_ticks(matplotlib.ticker.MaxNLocator(nbins=10, steps=[1, 2, 5, 10])
                     .tick_values(0.0, vmax))
        cb.ax.set_ylim(0, vmax)
    else:
        cb.set_label("signal [HG ADC, pedestal subtracted]", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # ---- the total that everything above was divided by ----
    if normalize:
        tmax = total_max if (total_max and total_max > 0) else 1.0
        val = total if np.isfinite(total) else 0.0
        ax_tot.bar([0], [max(val, 0.0)], width=0.62, color="0.35",
                   edgecolor="0.15", lw=0.6, zorder=2)
        ax_tot.set_xlim(-0.5, 0.5)
        ax_tot.set_ylim(0, tmax)
        ax_tot.set_xticks([])
        ticks = matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]) \
            .tick_values(0.0, tmax)
        ticks = [t for t in ticks if 0 <= t <= tmax]
        ax_tot.set_yticks(ticks)
        ax_tot.set_yticklabels(["%g" % (t / 1000.0) for t in ticks],
                               fontsize=5.5)
        ax_tot.set_ylabel("total [k ADC]", fontsize=6, labelpad=2)
        ax_tot.set_xlabel("sum over cells", fontsize=6.5, labelpad=2)
        ax_tot.tick_params(length=2, pad=1.5)
        ax_tot.grid(axis="y", color="0.85", lw=0.5, zorder=0)
        ax_tot.set_axisbelow(True)
        for sp in ("top", "right"):
            ax_tot.spines[sp].set_visible(False)
        # the number itself: inside the bar when there is room, else above it
        top = min(max(val, 0.0), tmax)
        inside = top > 0.18 * tmax
        ax_tot.annotate("%.0f" % val, xy=(0, top),
                        xytext=(0, -4 if inside else 3),
                        textcoords="offset points", ha="center",
                        va="top" if inside else "bottom", fontsize=6.5,
                        color="white" if inside else "0.15", clip_on=False,
                        zorder=3)
        if val > tmax:
            ax_tot.annotate("", xy=(0, tmax), xytext=(0, 0.93 * tmax),
                            arrowprops=dict(arrowstyle="-|>", color="0.15",
                                            lw=0.8))

    fig.suptitle(title, fontsize=13, y=0.985 if not normalize else 0.987)
    fig.text(0.5, 0.945 if not normalize else 0.955, subtitle, ha="center",
             va="top", fontsize=8, color="0.25")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("listfile", help="Janus list text file")
    p.add_argument("-n", "--nevents", type=int, default=NEVENTS,
                   help="number of events to draw (default %d)" % NEVENTS)
    p.add_argument("--ped", default=PedestalAndNoiseFile,
                   help="pedestal/noise file (default: %s)"
                        % PedestalAndNoiseFile)
    p.add_argument("--scale", default=LGScaleFactor,
                   help="LG scale file (default: %s)" % LGScaleFactor)
    p.add_argument("--scale-col", default="k",
                   choices=["k", "k_gm", "k_orig"],
                   help="which slope column of the LG scale file to use "
                        "(default k)")
    p.add_argument("--use-intercept", action="store_true",
                   help="use LGcorr*k + c instead of LGcorr*k for saturated "
                        "channels")
    p.add_argument("--hg-sat", type=float, default=HG_SAT,
                   help="raw HG at or above this is saturated (default %g)"
                        % HG_SAT)
    p.add_argument("--skip-events", type=int, default=1,
                   help="drop this many events from the start of the run "
                        "(default 1: event 0 is a start-of-run artifact)")
    p.add_argument("--zmax", type=float, default=None,
                   help="fix the top of the colour scale (default: 99th "
                        "percentile over the events drawn)")
    p.add_argument("--zmin", type=float, default=0.0,
                   help="bottom of the colour scale (default 0)")
    p.add_argument("--normalize", action="store_true",
                   help="show every cell as a percentage of the sum over the "
                        "cells instead of in ADC: colour scale 0-100 %%, cell "
                        "labels to two significant figures, and a vertical "
                        "bar at the lower left giving the total that was "
                        "divided out")
    p.add_argument("--total-max", type=float, default=TOTAL_MAX,
                   help="top of the --normalize total bar, in ADC (default "
                        "%g, the round number for Run126; 0 = pick one from "
                        "the events actually drawn)" % TOTAL_MAX)
    p.add_argument("--per-event-scale", action="store_true",
                   help="rescale the colours for every event instead of "
                        "using one common scale")
    p.add_argument("--log", action="store_true",
                   help="logarithmic colour scale")
    p.add_argument("--cmap", default="inferno", help="matplotlib colormap")
    p.add_argument("--outdir", default=None,
                   help="output folder (default: RunX_plots next to the "
                        "input file)")
    args = p.parse_args(argv)

    src = args.listfile
    for f, what in ((src, "list file"), (args.ped, "pedestal file"),
                    (args.scale, "LG scale file")):
        if not os.path.exists(f):
            sys.exit("no such %s: %s" % (what, f))

    # ---- calibration inputs ----
    ped, ped_comments = read_pedestals(args.ped)
    sc, sc_comments = read_lgscale(args.scale, args.scale_col)
    kfill = float(np.nanmedian(sc["k"]))
    print_inputs(args.ped, ped, ped_comments, args.scale, sc, sc_comments,
                 kfill)

    k = np.where(np.isfinite(sc["k"]), sc["k"], kfill)
    cfit = np.where(np.isfinite(sc["c"]), sc["c"], 0.0)

    # ---- events ----
    want = args.skip_events + args.nevents
    header, events = parse_events(src, want)
    events = events[args.skip_events:]
    if not events:
        sys.exit("no events to draw from %s" % src)
    events = events[:args.nevents]

    tag = run_tag(src)
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(src)),
                                         "%s_plots" % tag)
    os.makedirs(outdir, exist_ok=True)

    sigs, sats, boths = [], [], []
    for ev in events:
        s, sa, bo = event_signal(ev, ped, k, args.hg_sat, LG_SAT,
                                 args.use_intercept, cfit)
        sigs.append(s)
        sats.append(sa)
        boths.append(bo)

    allsig = np.concatenate([s[np.isfinite(s)] for s in sigs]) \
        if sigs else np.array([])
    if args.zmax is not None:
        gmax = args.zmax
    elif allsig.size:
        gmax = float(np.percentile(allsig, 99))
        gmax = max(gmax, args.zmin + 1.0)
    else:
        gmax = 1.0

    # ---- the per-event total, and the axis the --normalize bar runs on ----
    totals = [grid_total(s) for s in sigs]
    total_max = args.total_max
    auto_max = not total_max or total_max <= 0
    if args.normalize and auto_max:
        total_max = round_up_nice(max(totals) if totals else 1.0)

    print("run file   %s" % src)
    print("events     %d drawn (first %d skipped), %d channels each"
          % (len(events), args.skip_events, NCH))
    print("signal     HG - HG_ped, or (LG - LG_ped) * %s%s when raw HG >= %g"
          % (args.scale_col, " + c" if args.use_intercept else "",
             args.hg_sat))
    if args.normalize:
        print("normalize  ON: each cell / (sum over the %d image cells) * 100"
              % len(GRID_CH))
        print("colour     0 to %g %%%s   (%s; --zmin/--per-event-scale "
              "ignored)"
              % (100.0 if args.zmax is None else args.zmax,
                 "  (log)" if args.log else "",
                 "default" if args.zmax is None
                 else "--zmax caps the colour scale"))
        biggest = max((100.0 * np.nanmax(s[GRID_CH]) / t) if t else 0.0
                      for s, t in zip(sigs, totals))
        print("           largest single-cell share among these events: "
              "%.1f %%%s" % (biggest, "   <- consider --zmax %g"
                             % round_up_nice(biggest)
                             if args.zmax is None else ""))
        print("totals     min %.0f  median %.0f  max %.0f ADC   ->  bar axis "
              "0 to %g%s"
              % (min(totals), float(np.median(totals)), max(totals), total_max,
                 "  (auto, round)" if auto_max else ""))
        over = sum(1 for t in totals if t > total_max)
        if over:
            print("           %d event(s) exceed the bar axis and are drawn "
                  "clipped with an arrow; raise --total-max" % over)
    else:
        print("colour     %s, %.1f to %.1f%s"
              % ("per event" if args.per_event_scale else "common",
                 args.zmin, gmax, "  (log)" if args.log else ""))
    print("output     %s" % outdir)

    nsat_tot = 0
    for ev, s, sa, bo, tot_cells in zip(events, sigs, sats, boths, totals):
        fin = s[np.isfinite(s)]
        if args.per_event_scale:
            vmax = float(np.nanmax(fin)) if fin.size else 1.0
            vmax = max(vmax, args.zmin + 1.0)
        else:
            vmax = gmax

        nsat = int(sa.sum())
        nsat_tot += nsat
        tot = float(np.nansum(s))
        nhit = int(np.sum(fin > 0))

        title = "%s   event %d   TrgID %d" % (tag, ev["evt"], ev["trgid"])
        sub = ("t = %.3f us    sum = %.0f ADC    channels > 0 : %d/%d"
               "    HG saturated : %d" %
               (ev["tstamp_us"], tot, nhit, NCH, nsat))
        if args.normalize:
            sub += "\ncells as %% of the sum over the %d image cells " \
                   "(%.0f ADC)" % (len(GRID_CH), tot_cells)
        out_png = os.path.join(outdir, "%s_evt%05d.png" % (tag, ev["evt"]))
        draw_event(s, sa, bo, out_png, title, sub,
                   args.zmin, vmax, cmap_name=args.cmap, logscale=args.log,
                   normalize=args.normalize, total=tot_cells,
                   total_max=total_max, pct_max=args.zmax)

    print("wrote %d png files to %s" % (len(events), outdir))
    print("  %d saturated HG channels in total were replaced by LG * scale"
          % nsat_tot)


if __name__ == "__main__":
    main()
