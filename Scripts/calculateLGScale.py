#!/usr/bin/env python3
"""
calculateLGScale.py

Find the LG -> HG scale factor for every channel of a CAEN Janus / DT5202 run.

For each event and channel the pedestal is subtracted,

    HGcorr = HG - HG_pedestal
    LGcorr = LG - LG_pedestal

and HGcorr is fitted against LGcorr with a straight line over unsaturated
events (raw HG < HG_SAT).  The slope is the number you multiply LG by to get
HG.

Usage
-----
    python calculateLGScale.py RUNFILE.txt [options]

The pedestal/noise file is hardcoded below (PedestalNoiseFile) and is the
output of calculatePedestalsAndNoise.py.  --ped overrides it.  Everything
it reads is echoed to the screen before the fit.

Outputs
-------
    RUNFILE_LGscale.txt        one row per channel with the scale factor
    RUNFILE_LGscale.png        8x8 grid of LGcorr vs HGcorr with the fit
    RUNFILE_LGscale_map.png    the scale factor drawn on the detector map

This script is self-contained: it does not import anything from
calculatePedestalsAndNoise.py.

Channel map (Sept 9 assignment)
-------------------------------
    ch 0        -> CW top      (CosmicWatch trigger, drawn above the grid)
    ch 1        -> CW bottom   (CosmicWatch trigger, drawn below the grid)
    ch 2..5     -> A1, C1, A2, C2  -- unconnected: greyed out in the map
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
PedestalNoiseFile = "Data/Run124_list_pedestalsAndNoise.txt"
# ----------------------------------------------------------------------------

NCH = 64
COLS = ["A", "B", "C", "D"]          # left to right
NROW = 16                            # 0 at the bottom, 15 at the top
UNCONNECTED = {2, 3, 4, 5}           # real channels, but nothing plugged in

HG_SAT = 4000.0     # raw HG at or above this is saturated and is dropped
NSIGMA = 3.0        # keep events with HGcorr above this many HG pedestal RMS
CLIP = 4.0          # residual clipping, in sigma, during the fit
MINPTS = 20         # fewest surviving events for a fit to be attempted


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


def short_name(ch):
    """Channel name with no spaces, so output tables stay whitespace-split."""
    return CHMAP[ch][0].replace(" ", "")


def run_label(path):
    """'Run124_list.txt' -> 'Run 124'; falls back to the bare file name."""
    base = os.path.basename(path)
    m = re.search(r"run[_\-\s]*0*(\d+)", base, re.IGNORECASE)
    if m:
        return "Run %s" % m.group(1)
    return os.path.splitext(base)[0]


# ----------------------------------------------------------------------------
# reading the Janus list file
# ----------------------------------------------------------------------------
def parse_list_file(path, max_events=None):
    """Parse a Janus list file.

    Returns (header, ch, lg, hg, ev, n_events) where ch/lg/hg/ev are flat
    arrays with one entry per channel row, tagged with the event index.

    The column layout is taken from the "Brd Ch ..." header line, so the same
    code works for Spect and Spect_Timing files.
    """
    header = {}
    chan_cols = None
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
                    if k.strip() and not k.strip().startswith("*"):
                        header[k.strip()] = v.strip()
                continue

            parts = s.split()

            if parts[0] == "Brd":                      # column header line
                names = parts
                split_at = len(names)
                for i, n in enumerate(names):
                    if n.lower().startswith("tstamp"):
                        split_at = i
                        break
                chan_cols = names[:split_at]
                i_lg = chan_cols.index("LG") if "LG" in chan_cols else 2
                i_hg = chan_cols.index("HG") if "HG" in chan_cols else 3
                continue

            if not parts[0].lstrip("-").isdigit():
                continue

            if chan_cols is None:                      # no header line seen
                chan_cols = ["Brd", "Ch", "LG", "HG", "ToA_ns", "ToT_ns"]
                i_lg, i_hg = 2, 3

            nchan = len(chan_cols)
            if len(parts) < i_hg + 1:
                continue                               # truncated final line

            if len(parts) > nchan:                     # starts a new event
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
# reading the pedestal / noise file
# ----------------------------------------------------------------------------
def read_pedestals(path):
    """Read calculatePedestalsAndNoise.py output.

    Returns (values, comment_lines).  values is a dict of 64-length arrays.
    Numbers are taken from the right-hand end of each row so a channel name
    containing a space (an older "CW top") still parses.
    """
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
                v["hg_ped"][c], v["hg_rms"][c] = float(p[-5]), float(p[-4])
                v["lg_ped"][c], v["lg_rms"][c] = float(p[-3]), float(p[-2])
                names[c] = " ".join(p[1:-5])
            except (ValueError, IndexError):
                continue

    if not np.isfinite(v["hg_ped"]).any():
        sys.exit("no pedestal rows parsed from %s" % path)
    v["name"] = names
    return v, comments


def print_pedestals(path, ped, comments):
    """Echo the pedestal file and every value taken from it."""
    print("=" * 78)
    print("PedestalNoiseFile = %s" % path)
    print("  (absolute path: %s)" % os.path.abspath(path))
    print("-" * 78)
    for c in comments:
        print("  %s" % c)
    print("-" * 78)
    print("  %4s %10s %12s %12s %12s %12s" %
          ("ch", "name", "HG_ped", "HG_noise", "LG_ped", "LG_noise"))
    nmiss = 0
    for c in range(NCH):
        nm = ped["name"][c] if ped["name"][c] else short_name(c)
        if not np.isfinite(ped["hg_ped"][c]):
            print("  %4d %10s %12s %12s %12s %12s   <- MISSING, channel skipped"
                  % (c, nm, "-", "-", "-", "-"))
            nmiss += 1
            continue
        print("  %4d %10s %12.4f %12.4f %12.4f %12.4f"
              % (c, nm, ped["hg_ped"][c], ped["hg_rms"][c],
                 ped["lg_ped"][c], ped["lg_rms"][c]))
    ok = np.isfinite(ped["hg_ped"])
    print("-" * 78)
    print("  %d of %d channels have pedestals%s"
          % (int(ok.sum()), NCH,
             "" if nmiss == 0 else "   (%d missing)" % nmiss))
    print("  HG pedestal  mean %8.3f   range %8.3f - %8.3f"
          % (np.nanmean(ped["hg_ped"]), np.nanmin(ped["hg_ped"]),
             np.nanmax(ped["hg_ped"])))
    print("  HG noise     mean %8.3f   range %8.3f - %8.3f"
          % (np.nanmean(ped["hg_rms"]), np.nanmin(ped["hg_rms"]),
             np.nanmax(ped["hg_rms"])))
    print("  LG pedestal  mean %8.3f   range %8.3f - %8.3f"
          % (np.nanmean(ped["lg_ped"]), np.nanmin(ped["lg_ped"]),
             np.nanmax(ped["lg_ped"])))
    print("  LG noise     mean %8.3f   range %8.3f - %8.3f"
          % (np.nanmean(ped["lg_rms"]), np.nanmin(ped["lg_rms"]),
             np.nanmax(ped["lg_rms"])))
    print("=" * 78)


# ----------------------------------------------------------------------------
# the fit
# ----------------------------------------------------------------------------
def fit_channel(lgc, hgc, clip=CLIP):
    """Fit HGcorr = k * LGcorr + c.

    Returns a dict with
        k        slope of the ordinary least-squares fit of HG on LG
        kerr     its uncertainty
        c        intercept
        kgm      sigma(HG)/sigma(LG), the geometric-mean slope.  Immune to the
                 regression dilution that pulls the OLS slope low when LG
                 carries real scatter of its own.
        korig    best scale through the origin, sum(LG*HG)/sum(LG*LG)
        r        Pearson correlation
        n        events used after clipping
    """
    out = {k: np.nan for k in ("k", "kerr", "c", "kgm", "korig", "r")}
    out["n"] = 0
    n = lgc.size
    if n < MINPTS:
        return out

    keep = np.ones(n, dtype=bool)
    k = c = np.nan
    for _ in range(5):
        if keep.sum() < MINPTS:
            break
        x, y = lgc[keep], hgc[keep]
        if x.std() == 0:
            break
        A = np.vstack([x, np.ones(x.size)]).T
        (k, c) = np.linalg.lstsq(A, y, rcond=None)[0]
        resid = hgc - (k * lgc + c)
        s = resid[keep].std(ddof=2 if keep.sum() > 2 else 0)
        if not np.isfinite(s) or s == 0:
            break
        new = np.abs(resid) < clip * s
        if new.sum() < MINPTS or np.array_equal(new, keep):
            keep = new if new.sum() >= MINPTS else keep
            break
        keep = new

    if keep.sum() < MINPTS or not np.isfinite(k):
        return out

    x, y = lgc[keep], hgc[keep]
    m = x.size
    sxx = np.sum((x - x.mean()) ** 2)
    resid = y - (k * x + c)
    s2 = np.sum(resid ** 2) / max(m - 2, 1)

    out["k"] = float(k)
    out["kerr"] = float(np.sqrt(s2 / sxx)) if sxx > 0 else np.nan
    out["c"] = float(c)
    out["kgm"] = float(y.std() / x.std()) if x.std() > 0 else np.nan
    denom = np.sum(x * x)
    out["korig"] = float(np.sum(x * y) / denom) if denom > 0 else np.nan
    if x.std() > 0 and y.std() > 0:
        out["r"] = float(np.corrcoef(x, y)[0, 1])
    out["n"] = int(m)
    return out


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------
def scatter_grid(per_ch, fits, out_png, suptitle):
    """8x8 panels: LGcorr (y) versus HGcorr (x) with the fitted line."""
    fig, axes = plt.subplots(8, 8, figsize=(20, 20))
    for c in range(NCH):
        ax = axes[c // 8][c % 8]
        lgc, hgc = per_ch[c]
        f = fits[c]

        if lgc.size:
            ax.hexbin(hgc, lgc, gridsize=35, bins="log", cmap="Blues",
                      mincnt=1, linewidths=0.0)
            xlo, xhi = np.percentile(hgc, [0.2, 99.8])
            ylo, yhi = np.percentile(lgc, [0.2, 99.8])
            pad = 0.05 * max(xhi - xlo, 1.0)
            ax.set_xlim(xlo - pad, xhi + pad)
            pad = 0.05 * max(yhi - ylo, 1.0)
            ax.set_ylim(ylo - pad, yhi + pad)

        if np.isfinite(f["k"]) and f["k"] != 0:
            # fit is HG = k*LG + c, drawn here in the (HG, LG) plane
            hx = np.array(ax.get_xlim())
            ax.plot(hx, (hx - f["c"]) / f["k"], "-", color="#d62728", lw=1.2)

        ax.axhline(0, color="0.75", lw=0.5, zorder=0)
        ax.axvline(0, color="0.75", lw=0.5, zorder=0)

        tag = "ch%d  %s" % (c, CHMAP[c][0])
        if c in UNCONNECTED:
            tag += "  (n/c)"
        ax.set_title(tag, fontsize=8, pad=3)
        if np.isfinite(f["k"]):
            ax.text(0.03, 0.95, "HG/LG = %.2f\nN = %d" % (f["k"], f["n"]),
                    transform=ax.transAxes, ha="left", va="top", fontsize=7,
                    color="#b2182b")
        else:
            ax.text(0.03, 0.95, "no fit", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7, color="0.5")
        ax.tick_params(labelsize=6)
        if c // 8 == 7:
            ax.set_xlabel("HGcorr [ADC]", fontsize=7)
        if c % 8 == 0:
            ax.set_ylabel("LGcorr [ADC]", fontsize=7)

    fig.suptitle(suptitle, fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _text_colour(rgba):
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "black" if lum > 0.55 else "white"


def scale_map(values, out_png, suptitle, cmap_name="cividis"):
    """The per-channel scale factor on the A/B/C/D detector map.

    4 columns A,B,C,D left to right, rows 0-15 bottom to top, with the two
    CosmicWatch trigger channels above and below the grid.
    """
    fig = plt.figure(figsize=(6.4, 9.4))
    outer = fig.add_gridspec(4, 2, height_ratios=[1.0, 0.7, 16.0, 1.0],
                             width_ratios=[24, 1], hspace=0.08, wspace=0.05,
                             left=0.10, right=0.88, top=0.90, bottom=0.04)
    ax_top = fig.add_subplot(outer[0, 0])
    ax_mid = fig.add_subplot(outer[2, 0])
    ax_bot = fig.add_subplot(outer[3, 0])
    ax_cb = fig.add_subplot(outer[:, 1])

    grid = np.full((NROW, len(COLS)), np.nan)
    for ch, (name, col, row) in CHMAP.items():
        if col is None or ch in UNCONNECTED:
            continue
        grid[row, COLS.index(col)] = values[ch]

    conn = np.array([values[c] for c in range(NCH)
                     if c not in UNCONNECTED and np.isfinite(values[c])])
    if conn.size:
        vmin, vmax = np.percentile(conn, [2, 98])
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5
    else:
        vmin, vmax = 0.0, 1.0

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#e8e8e8")
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    im = ax_mid.imshow(np.ma.masked_invalid(grid), origin="lower",
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
            ax_mid.text(x, y, "%.2f" % v, ha="center", va="center",
                        fontsize=6.5, color=_text_colour(cmap(norm(v))))

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
        v = values[ch]
        ax.set_xlim(0, len(COLS))
        ax.set_ylim(0, 1)
        if np.isfinite(v):
            fc = cmap(norm(v))
            tc = _text_colour(fc)
            txt = "%s   %.2f" % (CHMAP[ch][0], v)
        else:
            fc, tc, txt = "#e8e8e8", "#555555", "%s   --" % CHMAP[ch][0]
        ax.add_patch(Rectangle((1.0, 0.06), 2.0, 0.88, facecolor=fc,
                               edgecolor="0.35", lw=0.8))
        ax.text(2.0, 0.5, txt, ha="center", va="center", fontsize=7, color=tc)
        ax.text(0.06, 0.5, "ch%d" % ch, ha="left", va="center",
                fontsize=6, color="0.4")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    ax_top.set_title("HG / LG slope", fontsize=10, pad=6)
    lo = bool(conn.size and conn.min() < vmin)
    hi = bool(conn.size and conn.max() > vmax)
    extend = ("both" if lo and hi else "min" if lo else "max" if hi
              else "neither")
    cb = fig.colorbar(im, cax=ax_cb, extend=extend)
    cb.set_label("HG per LG", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    fig.suptitle(suptitle, fontsize=11, y=0.975)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------------
def write_table(fits, out_txt, src, ped_file, header, cuts):
    with open(out_txt, "w") as f:
        f.write("# LG -> HG scale factors from %s\n" % os.path.basename(src))
        f.write("# pedestals from %s\n" % ped_file)
        for k in ("Board", "Acquisition Mode", "Run start time"):
            if k in header:
                f.write("# %s: %s\n" % (k, header[k]))
        f.write("# cuts: %s\n" % cuts)
        f.write("# HGcorr = k * LGcorr + c   ->  multiply LG by k to get HG\n")
        f.write("# k_gm = sigma(HGcorr)/sigma(LGcorr), less biased than the\n"
                "#        OLS slope k when LGcorr has scatter of its own\n")
        f.write("# k_orig = best scale through the origin\n")
        f.write("#%4s %10s %10s %9s %10s %10s %10s %8s %8s\n"
                % ("ch", "name", "k", "k_err", "c", "k_gm", "k_orig",
                   "r", "N"))
        for c in range(NCH):
            fi = fits[c]
            name = short_name(c) + ("*" if c in UNCONNECTED else "")
            f.write("%5d %10s %10.4f %9.4f %10.4f %10.4f %10.4f %8.4f %8d\n"
                    % (c, name, fi["k"], fi["kerr"], fi["c"],
                       fi["kgm"], fi["korig"], fi["r"], fi["n"]))
        f.write("# * = unconnected channel\n")


# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("listfile", help="Janus list text file")
    p.add_argument("--ped", default=PedestalNoiseFile,
                   help="pedestal/noise file (default: %s)" % PedestalNoiseFile)
    p.add_argument("--hg-sat", type=float, default=HG_SAT,
                   help="drop events with raw HG >= this (default %g)" % HG_SAT)
    p.add_argument("--nsigma", type=float, default=NSIGMA,
                   help="keep events with HGcorr above this many HG pedestal "
                        "RMS (default %g); 0 keeps the pedestal blob, which "
                        "biases the slope low" % NSIGMA)
    p.add_argument("--min-hg", type=float, default=None,
                   help="absolute HGcorr threshold in ADC, instead of --nsigma")
    p.add_argument("--keep-clipped", action="store_true",
                   help="keep events with raw HG == 0 (baseline undershoot "
                        "clipped by the ADC); they are dropped by default")
    p.add_argument("--skip-events", type=int, default=1,
                   help="drop this many events from the start of the run "
                        "(default 1: event 0 is a start-of-run artifact)")
    p.add_argument("--max-events", type=int, default=None,
                   help="stop after this many events")
    p.add_argument("--runlabel", default=None, help="title text for the plots")
    p.add_argument("-o", "--outprefix", default=None,
                   help="output prefix (default: input file without .txt)")
    p.add_argument("--no-plot", action="store_true", help="skip the figures")
    args = p.parse_args(argv)

    src = args.listfile
    if not os.path.exists(src):
        sys.exit("no such file: %s" % src)
    if not os.path.exists(args.ped):
        sys.exit("no such pedestal file: %s\n(edit PedestalNoiseFile at the "
                 "top of this script, or pass --ped)" % args.ped)

    prefix = args.outprefix
    if prefix is None:
        prefix = src[:-4] if src.lower().endswith(".txt") else src
    out_txt = prefix + "_LGscale.txt"
    out_png = prefix + "_LGscale.png"
    out_map = prefix + "_LGscale_map.png"

    # ---- pedestals: read and echo everything that will be used ----
    ped, comments = read_pedestals(args.ped)
    print_pedestals(args.ped, ped, comments)

    header, ch, lg, hg, ev, n_events = parse_list_file(src, args.max_events)
    if ch.size == 0:
        sys.exit("no data rows found in %s" % src)

    # ---- event loop: pedestal subtraction and cuts, channel by channel ----
    good = ev >= args.skip_events
    ch, lg, hg = ch[good], lg[good], hg[good]

    per_ch, fits = {}, {}
    for c in range(NCH):
        m = ch == c
        rawlg, rawhg = lg[m].astype(float), hg[m].astype(float)

        if not np.isfinite(ped["hg_ped"][c]) or not np.isfinite(ped["lg_ped"][c]):
            per_ch[c] = (np.array([]), np.array([]))
            fits[c] = fit_channel(np.array([]), np.array([]))
            continue

        sel = rawhg < args.hg_sat
        if not args.keep_clipped:
            sel &= rawhg > 0
        lgc = rawlg - ped["lg_ped"][c]
        hgc = rawhg - ped["hg_ped"][c]

        if args.min_hg is not None:
            sel &= hgc > args.min_hg
        elif args.nsigma > 0 and np.isfinite(ped["hg_rms"][c]):
            sel &= hgc > args.nsigma * ped["hg_rms"][c]

        per_ch[c] = (lgc[sel], hgc[sel])
        fits[c] = fit_channel(lgc[sel], hgc[sel])

    cuts = "raw HG < %g" % args.hg_sat
    if not args.keep_clipped:
        cuts += ", raw HG > 0"
    if args.min_hg is not None:
        cuts += ", HGcorr > %g ADC" % args.min_hg
    elif args.nsigma > 0:
        cuts += ", HGcorr > %g x HG pedestal RMS" % args.nsigma
    cuts += ", first %d event(s) skipped" % args.skip_events

    write_table(fits, out_txt, src, args.ped, header, cuts)

    if not args.no_plot:
        label = args.runlabel or run_label(src)
        scatter_grid(per_ch, fits, out_png,
                     "LG to HG scale   -   %s   |   %s" % (label, cuts))
        kvals = np.array([fits[c]["k"] for c in range(NCH)])
        scale_map(kvals, out_map, "LG to HG scale   -   %s" % label)

    # ---- console summary ----
    ok = np.array([c for c in range(NCH)
                   if np.isfinite(fits[c]["k"]) and c not in UNCONNECTED])
    print("parsed %d events from %s" % (n_events, os.path.basename(src)))
    print("cuts: %s" % cuts)
    if ok.size:
        k = np.array([fits[c]["k"] for c in ok])
        kg = np.array([fits[c]["kgm"] for c in ok])
        nn = np.array([fits[c]["n"] for c in ok])
        print("%d channels fitted, median events used %d"
              % (ok.size, int(np.median(nn))))
        print("  HG/LG slope   median %.3f   mean %.3f +- %.3f   "
              "range %.3f - %.3f" % (np.median(k), k.mean(), k.std(),
                                     k.min(), k.max()))
        print("  k_gm          median %.3f   (sigma ratio, dilution-free)"
              % np.median(kg))
        print("  -> LG * %.3f  ~  HG" % np.median(k))
        bad = [c for c in range(NCH)
               if c not in UNCONNECTED and not np.isfinite(fits[c]["k"])]
        if bad:
            print("  no fit for channels: %s"
                  % ", ".join(str(b) for b in bad))
    else:
        print("no channel could be fitted - loosen --nsigma or check the cuts")
    print("wrote %s" % out_txt)
    if not args.no_plot:
        print("wrote %s" % out_png)
        print("wrote %s" % out_map)


if __name__ == "__main__":
    main()
