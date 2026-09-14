#!/usr/bin/env python3
"""
makeTotalPhotonHisto.py

Total light per event, from data and from MC, and the ADC-per-photon scale
factor that makes the two shapes agree.

Usage
-----
    python makeTotalPhotonHisto.py DATATEXTFILE MCTEXTFILE [-n 100000]
    python makeTotalPhotonHisto.py MCTEXTFILE                (MC only)

MC file format
--------------
One event is 9 non-blank lines, events separated by a blank line:

    line 1      8 numbers    event header (truth / summary quantities)
    lines 2-5   16 numbers   per-SiPM times, rows A B C D, -999999 = no hit
    lines 6-9   16 numbers   per-SiPM photon counts, rows A B C D

Within a row the 16 columns are SiPM 0 .. 15, giving A0..A15, B0..B15,
C0..C15, D0..D15 -- 64 SiPMs in total.

Data file format
----------------
A CAEN Janus / DT5202 list file.  The signal is built exactly as in
makeEventDisplays.py:

    HGcorr = HG - HG_pedestal
    LGcorr = LG - LG_pedestal
    signal = HGcorr                   normally
           = LGcorr * LGscale         when raw HG is saturated

with the pedestal and LG-scale files hardcoded below.

What is summed -- the same 50 SiPMs in both
-------------------------------------------
All SiPMs except

    columns 0, 1, 2 of every row   ->  A0 A1 A2 B0 B1 B2 C0 C1 C2 D0 D1 D2
    C4 and C12

In the data those are exactly the channels that are unconnected or dead to
bias (ch 2, 3, 4, 5, 9, 25, 33, 35) plus the CosmicWatch trigger channels,
so both sides sum the same 50 physical SiPMs:

    A3-A15, B3-B15, C3 C5-C11 C13-C15, D3-D15

The scale factor
----------------
Data are in ADC, MC in photons.  ADCCountsPerPhoton is scanned and chosen to
minimise the shape difference between the MC histogram and the data histogram
divided by it.  Both histograms are area-normalised inside the fit window, so
only the shape matters, not the relative number of events or any overall
acceptance difference.

Output
------
    MCTEXTFILE_totalPhotons.png         MC alone, in photons
    DATATEXTFILE_totalPhotons.png       data alone, in ADC
    DATATEXTFILE_vs_MC_totalPhotons.png the overlay after scaling  <- the point

with a matching .pdf for each; --no-pdf skips them.

Channel map (Sept 9 assignment)
-------------------------------
    ch 0        -> CW top      (CosmicWatch trigger, not summed)
    ch 1        -> CW bottom   (CosmicWatch trigger, not summed)
    ch 2..5     -> A1, C1, A2, C2  -- unconnected
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

# ----------------------------------------------------------------------------
PedestalAndNoiseFile = "Data/Run130_list_pedestalsAndNoise.txt"
LGScaleFactor = "Data/Run131_list_LGscale.txt"
# ----------------------------------------------------------------------------

ROWS = ["A", "B", "C", "D"]          # the four rows, in MC file order
NCOL = 16                            # SiPMs 0 .. 15 in each row
NCH = 64                             # Janus channels

NEVENTS = 100000                     # MC events read by default

# SiPMs left out of the sum
EXCLUDE_COLS = [0, 1, 2]             # these columns in every row
EXCLUDE_CELLS = [("C", 4), ("C", 12)]

NHEAD = 8                            # numbers on the MC event header line
NTIME_ROWS = 4                       # rows of per-SiPM times
NCOUNT_ROWS = 4                      # rows of per-SiPM photon counts

UNCONNECTED = {2, 3, 4, 5}           # real channels, nothing plugged in
HG_SAT = 4000.0                      # raw HG at or above this is saturated
LG_SAT = 4000.0                      # even the LG branch is saturated

DATA_COLOUR = "black"
MC_COLOUR = "#c0392b"
LINEWIDTH = 1.0
DPI = 150


# ----------------------------------------------------------------------------
# which SiPMs are summed
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


def build_channel_map():
    """ch -> (name, row letter or None, column index or None)."""
    cmap = {0: ("CW top", None, None), 1: ("CW bottom", None, None)}
    for ch in range(2, 32):
        col, idx = ("A", ch // 2) if ch % 2 == 0 else ("C", (ch - 1) // 2)
        cmap[ch] = ("%s%d" % (col, idx), col, idx)
    for ch in range(32, 64):
        col, idx = ("B", (ch - 32) // 2) if ch % 2 == 0 \
            else ("D", (ch - 33) // 2)
        cmap[ch] = ("%s%d" % (col, idx), col, idx)
    return cmap


CHMAP = build_channel_map()

# the Janus channels behind the 50 summed SiPMs
SUM_CH = [ch for ch in range(NCH)
          if CHMAP[ch][1] is not None and ch not in UNCONNECTED
          and KEEP[ROWS.index(CHMAP[ch][1]), CHMAP[ch][2]]]


def run_tag(path):
    """'Run125_list.txt' -> 'Run125'; falls back to the bare file name."""
    base = os.path.basename(path)
    m = re.search(r"run[_\-\s]*0*(\d+)", base, re.IGNORECASE)
    if m:
        return "Run%s" % m.group(1)
    return os.path.splitext(base)[0]


# ----------------------------------------------------------------------------
# MC
# ----------------------------------------------------------------------------
def read_mc(path, max_events=None):
    """Per-event totals over the 50 summed SiPMs, plus the per-SiPM sums.

    An event starts at a line with NHEAD numbers; the four count rows are the
    last four of the eight 16-number rows that follow it.  Blank lines and
    anything unparsable are skipped, so a trailing partial event is dropped.
    """
    totals = []
    per_sipm = np.zeros((len(ROWS), NCOL), dtype=np.int64)
    nneg = 0
    rows = []
    in_event = False

    def finish():
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
                continue
            if len(vals) == NHEAD:                 # starts a new event
                if finish():
                    break
                in_event = True
            elif len(vals) == NCOL and in_event:
                rows.append(vals)

    if not (max_events is not None and len(totals) >= max_events):
        finish()

    return np.asarray(totals, dtype=float), per_sipm, len(totals), nneg


# ----------------------------------------------------------------------------
# data: calibration inputs  (identical to makeEventDisplays.py)
# ----------------------------------------------------------------------------
def read_pedestals(path):
    """calculatePedestalsAndNoise.py output -> dict of 64-length arrays."""
    v = {k: np.full(NCH, np.nan) for k in
         ("hg_ped", "hg_rms", "lg_ped", "lg_rms")}
    names = [None] * NCH
    with open(path) as f:
        for line in f:
            s = line.rstrip("\n")
            if not s.strip() or s.lstrip().startswith("#"):
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
    return v


def read_lgscale(path, column="k"):
    """calculateLGScale.py output -> the per-channel scale and offset.

    Row layout: ch name k k_err c k_gm k_orig r N   (7 numbers after the name)
    """
    idx = {"k": -7, "k_gm": -4, "k_orig": -3}
    if column not in idx:
        sys.exit("unknown scale column %r (use k, k_gm or k_orig)" % column)
    k = np.full(NCH, np.nan)
    c0 = np.full(NCH, np.nan)
    nfit = np.zeros(NCH, dtype=int)
    with open(path) as f:
        for line in f:
            s = line.rstrip("\n")
            if not s.strip() or s.lstrip().startswith("#"):
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
    return {"k": k, "c": c0, "n": nfit, "column": column}


# ----------------------------------------------------------------------------
# data: reading the run and building the signal
# ----------------------------------------------------------------------------
def parse_events(path, max_events=None):
    """Janus list file -> list of events with 64-channel LG/HG arrays."""
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
                      "tstamp_us": np.nan, "trgid": -1}
                extra = parts[nchan:]
                try:
                    ev["tstamp_us"] = float(extra[0])
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
    sig[~np.isfinite(ped["hg_ped"])] = np.nan
    both = sat & np.isfinite(lg) & (lg >= lg_sat)
    return sig, sat, both


def read_data(path, ped_path, scale_path, scale_col, use_intercept,
              hg_sat, skip_events, max_events):
    """Per-event ADC totals over the same 50 SiPMs, from a Janus list file."""
    ped = read_pedestals(ped_path)
    sc = read_lgscale(scale_path, scale_col)
    kfill = float(np.nanmedian(sc["k"]))
    k = np.where(np.isfinite(sc["k"]), sc["k"], kfill)
    cfit = np.where(np.isfinite(sc["c"]), sc["c"], 0.0)

    want = None if max_events is None else skip_events + max_events
    header, events = parse_events(path, want)
    events = events[skip_events:]
    if max_events is not None:
        events = events[:max_events]
    if not events:
        sys.exit("no events left in %s after skipping %d" % (path, skip_events))

    totals = np.empty(len(events), dtype=float)
    per_ch = np.zeros(NCH, dtype=float)
    nsat = 0
    for i, ev in enumerate(events):
        s, sa, _ = event_signal(ev, ped, k, hg_sat, LG_SAT, use_intercept,
                                cfit)
        totals[i] = float(np.nansum(s[SUM_CH]))
        per_ch += np.where(np.isfinite(s), s, 0.0)
        nsat += int(sa[SUM_CH].sum())

    info = {"header": header, "ped": ped, "sc": sc, "kfill": kfill,
            "per_ch": per_ch / len(events), "nsat": nsat,
            "nev": len(events)}
    return totals, info


# ----------------------------------------------------------------------------
# the scale factor
# ----------------------------------------------------------------------------
def shape_chi2(dat_ph, mc, edges):
    """Reduced chi2 between two area-normalised histograms on `edges`."""
    hd, _ = np.histogram(dat_ph, bins=edges)
    hm, _ = np.histogram(mc, bins=edges)
    nd, nm = hd.sum(), hm.sum()
    if nd < 10 or nm < 10:
        return np.inf
    a = hd / float(nd)
    b = hm / float(nm)
    var = a / float(nd) + b / float(nm)          # binomial variance of each
    m = var > 0
    if m.sum() < 2:
        return np.inf
    return float(np.sum((a[m] - b[m]) ** 2 / var[m])) / (m.sum() - 1)


def shape_ks(dat_ph, mc, lo, hi):
    """KS distance between the two samples restricted to [lo, hi]."""
    d = np.sort(dat_ph[(dat_ph >= lo) & (dat_ph <= hi)])
    m = np.sort(mc[(mc >= lo) & (mc <= hi)])
    if d.size < 10 or m.size < 10:
        return np.inf
    grid = np.union1d(d, m)
    cd = np.searchsorted(d, grid, side="right") / float(d.size)
    cm = np.searchsorted(m, grid, side="right") / float(m.size)
    return float(np.max(np.abs(cd - cm)))


def fit_scale(data_adc, mc, lo, hi, nbins, metric="chi2",
              smin=None, smax=None, npoints=400):
    """Scan ADCCountsPerPhoton and return the best value and the scan curve."""
    edges = np.linspace(lo, hi, nbins + 1)

    def goodness(s):
        dp = data_adc / s
        if metric == "ks":
            return shape_ks(dp, mc, lo, hi)
        return shape_chi2(dp, mc, edges)

    s0 = np.median(data_adc) / np.median(mc)      # starting guess
    smin = smin if smin else s0 / 10.0
    smax = smax if smax else s0 * 10.0

    grid = np.geomspace(smin, smax, npoints)
    vals = np.array([goodness(s) for s in grid])
    best = int(np.nanargmin(vals))

    # refine on a fine linear grid around the coarse minimum
    ilo = max(best - 1, 0)
    ihi = min(best + 1, len(grid) - 1)
    fine = np.linspace(grid[ilo], grid[ihi], 201)
    fvals = np.array([goodness(s) for s in fine])
    sbest = float(fine[int(np.nanargmin(fvals))])

    return sbest, grid, vals, float(np.nanmin(fvals))


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------
def single_hist(vals, out_paths, title, subtitle, xlabel, lo, hi, nbins,
                colour, logy=False, nzero=0):
    edges = np.linspace(lo, hi, nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    h, _ = np.histogram(vals, bins=edges)

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    fig.subplots_adjust(left=0.10, right=0.975, top=0.845, bottom=0.10)

    ax.step(centres, h, where="mid", color=colour, lw=LINEWIDTH,
            solid_joinstyle="miter")
    ax.fill_between(centres, h, step="mid", color=colour, alpha=0.08)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("events / bin", fontsize=11)
    ax.set_xlim(lo, hi)
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 1.10 * h.max() if h.size and h.max() else 1)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=9)

    q = np.percentile(vals, [50, 90, 99]) if vals.size else [0, 0, 0]
    stats = ("entries  %d\nmean     %.1f\nRMS      %.1f\nmedian   %.1f\n"
             "90 %%      %.1f\n99 %%      %.1f\nmax      %.0f"
             % (vals.size, vals.mean(),
                vals.std(ddof=1) if vals.size > 1 else 0.0,
                q[0], q[1], q[2], vals.max() if vals.size else 0))
    if nzero:
        stats += "\n(%d zeros cut)" % nzero
    ax.text(0.985, 0.97, stats, transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, family="monospace", color="0.15",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8", lw=0.6))

    fig.suptitle(title, fontsize=14, y=0.965)
    fig.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=9,
             color="0.25", linespacing=1.5)
    for path in out_paths:
        fig.savefig(path, dpi=DPI)
    plt.close(fig)


def overlay(data_ph, mc, scale, out_paths, title, subtitle, lo, hi, nbins,
            fitlo, fithi, grid, vals, metric, chi2, logy=False):
    """MC and data/ADCCountsPerPhoton on one axis, with a ratio panel."""
    edges = np.linspace(lo, hi, nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    hd, _ = np.histogram(data_ph, bins=edges)
    hm, _ = np.histogram(mc, bins=edges)

    # area-normalise inside the fit window, the region the scale was fitted on
    win = (centres >= fitlo) & (centres <= fithi)
    nd = hd[win].sum() or 1
    nm = hm[win].sum() or 1
    ad, am = hd / float(nd), hm / float(nm)
    ed = np.sqrt(hd) / float(nd)
    em = np.sqrt(hm) / float(nm)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(9.6, 7.4), sharex=True,
        gridspec_kw=dict(height_ratios=[3.0, 1.0], hspace=0.06))
    fig.subplots_adjust(left=0.10, right=0.975, top=0.845, bottom=0.095)

    ax.fill_between(centres, am, step="mid", color=MC_COLOUR, alpha=0.12)
    ax.step(centres, am, where="mid", color=MC_COLOUR, lw=LINEWIDTH + 0.2,
            solid_joinstyle="miter", label="MC  (%d events)" % mc.size)
    ax.errorbar(centres, ad, yerr=ed, fmt="o", ms=2.4, lw=0, elinewidth=0.7,
                color=DATA_COLOUR,
                label="data / %.1f  (%d events)" % (scale, data_ph.size))

    ax.axvspan(fitlo, fithi, color="0.5", alpha=0.07, lw=0, zorder=0)
    ax.set_ylabel("fraction of events / bin", fontsize=11)
    ax.set_xlim(lo, hi)
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 1.15 * max(am.max(), ad.max()))
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, frameon=False, loc="upper right")

    txt = ("ADCCountsPerPhoton = %.1f   (shape fit)\n"
           "  from median match  %.1f\n"
           "  from mean match    %.1f\n"
           "fit window  %.0f - %.0f photons\n"
           "%s at best  %.3f"
           % (scale,
              np.median(data_ph) * scale / np.median(mc)
              if np.median(mc) else np.nan,
              data_ph.mean() * scale / mc.mean() if mc.mean() else np.nan,
              fitlo, fithi,
              "reduced chi2" if metric == "chi2" else "KS distance", chi2))
    ax.text(0.015, 0.97, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, family="monospace", color="0.15",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8", lw=0.6))

    # ---- the goodness-of-fit scan, as an inset ----
    axi = ax.inset_axes([0.60, 0.42, 0.36, 0.30])
    good = np.isfinite(vals)
    axi.plot(grid[good], vals[good], color="0.35", lw=0.9)
    axi.axvline(scale, color=MC_COLOUR, lw=0.9, ls="--")
    axi.set_xscale("log")
    axi.set_yscale("log")
    axi.set_xlabel("ADC / photon", fontsize=6.5, labelpad=1)
    axi.set_ylabel("reduced chi2" if metric == "chi2" else "KS", fontsize=6.5,
                   labelpad=1)
    axi.tick_params(labelsize=5.5, length=2, pad=1)
    axi.grid(alpha=0.25, lw=0.4)

    # ---- ratio ----
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(am > 0, ad / am, np.nan)
        rerr = np.where(am > 0, np.sqrt((ed / am) ** 2 +
                                        (ad * em / am ** 2) ** 2), np.nan)
    axr.axhline(1.0, color=MC_COLOUR, lw=0.9)
    axr.errorbar(centres, ratio, yerr=rerr, fmt="o", ms=2.2, lw=0,
                 elinewidth=0.7, color=DATA_COLOUR)
    axr.axvspan(fitlo, fithi, color="0.5", alpha=0.07, lw=0, zorder=0)
    axr.set_ylim(0, 2)
    axr.set_ylabel("data / MC", fontsize=9)
    axr.set_xlabel("total photons in the 50 summed SiPMs", fontsize=11)
    axr.grid(alpha=0.25, lw=0.5)
    axr.tick_params(labelsize=9)

    fig.suptitle(title, fontsize=14, y=0.972)
    fig.text(0.5, 0.932, subtitle, ha="center", va="top", fontsize=9,
             color="0.25", linespacing=1.5)
    for path in out_paths:
        fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ----------------------------------------------------------------------------
def outputs(png, no_pdf):
    return [png] if no_pdf else [png, os.path.splitext(png)[0] + ".pdf"]


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", metavar="FILE",
                   help="DATATEXTFILE MCTEXTFILE, or just MCTEXTFILE")
    p.add_argument("-n", "--nevents", type=int, default=NEVENTS,
                   help="MC events to read (default %d; 0 = the whole file)"
                        % NEVENTS)
    p.add_argument("--ndata", type=int, default=0,
                   help="data events to read (default 0 = the whole file)")
    p.add_argument("--skip-events", type=int, default=1,
                   help="data events dropped from the start of the run "
                        "(default 1: event 0 is a start-of-run artifact)")
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
    p.add_argument("--adc-per-photon", type=float, default=None,
                   help="fix ADCCountsPerPhoton instead of fitting it")
    p.add_argument("--metric", default="chi2", choices=["chi2", "ks"],
                   help="what the scale factor minimises (default chi2)")
    p.add_argument("--fit-min", type=float, default=None,
                   help="low edge of the fit window in photons (default: the "
                        "2nd percentile of the MC)")
    p.add_argument("--fit-max", type=float, default=None,
                   help="high edge of the fit window in photons (default: the "
                        "99th percentile of the MC)")
    p.add_argument("--fit-bins", type=int, default=60,
                   help="bins used in the fit window (default 60)")
    p.add_argument("--bins", type=int, default=100,
                   help="bins in the plotted histograms (default 100)")
    p.add_argument("--xmin", type=float, default=0.0,
                   help="low edge of the x axis (default 0)")
    p.add_argument("--xmax", type=float, default=None,
                   help="high edge of the x axis (default: the 99.5th "
                        "percentile of the MC, rounded up)")
    p.add_argument("--logy", action="store_true", help="logarithmic y axis")
    p.add_argument("--keep-zeros", action="store_true",
                   help="also histogram events whose total is exactly 0 "
                        "(default: drop them; counted and reported either way)")
    p.add_argument("-o", "--out", default=None,
                   help="output png for the overlay (default: "
                        "DATATEXTFILE_vs_MC_totalPhotons.png)")
    p.add_argument("--no-pdf", action="store_true",
                   help="only write the pngs, not the matching pdfs")
    args = p.parse_args(argv)

    if len(args.files) == 1:
        data_file, mc_file = None, args.files[0]
    elif len(args.files) == 2:
        data_file, mc_file = args.files
    else:
        sys.exit("give DATATEXTFILE MCTEXTFILE, or just MCTEXTFILE")

    for f in [mc_file] + ([data_file] if data_file else []):
        if not os.path.exists(f):
            sys.exit("no such file: %s" % f)
    if data_file:
        for f, what in ((args.ped, "pedestal file"),
                        (args.scale, "LG scale file")):
            if not os.path.exists(f):
                sys.exit("no such %s: %s" % (what, f))

    summed = sipm_names(KEEP, True)
    dropped = sipm_names(KEEP, False)

    # ---------------- MC ----------------
    max_mc = None if args.nevents in (0, None) else args.nevents
    mc_all, per_sipm, nmc, nneg = read_mc(mc_file, max_mc)
    if nmc == 0:
        sys.exit("no events parsed from %s" % mc_file)
    nzero = int(np.sum(mc_all == 0))
    mc = mc_all if args.keep_zeros else mc_all[mc_all > 0]
    nzero_cut = 0 if args.keep_zeros else nzero
    if mc.size == 0:
        sys.exit("every MC event has a total of exactly 0")

    print("=" * 84)
    print("MC file     %s" % mc_file)
    print("events      %d read%s" % (nmc, "" if max_mc is None
                                     else " (limit %d)" % max_mc))
    print("summed      %d SiPMs: %s" % (len(summed), " ".join(summed)))
    print("excluded    %d SiPMs: %s" % (len(dropped), " ".join(dropped)))
    if nneg:
        print("            %d negative entries in the MC count rows set to 0"
              % nneg)
    print("zeros       %d of %d MC events have a total of exactly 0 (%.1f %%)"
          % (nzero, nmc, 100.0 * nzero / nmc))
    print("            %s"
          % ("DROPPED from the histogram (--keep-zeros to include them)"
             if nzero_cut else "kept in the histogram"))
    print("MC totals   entries %d   mean %.1f   RMS %.1f   median %.1f   "
          "max %.0f"
          % (mc.size, mc.mean(), mc.std(ddof=1) if mc.size > 1 else 0.0,
             np.median(mc), mc.max()))

    # ---------------- data ----------------
    data = info = None
    if data_file:
        max_dat = None if args.ndata in (0, None) else args.ndata
        data_all, info = read_data(data_file, args.ped, args.scale,
                                   args.scale_col, args.use_intercept,
                                   args.hg_sat, args.skip_events, max_dat)
        npos = int(np.sum(data_all > 0))
        data = data_all[data_all > 0]
        print("-" * 84)
        print("data file   %s" % data_file)
        for kk in ("Board", "Acquisition Mode", "Run start time"):
            if kk in info["header"]:
                print("  %-22s %s" % (kk, info["header"][kk]))
        print("pedestals   %s" % args.ped)
        print("LG scale    %s   [column '%s', unfitted channels use %.4f]"
              % (args.scale, args.scale_col, info["kfill"]))
        print("signal      HG - HG_ped, or (LG - LG_ped) * %s%s when raw "
              "HG >= %g" % (args.scale_col,
                            " + c" if args.use_intercept else "", args.hg_sat))
        print("events      %d read (first %d skipped)"
              % (info["nev"], args.skip_events))
        print("            %d saturated HG channels among the summed 50 were "
              "replaced by LG * scale" % info["nsat"])
        print("            %d of %d events have a positive total (%.1f %%); "
              "the rest are dropped"
              % (npos, info["nev"], 100.0 * npos / info["nev"]))
        print("data totals entries %d   mean %.0f   RMS %.0f   median %.0f   "
              "max %.0f"
              % (data.size, data.mean(),
                 data.std(ddof=1) if data.size > 1 else 0.0,
                 np.median(data), data.max()))

    # ---------------- the scale factor ----------------
    fitlo = args.fit_min if args.fit_min is not None \
        else float(np.percentile(mc, 2))
    fithi = args.fit_max if args.fit_max is not None \
        else float(np.percentile(mc, 99))

    scale, grid, vals, best = None, None, None, np.nan
    if data is not None:
        if args.adc_per_photon:
            scale = args.adc_per_photon
            edges = np.linspace(fitlo, fithi, args.fit_bins + 1)
            best = (shape_ks(data / scale, mc, fitlo, fithi)
                    if args.metric == "ks"
                    else shape_chi2(data / scale, mc, edges))
            grid = np.geomspace(scale / 10.0, scale * 10.0, 200)
            vals = np.array([shape_chi2(data / s, mc,
                                        np.linspace(fitlo, fithi,
                                                    args.fit_bins + 1))
                             for s in grid])
        else:
            scale, grid, vals, best = fit_scale(
                data, mc, fitlo, fithi, args.fit_bins, args.metric)

        print("-" * 84)
        print("scale       fit window %.0f to %.0f photons, %d bins, "
              "metric '%s'" % (fitlo, fithi, args.fit_bins, args.metric))
        print("            ADCCountsPerPhoton = %.2f%s"
              % (scale, "  (fixed by hand)" if args.adc_per_photon
                 else "  (scanned)"))
        print("            %s at the minimum: %.4f"
              % ("reduced chi2" if args.metric == "chi2" else "KS distance",
                 best))
        print("            cross-checks:  matching medians gives %.2f, "
              "matching means gives %.2f"
              % (np.median(data) / np.median(mc), data.mean() / mc.mean()))
        print("            (agreement between the three says the shapes "
              "really do match, not just one moment)")
        print("            data in photons: mean %.1f   median %.1f   max %.0f"
              % (data.mean() / scale, np.median(data) / scale,
                 data.max() / scale))

    # ---------------- plots ----------------
    if args.xmax is not None:
        hi = args.xmax
    else:
        hi = float(np.percentile(mc, 99.5))
        hi = max(np.ceil(hi / 100.0) * 100.0, args.xmin + 1.0)
    lo = args.xmin

    mc_png = "%s_totalPhotons.png" % os.path.splitext(
        os.path.abspath(mc_file))[0]
    single_hist(mc, outputs(mc_png, args.no_pdf), "Total photons per event",
                "MC: %d events read%s  |  sum over %d SiPMs\n%s"
                % (nmc, ", %d with total 0 excluded" % nzero_cut
                   if nzero_cut else "", len(summed),
                   os.path.basename(mc_file)),
                "total photons in the 50 summed SiPMs", lo, hi, args.bins,
                MC_COLOUR, logy=args.logy, nzero=nzero_cut)
    written = outputs(mc_png, args.no_pdf)

    if data is not None:
        dhi = float(np.percentile(data, 99.5))
        dhi = np.ceil(dhi / 1000.0) * 1000.0
        d_png = "%s_totalPhotons.png" % os.path.splitext(
            os.path.abspath(data_file))[0]
        single_hist(data, outputs(d_png, args.no_pdf),
                    "Total signal per event",
                    "data: %s, %d events  |  sum over the same %d SiPMs\n%s"
                    % (run_tag(data_file), data.size, len(summed),
                       os.path.basename(data_file)),
                    "total signal in the 50 summed SiPMs [ADC]", 0.0, dhi,
                    args.bins, DATA_COLOUR, logy=args.logy)
        written += outputs(d_png, args.no_pdf)

        o_png = args.out or "%s_vs_MC_totalPhotons.png" % os.path.splitext(
            os.path.abspath(data_file))[0]
        sub = ("%s (%d events) vs MC (%d events)  |  sum over the same %d "
               "SiPMs\nADCCountsPerPhoton = %.1f, fitted on %.0f-%.0f photons "
               "(shaded); both areas normalised there"
               % (run_tag(data_file), data.size, mc.size, len(summed),
                  scale, fitlo, fithi))
        overlay(data / scale, mc, scale, outputs(o_png, args.no_pdf),
                "Total photons per event: data vs MC", sub, lo, hi, args.bins,
                fitlo, fithi, grid, vals, args.metric, best, logy=args.logy)
        written += outputs(o_png, args.no_pdf)

    print("=" * 84)
    for path in written:
        print("wrote %s" % path)


if __name__ == "__main__":
    main()
