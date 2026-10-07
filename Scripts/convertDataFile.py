#!/usr/bin/env python3
"""
convertDataFile.py

Turn a CAEN Janus / DT5202 list file into the MC "CNN" text format, so real
data and simulation can go through the same downstream code.

    python convertDataFile.py Run140_list.txt
        -> Run140_list_formatted_1000photons_innerRing3.txt

Photon counts
-------------
No pedestal subtraction.  Per channel,

    photons = HG / ADC_PER_PHOTON                      normally
            = (LG * LG_SCALE) / ADC_PER_PHOTON         when HG > HG_SAT

with LG_SCALE = 10 (constant, not the fitted per-channel value), HG_SAT = 4000
and ADC_PER_PHOTON = 35.  The result is rounded to the nearest integer and
clipped at 0; a channel missing from the event counts as 0.

Geometry
--------
ch 0-5 are the CosmicWatch paddles (CW top, CW bottom, CWA-CWD) and take no
part in the image.  ch 6-63 are the SiPMs:

    ch 6..31    even -> A(row),  odd -> C(row),   row = 3 + (ch-6)//2
    ch 32..63   even -> B(row),  odd -> D(row),   row = (ch-32)//2

Rings (rows) 0-4 are dropped, leaving rings 5-15: eleven of them, which is the
width of the MC array.  The output is written with ring 15 FIRST, so

    output column  0  1  2  3  4  5  6  7  8  9 10
    detector ring 15 14 13 12 11 10  9  8  7  6  5

and the four output rows are detector columns A, B, C, D in that order --
the same order the CNN pads circularly over.

Output format
-------------
One event is five lines: the TrgID on its own, then the 4 x 11 photon counts.
There is no muon-truth header (data has no truth) and no leading placeholder
array.

    NOTE: the MC files carry a 4 x 11 block BEFORE the counts, and the CNN
    loader skips it (skip_leading_blocks = 1).  A file written here will
    therefore NOT load with that script as it stands.  Use --placeholder to
    write a block of -999999 in that slot and the output becomes drop-in
    readable by the existing loader; the trainer's truth columns will be the
    TrgID, which is meaningless, so that is for inference only.

Selection
---------
    0.  --requireTrigger (optional): ALL FOUR of ch2, ch3, ch4, ch5 (CWA,
        CWB, CWC, CWD) fired, a channel counting as fired when its RAW
        HG > 1000 OR its RAW LG > 500.  ch0 and ch1 (CW top, CW bottom) are
        measured and reported but do not gate the event.  Same criterion as
        makeEventDisplays.py --requireTrigger.  Applied first, so the photon
        fractions below are then quoted among the events that fired, and
        '_requireTrigger' is added to the output file name.
    1.  total photons over the 4 x 11 array >= 1000
    2.  the column with the most photons (summed over the 4 SiPMs in it) is
        not one of the first two (0, 1) or the last two (9, 10)

Every fraction is reported.  A four-fold AND is only as good as its worst
paddle -- one channel that stops firing takes the whole sample to zero -- so
the per-channel fire rate is always printed, cut on or not, and a required
channel that never fires is called out by name.  --trig-channels and
--trig-nmin change which channels are required and how many of them.

Diagnostic plots
----------------
Written on every run (--no-map to skip) as <output>_avgPhotons.pdf, 16 pages:

    1      average photons per channel, 16 x 4 map + the trigger channels
           (also saved on its own as <output>_avgPhotons.png)
    2-5    photon-count distributions, HG/LG combination, four rings per
           page in a 2 x 2 array (15-12, 11-8, 7-4, 3-0); in each panel the
           ring's four SiPMs A, B, C, D are black, red, green, blue
    6      the 11 SiPMs of rings 15-5 summed per column: A (black) against
           B, C, D
    7-11   the same five pages, HG only
    12-16  the same five pages, LG only

HG only is HG / ADC_PER_PHOTON and LG only is (LG * LG_SCALE) / ADC_PER_PHOTON
for every entry, with no switch between them, so all three versions are in
photons on the same bins.  HG only therefore piles up near
HG_SAT / ADC_PER_PHOTON where the ADC saturates.

The distributions are over the same events as the map (all events read, or
the written ones with --map-selected).  Bins are variable width (HIST_EDGES):
5 photons wide up to 50, 10 for 50-100, 20 for 100-200, 50 for 200-300 and
100 for 300-500, and the y axis is events per photon (bin content / bin
width) so the shape is continuous across the width changes.  Entries above
500 go into the last bin.  Under every distribution is a ratio panel: each
curve divided by the average of the curves in that panel, same binning, with
Poisson error bars (the curve's own share of the average is accounted for).
--hist-xmax and/or --hist-bins switch to uniform
bins instead.

The first few events are printed in full so the ring ordering and the
conversion can be checked by eye.
"""

import argparse
import os
import sys
import time

# ----------------------------------------------------------------------------
NCH = 64
COLS = ["A", "B", "C", "D"]      # the four output rows, in this order
RING_LO, RING_HI = 5, 15         # rings kept; 0-4 are dropped
NRING = RING_HI - RING_LO + 1    # 11 -> the width of the MC array

ADC_PER_PHOTON = 35.0
LG_SCALE = 10.0                  # constant, not the fitted per-channel k
HG_SAT = 4000.0                  # HG strictly above this uses the LG branch

MIN_PHOTONS = 1000
NEDGE = 2                        # columns barred at each end for the max
NPRINT = 5
REPORT_EVERY = 250000

# --requireTrigger.  A channel counts as fired when raw HG > TRIG_HG_MIN or
# raw LG > TRIG_LG_MIN, and the event is kept when ALL of TRIG_REQUIRED fired.
# Same criterion as makeEventDisplays.py --requireTrigger.
#
# The required set is ch2-5 (CWA, CWB, CWC, CWD) -- a four-fold AND.  ch0 and
# ch1 (CW top, CW bottom) are still measured and reported, they just do not
# gate the event.  An AND of four is fragile: one dead paddle takes the whole
# sample to zero, which is why the per-channel fire rate is printed.
TRIG_HG_MIN = 1000.0
TRIG_LG_MIN = 500.0
TRIG_REQUIRED = (2, 3, 4, 5)     # all of these must fire


# ----------------------------------------------------------------------------
def build_channel_map():
    """ch -> (column letter, ring) for the 58 SiPMs; ch 0-5 are not in it."""
    cmap = {}
    for ch in range(6, 32):
        col = "A" if ch % 2 == 0 else "C"
        cmap[ch] = (col, 3 + (ch - 6) // 2)
    for ch in range(32, 64):
        col = "B" if ch % 2 == 0 else "D"
        cmap[ch] = (col, (ch - 32) // 2)
    return cmap


CHMAP = build_channel_map()

# ch -> (output row, output column), for the channels that survive the ring cut
# output column 0 is ring 15, column NRING-1 is ring RING_LO
CELL = {}
for _ch, (_col, _ring) in CHMAP.items():
    if RING_LO <= _ring <= RING_HI:
        CELL[_ch] = (COLS.index(_col), RING_HI - _ring)

TRIG_CH = tuple(range(6))
# Sept 14 assignment.  Note ch3 is TrigC and ch4 is TrigB -- B and C are the
# other way round from the older CWB/CWC labels, so a printout from before
# Sept 14 names those two channels differently.  Only the labels moved; the
# required set below is still ch2-5, so the AND is unaffected.
TRIG_NAME = {0: "TrigE", 1: "TrigF", 2: "TrigA", 3: "TrigC", 4: "TrigB",
             5: "TrigD"}

# the full detector grid, before the ring cut: 16 rings x 4 columns
NRING_ALL = 16
RING_ALL_HI = NRING_ALL - 1


def fired_trigger_channels(hg, lg, hgmin=TRIG_HG_MIN, lgmin=TRIG_LG_MIN):
    """Which of ch0-5 fired, as a list of channel numbers.

    A channel fires when its RAW HG is strictly above hgmin OR its RAW LG is
    strictly above lgmin -- raw, not pedestal subtracted, because the
    threshold is a discriminator level on the number the board wrote down.
    The LG arm catches a paddle whose HG has saturated, which would otherwise
    fail an HG-only cut.
    """
    out = []
    for c in TRIG_CH:
        h, l = hg.get(c), lg.get(c)
        if (h is not None and h > hgmin) or (l is not None and l > lgmin):
            out.append(c)
    return out


def photons(hg, lg, adc_per_photon=ADC_PER_PHOTON, lg_scale=LG_SCALE,
            hg_sat=HG_SAT):
    """One channel's photon count.  None in -> 0 out."""
    if hg is None:
        return 0
    if hg > hg_sat:
        if lg is None:
            return 0
        v = (lg * lg_scale) / adc_per_photon
    else:
        v = hg / adc_per_photon
    n = int(round(v))
    return n if n > 0 else 0


def photons_single(adc, scale, adc_per_photon=ADC_PER_PHOTON):
    """Photon count from ONE gain alone, no HG/LG switch: adc*scale/adc_per_
    photon, rounded and clipped at 0 like photons().  scale is 1 for HG and
    LG_SCALE for LG.  None in -> 0 out."""
    if adc is None:
        return 0
    n = int(round((adc * scale) / adc_per_photon))
    return n if n > 0 else 0


# ----------------------------------------------------------------------------
def iter_events(path):
    """Stream the list file, yielding one event at a time.

    Yields (trgid, tstamp_us, hg dict, lg dict).  Only the channels that
    actually appear are in the dicts; everything else is treated as 0.
    Nothing is accumulated, so file size does not matter.
    """
    chan_cols = None
    i_lg = i_hg = 2, 3
    cur = None

    with open(path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//"):
                continue

            parts = s.split()

            if parts[0] == "Brd":                      # column header line
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

            if chan_cols is None:                      # no header line seen
                chan_cols = ["Brd", "Ch", "LG", "HG", "ToA_ns", "ToT_ns"]
                i_lg, i_hg = 2, 3

            nchan = len(chan_cols)
            if len(parts) < i_hg + 1:
                continue                               # truncated final line

            if len(parts) > nchan:                     # starts a new event
                if cur is not None:
                    yield cur
                extra = parts[nchan:]
                try:
                    tstamp = float(extra[0])
                    trgid = int(extra[2])
                except (ValueError, IndexError):
                    tstamp, trgid = float("nan"), -1
                cur = (trgid, tstamp, {}, {})

            if cur is None:
                continue

            try:
                ch = int(parts[1])
                lg = float(parts[i_lg])
                hg = float(parts[i_hg])
            except ValueError:
                continue
            if 0 <= ch < NCH:
                cur[2][ch] = hg
                cur[3][ch] = lg

    if cur is not None:
        yield cur


# ----------------------------------------------------------------------------
def make_image(hg, lg, adc_per_photon, lg_scale, hg_sat):
    """(4, NRING) photon counts, plus how many channels used the LG branch."""
    img = [[0] * NRING for _ in range(len(COLS))]
    nsat = 0
    for ch, (r, c) in CELL.items():
        h = hg.get(ch)
        if h is not None and h > hg_sat:
            nsat += 1
        img[r][c] = photons(h, lg.get(ch), adc_per_photon, lg_scale, hg_sat)
    return img, nsat


def evaluate(img, min_photons, nedge):
    colsum = [sum(img[r][c] for r in range(len(COLS))) for c in range(NRING)]
    total = sum(colsum)
    best = max(colsum)
    imax = colsum.index(best)
    return {"total": total, "colsum": colsum, "imax": imax,
            "ties": colsum.count(best),
            "pass1": total >= min_photons,
            "pass2": nedge <= imax < NRING - nedge}


def show_event(n, trgid, img, res, nsat, min_photons, nedge,
               fired=None, required=TRIG_REQUIRED, nmin=None):
    v1 = "PASS" if res["pass1"] else "fail"
    v2 = "PASS" if res["pass2"] else "fail"
    why = ""
    if not res["pass2"]:
        why = ("  (column %d is %s)"
               % (res["imax"],
                  "one of the first %d" % nedge if res["imax"] < nedge
                  else "one of the last %d" % nedge))
    print("event %-6d TrgID %-7d total = %-7d %s (>= %d)   max column = %-3d "
          "%s%s" % (n, trgid, res["total"], v1, min_photons, res["imax"],
                    v2, why))
    if fired is not None:
        need = len(required) if nmin is None else nmin
        nreq = sum(1 for c in required if c in fired)
        missing = [c for c in required if c not in fired]
        print("              trigger %d/%d of the required %s fired%s   %s"
              % (nreq, len(required),
                 ",".join("ch%d" % c for c in required),
                 "   [also %s]" % ", ".join("ch%d %s" % (c, TRIG_NAME[c])
                                            for c in fired
                                            if c not in required)
                 if any(c not in required for c in fired) else "",
                 "PASS" if nreq >= need
                 else "fail (missing %s)"
                      % ",".join("ch%d %s" % (c, TRIG_NAME[c])
                                 for c in missing)))
    if nsat:
        print("              %d channel(s) above HG %g -> LG * %g used there"
              % (nsat, HG_SAT, LG_SCALE))
    print("   %-9s %s" % ("out col",
                          " ".join("%5d" % c for c in range(NRING))))
    print("   %-9s %s" % ("det ring",
                          " ".join("%5d" % (RING_HI - c)
                                   for c in range(NRING))))
    for r, name in enumerate(COLS):
        print("   %-9s %s" % ("row %d = %s" % (r, name),
                              " ".join("%5d" % v for v in img[r])))
    print("   %-9s %s" % ("col sum", " ".join("%5d" % v
                                              for v in res["colsum"])))
    print("   %-9s %s" % ("", " ".join("%5s" % ("<<<" if c == res["imax"]
                                                else "")
                                       for c in range(NRING))))
    if res["ties"] > 1:
        print("   NOTE  %d columns share the maximum; the lowest index was "
              "used" % res["ties"])
    print("")


# ----------------------------------------------------------------------------
# the average-photon map
# ----------------------------------------------------------------------------
def channel_grid(avg):
    """avg[ch] -> a (16, 4) grid, ring 15 on the top row, columns A B C D.

    Positions with no channel behind them -- A0-A2 and C0-C2, the six corners
    of the 64-cell grid that the 58 SiPMs do not fill -- come back as None and
    are drawn grey.
    """
    grid = [[None] * len(COLS) for _ in range(NRING_ALL)]
    for ch, (col, ring) in CHMAP.items():
        if 0 <= ring < NRING_ALL:
            grid[RING_ALL_HI - ring][COLS.index(col)] = avg[ch]
    return grid


# variable-width bin edges for the distribution pages: (upper edge, step)
HIST_STEPS = [(50, 5), (100, 10), (200, 20), (300, 50), (500, 100)]
HIST_EDGES = [0]
for _hi, _step in HIST_STEPS:
    while HIST_EDGES[-1] < _hi:
        HIST_EDGES.append(HIST_EDGES[-1] + _step)

HIST_COLORS = {"A": "black", "B": "red", "C": "green", "D": "blue"}
RING_PAGES = [(15, 14, 13, 12), (11, 10, 9, 8), (7, 6, 5, 4), (3, 2, 1, 0)]

# (column letter, ring) -> ch, the inverse of CHMAP
CH_AT = {v: k for k, v in CHMAP.items()}


def _merge(counters):
    tot = {}
    for d in counters:
        for k, v in d.items():
            tot[k] = tot.get(k, 0) + v
    return tot


def _stats(d):
    """(entries, mean, max) of a {photons: events} counter."""
    n = sum(d.values())
    if not n:
        return 0, 0.0, 0
    return n, sum(k * v for k, v in d.items()) / float(n), max(d)


def _auto_xmax(hist_ph):
    """Upper edge that holds 99.5 % of all non-zero SiPM entries."""
    tot = _merge(hist_ph[ch] for ch in CHMAP)
    tot.pop(0, None)
    n = sum(tot.values())
    if not n:
        return 50
    acc = 0
    for k in sorted(tot):
        acc += tot[k]
        if acc >= 0.995 * n:
            return max(50, int(k) + 1)
    return max(50, max(tot) + 1)


def _draw_hist(ax, np, d, edges, color, label):
    """One step histogram from a {photons: events} counter; overflow goes
    into the last bin so nothing is silently lost."""
    if not d:
        ax.plot([], [], color=color, lw=1.1, label=label + "  (no entries)")
        return 0, np.zeros(len(edges) - 1)
    vals = np.fromiter(d.keys(), dtype=float, count=len(d))
    wts = np.fromiter(d.values(), dtype=float, count=len(d))
    nover = float(wts[vals >= edges[-1]].sum())
    vals = np.minimum(vals, 0.5 * (edges[-1] + edges[-2]))
    h, _ = np.histogram(vals, bins=edges, weights=wts)
    ax.stairs(h / np.diff(edges), edges, color=color, lw=1.1, label=label)
    return nover, h


RATIO_YMAX = 2.0


def _draw_ratio(ax, np, edges, counts, any_over):
    """Each curve over the average of the curves in the panel.

    counts is [(color, raw bin contents)].  With S the bin's sum over the n
    curves and h one curve's content, ratio = n h / S.  The curve is part of
    its own denominator, so the Poisson error is not the naive quadrature sum
    but  sigma = n sqrt(h (S - h) / S^3).
    Points above the axis range are drawn as arrows at the top edge.
    """
    n = len(counts)
    ax.axhline(1.0, color="0.45", lw=0.8)
    if n:
        S = np.sum([h for _, h in counts], axis=0)
        ok = S > 0
        ctr = 0.5 * (edges[:-1] + edges[1:])
        wid = np.diff(edges)
        for i, (color, h) in enumerate(counts):
            Ssafe = np.where(ok, S, 1.0)
            r = n * h / Ssafe
            e = n * np.sqrt(np.clip(h * (S - h), 0, None) / Ssafe ** 3)
            # small sideways shift so the four sets of bars do not overlap
            x = ctr + (i - 0.5 * (n - 1)) * 0.16 * wid
            m = ok & (r <= RATIO_YMAX)
            ax.errorbar(x[m], r[m], yerr=e[m], fmt="o", ms=2.2, lw=0.8,
                        color=color, capsize=0)
            hi = ok & (r > RATIO_YMAX)
            ax.plot(x[hi], np.full(hi.sum(), RATIO_YMAX * 0.96), "^", ms=3.5,
                    color=color, clip_on=False)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0.0, RATIO_YMAX)
    ax.set_yticks([0.5, 1.0, 1.5])
    ax.set_ylabel("ratio to avg", fontsize=8.5)
    ax.set_xlabel("photons%s" % ("   (last bin = overflow)" if any_over
                                 else ""), fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="major", color="0.88", lw=0.5)
    ax.set_axisbelow(True)


def _panel_pair(fig, cell):
    """A distribution axis with a ratio axis glued underneath, sharing x."""
    sub = cell.subgridspec(2, 1, height_ratios=[3.0, 1.15], hspace=0.0)
    ax = fig.add_subplot(sub[0])
    axr = fig.add_subplot(sub[1], sharex=ax)
    return ax, axr


def _hist_axes(ax, np, edges, logy, any_over):
    ax.set_xlim(edges[0], edges[-1])
    if logy:
        ax.set_yscale("log")
        # half of one event in the widest bin, so a single entry still shows
        ax.set_ylim(bottom=0.5 / float(np.diff(edges).max()))
    else:
        ax.set_ylim(bottom=0)
    ax.set_ylabel("events / photon   (bin content / bin width)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.tick_params(axis="x", labelbottom=False)     # the ratio panel has it
    ax.grid(True, which="major", color="0.88", lw=0.5)
    ax.set_axisbelow(True)


def all_hist_pages(plt, np, variants, nev, scope, src, args):
    """Pages 2-16: every distribution page three times over.

    variants is [(tag, formula, hist)], the combination first.  All five
    pages of one variant come out before the next variant starts: combination
    (pages 2-6), HG only (7-11), LG only (12-16).  All three use the same bin
    edges (taken from the combination when --hist-xmax is automatic).
    """
    xmax_hist = variants[0][2]
    for tag, formula, hist in variants:
        for fig in hist_pages(plt, np, hist, nev, scope, src, args, tag,
                              formula, xmax_hist):
            yield fig


def hist_pages(plt, np, hist_ph, nev, scope, src, args, tag="", formula=None,
               xmax_hist=None):
    """The five distribution pages of ONE variant, one figure at a time."""
    if formula is None:
        formula = ("HG/%g, or (LG*%g)/%g when HG > %g"
                   % (args.adc_per_photon, args.lg_scale,
                      args.adc_per_photon, args.hg_sat))
    ttag = ", %s" % tag if tag else ""
    if args.hist_xmax or args.hist_bins:         # uniform bins on request
        xmax = args.hist_xmax if args.hist_xmax else _auto_xmax(
            xmax_hist if xmax_hist is not None else hist_ph)
        nb = args.hist_bins
        if not nb:                   # integer-width bins, about 100 of them
            width = max(1, int(round(xmax / 100.0)))
            nb = int(-(-xmax // width))
            xmax = nb * width
        edges = np.linspace(0.0, float(xmax), nb + 1)
    else:
        edges = np.array(HIST_EDGES, dtype=float)
    logy = not args.hist_liny
    sub = ("%d events %s   |   %s   |   photons = %s"
           % (nev, scope, os.path.basename(src), formula))

    # ---- four pages, four rings each, A B C D overlaid in each panel ----
    for rings in RING_PAGES:
        fig = plt.figure(figsize=(11.0, 8.5))
        outer = fig.add_gridspec(2, 2, left=0.075, right=0.985, bottom=0.06,
                                 top=0.875, wspace=0.20, hspace=0.30)
        for cell, ring in zip(outer, rings):
            ax, axr = _panel_pair(fig, cell)
            over = 0
            counts = []
            for col in COLS:
                ch = CH_AT.get((col, ring))
                if ch is None:
                    ax.plot([], [], color=HIST_COLORS[col], lw=1.1,
                            label="%s%d  n/c" % (col, ring))
                    continue
                n, mean, mx = _stats(hist_ph[ch])
                nover, h = _draw_hist(
                    ax, np, hist_ph[ch], edges, HIST_COLORS[col],
                    "%s%d  ch%d  mean %.1f  max %d" % (col, ring, ch, mean, mx))
                over += nover
                counts.append((HIST_COLORS[col], h))
            _hist_axes(ax, np, edges, logy, over)
            _draw_ratio(axr, np, edges, counts, over)
            ax.set_title("ring %d%s" % (ring, "" if ring >= RING_LO
                                        else "   (not in the 4 x %d output)"
                                        % NRING), fontsize=10.5)
            ax.legend(fontsize=7.5, loc="upper right", frameon=False)
        fig.suptitle("SiPM photon distributions%s  -  rings %d-%d"
                     % (ttag, rings[0], rings[-1]), fontsize=13, y=0.975)
        fig.text(0.5, 0.935, sub, ha="center", va="top", fontsize=8,
                 color="0.25")
        yield fig

    # ---- rings 15-5 summed, one curve per column ----
    fig = plt.figure(figsize=(11.0, 8.5))
    outer = fig.add_gridspec(1, 1, left=0.075, right=0.985, bottom=0.06,
                             top=0.875)
    ax, axr = _panel_pair(fig, outer[0])
    over = 0
    counts = []
    for col in COLS:
        chans = [CH_AT[(col, r)] for r in range(RING_HI, RING_LO - 1, -1)
                 if (col, r) in CH_AT]
        tot = _merge(hist_ph[ch] for ch in chans)
        n, mean, mx = _stats(tot)
        nover, h = _draw_hist(ax, np, tot, edges, HIST_COLORS[col],
                              "%s%d-%s%d  (%d SiPMs, %d entries)  mean %.1f  "
                              "max %d" % (col, RING_HI, col, RING_LO,
                                          len(chans), n, mean, mx))
        over += nover
        counts.append((HIST_COLORS[col], h))
    _hist_axes(ax, np, edges, logy, over)
    _draw_ratio(axr, np, edges, counts, over)
    ax.legend(fontsize=9.5, loc="upper right", frameon=False)
    fig.suptitle("SiPM photon distributions%s  -  rings %d-%d summed, by "
                 "column" % (ttag, RING_HI, RING_LO), fontsize=13, y=0.975)
    fig.text(0.5, 0.935, sub + "\none entry per SiPM per event", ha="center",
             va="top", fontsize=8, color="0.25", linespacing=1.5)
    yield fig


def write_photon_map(out_stem, sum_ph, nev, scope, src, args, hists=None):
    """Average photons per channel as a 16 x 4 image, triggers on the left.

    Written on every run.  The PNG is this map alone; the PDF has it as page
    1, followed by the photon-count distributions (see all_hist_pages) when
    hists, a list of (tag, formula, hist) variants, is given.  matplotlib is
    imported here rather
    than at the top so that a machine without it can still do the conversion:
    the map is a diagnostic, not the product.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.backends.backend_pdf import PdfPages
        import numpy as np
    except ImportError as exc:
        print("  no photon map: %s (matplotlib/numpy not available)" % exc)
        return []

    if not nev:
        print("  no photon map: no events were %s" % scope)
        return []

    avg = [v / float(nev) for v in sum_ph]
    grid = np.array([[np.nan if v is None else v for v in row]
                     for row in channel_grid(avg)], dtype=float)
    trig = np.array([[avg[c]] for c in TRIG_CH], dtype=float)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d0d0d0")                 # the unassigned cells
    vmax = float(np.nanmax(grid)) if np.isfinite(grid).any() else 1.0
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-9))
    # the paddles collect far more light than a SiPM, so putting them on the
    # SiPM scale would saturate them and flatten the grid.  They get their own
    # normalisation and every cell carries its number.
    tnorm = Normalize(vmin=0.0, vmax=max(float(trig.max()), 1e-9))

    fig = plt.figure(figsize=(8.6, 11.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 4.4, 0.2], wspace=0.30,
                          left=0.115, right=0.895, top=0.842, bottom=0.05)
    ax_t = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])
    ax_cb = fig.add_subplot(gs[0, 2])

    def cell_text(v, n):
        lum = 0.299 * n[0] + 0.587 * n[1] + 0.114 * n[2]
        return "black" if lum > 0.55 else "white"

    # ---- the 6 trigger channels ----
    ax_t.imshow(trig, cmap=cmap, norm=tnorm, aspect="auto",
                extent=[0, 1, len(TRIG_CH), 0], interpolation="nearest")
    for i, c in enumerate(TRIG_CH):
        ax_t.text(0.5, i + 0.5, "%.0f" % avg[c], ha="center", va="center",
                  fontsize=8.5, color=cell_text(avg[c], cmap(tnorm(avg[c]))))
    ax_t.set_xticks([])
    ax_t.set_yticks([i + 0.5 for i in range(len(TRIG_CH))])
    ax_t.set_yticklabels(["ch%d  %s" % (c, TRIG_NAME[c]) for c in TRIG_CH],
                         fontsize=8.5)
    ax_t.set_yticks(range(len(TRIG_CH) + 1), minor=True)
    ax_t.grid(which="minor", color="white", lw=1.0)
    ax_t.tick_params(which="both", length=0)
    ax_t.set_title("trigger\n(own scale)", fontsize=9.5)

    # ---- the 16 x 4 SiPM grid ----
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm,
                   aspect="auto", extent=[0, len(COLS), 0, NRING_ALL],
                   interpolation="nearest")
    for r in range(NRING_ALL):
        ring = RING_ALL_HI - r
        for c in range(len(COLS)):
            v = grid[r, c]
            y = NRING_ALL - r - 0.5
            if not np.isfinite(v):
                ax.text(c + 0.5, y, "n/c", ha="center", va="center",
                        fontsize=7.5, color="#707070")
                continue
            ax.text(c + 0.5, y, "%.1f" % v, ha="center", va="center",
                    fontsize=8, color=cell_text(v, cmap(norm(v))))
    ax.set_xticks([c + 0.5 for c in range(len(COLS))])
    ax.set_xticklabels(COLS, fontsize=11)
    ax.xaxis.tick_top()
    ax.set_yticks([NRING_ALL - r - 0.5 for r in range(NRING_ALL)])
    ax.set_yticklabels([str(RING_ALL_HI - r) for r in range(NRING_ALL)],
                       fontsize=8)
    ax.set_ylabel("detector ring", fontsize=10)
    ax.set_xticks(range(len(COLS) + 1), minor=True)
    ax.set_yticks(range(NRING_ALL + 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=2)

    # the rings that actually reach the 4 x 11 output.  Just the line: any
    # label here lands on a row of numbers, so it goes in the caption instead.
    ax.axhline(RING_LO, color="#cc3311", lw=2.0)

    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("average photons / event", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle("%s   -   average photons per channel"
                 % os.path.basename(src), fontsize=13, y=0.972)
    fig.text(0.5, 0.928,
             "%d events %s   |   photons = HG/%g, or (LG*%g)/%g when HG > %g"
             "\ngrey = no channel assigned (A0-A2, C0-C2);  "
             "Sept 14 channel assignment"
             "\nred line: rings %d-%d go to the 4 x %d output, rings %d-%d "
             "are dropped"
             % (nev, scope, args.adc_per_photon, args.lg_scale,
                args.adc_per_photon, args.hg_sat,
                RING_LO, RING_HI, NRING, 0, RING_LO - 1),
             ha="center", va="top", fontsize=8.5, color="0.25",
             linespacing=1.5)

    paths = [out_stem + ".png", out_stem + ".pdf"]
    fig.savefig(paths[0], dpi=150)
    with PdfPages(paths[1]) as pdf:
        pdf.savefig(fig)
        plt.close(fig)
        if hists:
            for hfig in all_hist_pages(plt, np, hists, nev, scope, src,
                                       args):
                pdf.savefig(hfig)
                plt.close(hfig)
    return paths


# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("listfile", help="Janus list text file")
    p.add_argument("-o", "--out", default=None,
                   help="output file (default: "
                        "INPUT_formatted_<min>photons_innerRing3.txt)")
    p.add_argument("--adc-per-photon", type=float, default=ADC_PER_PHOTON,
                   help="HG ADC counts per photon (default %g)"
                        % ADC_PER_PHOTON)
    p.add_argument("--lg-scale", type=float, default=LG_SCALE,
                   help="constant LG -> HG scale factor (default %g)"
                        % LG_SCALE)
    p.add_argument("--hg-sat", type=float, default=HG_SAT,
                   help="HG strictly above this uses (LG * scale) instead "
                        "(default %g)" % HG_SAT)
    p.add_argument("--min-photons", type=int, default=MIN_PHOTONS,
                   help="selection 1: least total photons over the 4 x %d "
                        "array (default %d)" % (NRING, MIN_PHOTONS))
    p.add_argument("--edge", type=int, default=NEDGE, metavar="N",
                   help="selection 2: bar the max-photon column from the "
                        "first N and last N columns (default %d)" % NEDGE)
    p.add_argument("--requireTrigger", action="store_true",
                   help="also require ALL of %s to have fired, where fired "
                        "means raw HG > %g OR raw LG > %g -- the same "
                        "criterion as makeEventDisplays.py.  "
                        "'_requireTrigger' is added to the default output "
                        "file name"
                        % (",".join("ch%d" % c for c in TRIG_REQUIRED),
                           TRIG_HG_MIN, TRIG_LG_MIN))
    p.add_argument("--trig-hg", type=float, default=TRIG_HG_MIN,
                   help="raw HG above which a trigger channel counts as "
                        "fired (default %g)" % TRIG_HG_MIN)
    p.add_argument("--trig-lg", type=float, default=TRIG_LG_MIN,
                   help="raw LG above which a trigger channel counts as "
                        "fired (default %g)" % TRIG_LG_MIN)
    p.add_argument("--trig-channels", default=None, metavar="LIST",
                   help="comma-separated channels that must fire (default "
                        "%s).  The channel assignment has moved once already "
                        "in this detector, so it is worth being explicit."
                        % ",".join(str(c) for c in TRIG_REQUIRED))
    p.add_argument("--trig-nmin", type=int, default=None, metavar="N",
                   help="require only N of the channels above instead of all "
                        "of them (default: all %d)" % len(TRIG_REQUIRED))
    p.add_argument("--placeholder", action="store_true",
                   help="also write a 4 x %d block of -999999 before the "
                        "counts, in the slot the MC files use for the "
                        "arrival-time array.  Makes the output readable by "
                        "the CNN loader as it stands (skip_leading_blocks=1)"
                        % NRING)
    p.add_argument("--map-out", default=None, metavar="STEM",
                   help="output stem for the average-photon map (default: the "
                        "output text file's name with _avgPhotons)")
    p.add_argument("--no-map", action="store_true",
                   help="do not write the average-photon map or the "
                        "distribution pages.  By default a "
                        "16 x 4 image of the average photons per channel, "
                        "with the six trigger channels on the left, is written "
                        "as PNG and as page 1 of a 16-page PDF that also "
                        "holds the per-SiPM photon distributions (HG/LG "
                        "combination, HG only and LG only)")
    p.add_argument("--map-selected", action="store_true",
                   help="average the map over the events WRITTEN to the "
                        "output instead of over every event read, i.e. after "
                        "the trigger and photon cuts")
    p.add_argument("--hist-xmax", type=float, default=None, metavar="N",
                   help="upper edge, in photons, of the distribution pages "
                        "with UNIFORM bins, instead of the default variable "
                        "bins (5 to 50, 10 to 100, 20 to 200, 50 to 300, 100 "
                        "to 500).  Anything above goes into the last bin")
    p.add_argument("--hist-bins", type=int, default=None, metavar="N",
                   help="use this many UNIFORM bins on the distribution "
                        "pages instead of the default variable bins")
    p.add_argument("--hist-liny", action="store_true",
                   help="linear y axis on the distribution pages (default "
                        "log)")
    p.add_argument("--no-select", action="store_true",
                   help="convert every event, applying neither selection")
    p.add_argument("--max-events", type=int, default=None,
                   help="stop after this many input events")
    p.add_argument("--nprint", type=int, default=NPRINT,
                   help="print this many events in full (default %d)"
                        % NPRINT)
    p.add_argument("--report-every", type=int, default=REPORT_EVERY,
                   help="progress line every this many events (default %d, "
                        "0 for silence)" % REPORT_EVERY)
    args = p.parse_args(argv)

    src = args.listfile
    if not os.path.isfile(src):
        sys.exit("no such file: %s" % src)

    stem = src[:-4] if src.lower().endswith(".txt") else src
    out_path = args.out or ("%s_formatted_%dphotons_innerRing3%s.txt"
                            % (stem, args.min_photons,
                               "_requireTrigger" if args.requireTrigger
                               else ""))
    if os.path.realpath(out_path) == os.path.realpath(src):
        sys.exit("the output would overwrite the input; use -o")

    print("=" * 78)
    print("input      %s   (%.2f MB)" % (src, os.path.getsize(src) / 1e6))
    print("output     %s" % out_path)
    print("photons    HG / %g,  or (LG * %g) / %g when HG > %g"
          % (args.adc_per_photon, args.lg_scale, args.adc_per_photon,
             args.hg_sat))
    print("geometry   ch0-5 = CW paddles, dropped;  ch6-63 = SiPMs")
    print("           rings %d-%d dropped, rings %d-%d kept -> %d columns"
          % (0, RING_LO - 1, RING_LO, RING_HI, NRING))
    print("           output column 0 = ring %d ... column %d = ring %d"
          % (RING_HI, NRING - 1, RING_LO))
    print("           output rows = detector columns %s"
          % ", ".join(COLS))
    print("format     TrgID line%s, then 4 x %d counts"
          % (", 4 x %d placeholder block" % NRING if args.placeholder else "",
             NRING))
    # which channels gate the event, and how many of them are needed
    if args.trig_channels:
        try:
            required = tuple(int(v) for v in args.trig_channels.replace(",", " ").split())
        except ValueError:
            sys.exit("--trig-channels %r is not a list of integers"
                     % args.trig_channels)
        if not required or any(not 0 <= c < NCH for c in required):
            sys.exit("--trig-channels must name channels in 0..%d" % (NCH - 1))
    else:
        required = TRIG_REQUIRED
    need = len(required) if args.trig_nmin is None else args.trig_nmin
    if not 1 <= need <= len(required):
        sys.exit("--trig-nmin %d makes no sense for %d channels"
                 % (need, len(required)))

    if args.requireTrigger:
        print("trigger    --requireTrigger ON: %s of %s fired, where "
              "fired = raw HG > %g OR raw LG > %g"
              % ("ALL %d" % need if need == len(required) else ">= %d" % need,
                 ",".join("ch%d (%s)" % (c, TRIG_NAME.get(c, "?"))
                          for c in required),
                 args.trig_hg, args.trig_lg))
    if args.no_select:
        print("selection  NONE (--no-select)%s"
              % ("   -- the trigger cut still applies"
                 if args.requireTrigger else ""))
    else:
        print("selection  1: total photons >= %d" % args.min_photons)
        print("           2: max column not in 0..%d or %d..%d"
              % (args.edge - 1, NRING - args.edge, NRING - 1))
    print("=" * 78)

    # running sum of photons per channel, for the map.  All 64 channels, not
    # just the 44 that reach the output, and streamed like everything else.
    # hist_ph[ch] is {photons: events}: the full distribution, exactly, at a
    # cost of one small dict per channel whatever the file size.
    sum_ph = [0.0] * NCH
    hist_ph = [dict() for _ in range(NCH)]
    # the same, from each gain on its own (no HG/LG switch), in photons
    hist_hg = [dict() for _ in range(NCH)]
    hist_lg = [dict() for _ in range(NCH)]
    n_map = 0

    def accumulate(hg, lg):
        for c in range(NCH):
            h, l = hg.get(c), lg.get(c)
            v = photons(h, l, args.adc_per_photon, args.lg_scale,
                        args.hg_sat)
            sum_ph[c] += v
            d = hist_ph[c]
            d[v] = d.get(v, 0) + 1
            v = photons_single(h, 1.0, args.adc_per_photon)
            d = hist_hg[c]
            d[v] = d.get(v, 0) + 1
            v = photons_single(l, args.lg_scale, args.adc_per_photon)
            d = hist_lg[c]
            d[v] = d.get(v, 0) + 1

    n = nt = n1 = n12 = nsat_tot = 0
    nfired_hist = [0] * (len(required) + 1)     # among the REQUIRED channels
    # per-channel fire counts over all six, so a dead paddle is obvious: an
    # AND of four goes to zero if any one of them stops firing
    chan_fired = {c: 0 for c in TRIG_CH}
    t0 = time.time()
    ph_line = " ".join(["-999999"] * NRING)

    out = open(out_path, "w")
    try:
        for trgid, _tstamp, hg, lg in iter_events(src):
            if args.max_events is not None and n >= args.max_events:
                break
            n += 1

            fired = fired_trigger_channels(hg, lg, args.trig_hg,
                                           args.trig_lg)
            for c in fired:
                if c in chan_fired:
                    chan_fired[c] += 1
            nreq = sum(1 for c in required if c in fired)
            nfired_hist[nreq] += 1
            trig_ok = (not args.requireTrigger) or nreq >= need

            if not args.no_map and not args.map_selected:
                n_map += 1
                accumulate(hg, lg)

            img, nsat = make_image(hg, lg, args.adc_per_photon,
                                   args.lg_scale, args.hg_sat)
            nsat_tot += nsat
            res = evaluate(img, args.min_photons, args.edge)

            if n <= args.nprint:
                show_event(n - 1, trgid, img, res, nsat, args.min_photons,
                           args.edge,
                           fired if args.requireTrigger else None,
                           required, need)

            # the trigger cut comes first: it is a property of the event, not
            # of the image, so the photon fractions below are quoted among
            # the events that fired
            if not trig_ok:
                continue
            nt += 1

            keep = args.no_select or (res["pass1"] and res["pass2"])
            if res["pass1"]:
                n1 += 1
                if res["pass2"]:
                    n12 += 1

            if keep and not args.no_map and args.map_selected:
                n_map += 1
                accumulate(hg, lg)

            if keep:
                out.write("%d\n" % trgid)
                if args.placeholder:
                    for _ in range(len(COLS)):
                        out.write(ph_line + "\n")
                for r in range(len(COLS)):
                    out.write(" ".join(str(v) for v in img[r]) + "\n")
                out.write("\n")

            if args.report_every and n % args.report_every == 0:
                dt = time.time() - t0
                sys.stderr.write("   ... %d events  %.0f/s  pass1 %d  "
                                 "pass1+2 %d\n"
                                 % (n, n / dt if dt else 0, n1, n12))
                sys.stderr.flush()
    finally:
        out.close()

    dt = time.time() - t0
    nwritten = nt if args.no_select else n12
    base = nt if args.requireTrigger else n

    def frac(a, b):
        return "%.4f" % (a / b) if b else "n/a"

    print("=" * 78)
    print("  events read                         %8d" % n)
    if args.requireTrigger:
        print("  pass --requireTrigger (%s of %d)%s %8d    fraction of all "
              "= %s"
              % ("all" if need == len(required) else ">= %d" % need,
                 len(required), " " * 5, nt, frac(nt, n)))
    print("  pass 1 (>= %d photons)            %8d    fraction of %-9s "
          "= %s" % (args.min_photons, n1,
                    "triggered" if args.requireTrigger else "all",
                    frac(n1, base)))
    print("  pass 1 and 2 (max column not edge)  %8d    fraction of pass 1 "
          "= %s" % (n12, frac(n12, n1)))
    print("  %-34s  %8d    fraction of all   = %s"
          % ("written to the output", nwritten, frac(nwritten, n)))
    print("-" * 78)
    print("  of the required %s, this many fired per event: %s"
          % (",".join("ch%d" % c for c in required),
             "  ".join("%d->%d" % (i, v)
                       for i, v in enumerate(nfired_hist) if v)))
    print("  per-channel fire rate (a dead paddle zeroes an AND):")
    for c in TRIG_CH:
        print("     ch%-2d %-10s %8d  %6.2f %%%s"
              % (c, TRIG_NAME.get(c, ""), chan_fired[c],
                 100.0 * chan_fired[c] / max(n, 1),
                 "   <- required" if c in required else ""))
    dead = [c for c in required if chan_fired[c] == 0]
    if dead:
        print("     WARNING  %s never fired, so an AND over the required set "
              "can never pass."
              % ", ".join("ch%d" % c for c in dead))
    print("  %d channel(s) over HG %g in total took the LG * %g branch"
          % (nsat_tot, args.hg_sat, args.lg_scale))
    print("  %.1f s" % dt)
    if n1 == 0 and n:
        print("  NOTE  nothing reached %d photons.  The largest total is "
              "worth a look with" % args.min_photons)
        print("        --no-select --nprint 20 before changing the "
              "threshold.")
    print("=" * 78)
    print("wrote %s" % out_path)

    if not args.no_map:
        map_stem = args.map_out or (
            (out_path[:-4] if out_path.lower().endswith(".txt") else out_path)
            + "_avgPhotons")
        scope = "written to the output" if args.map_selected else "read"
        variants = [
            ("HG/LG combination",
             "HG/%g, or (LG*%g)/%g when HG > %g"
             % (args.adc_per_photon, args.lg_scale, args.adc_per_photon,
                args.hg_sat), hist_ph),
            ("HG only",
             "HG/%g for every entry (no LG switch; saturates near %g/%g)"
             % (args.adc_per_photon, args.hg_sat, args.adc_per_photon),
             hist_hg),
            ("LG only",
             "(LG*%g)/%g for every entry (no HG)"
             % (args.lg_scale, args.adc_per_photon), hist_lg),
        ]
        for pth in write_photon_map(map_stem, sum_ph, n_map, scope, src, args,
                                    variants):
            print("wrote %s" % pth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
