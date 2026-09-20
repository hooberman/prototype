#!/usr/bin/env python3
"""
compare.py DATAFILE MCFILE

Data vs MC comparison of three whole-event quantities on the 4 x 11 SiPM grid:

    total photons            sum over all 44 cells
    N SiPMs >= 50 photons    occupancy at a low threshold
    N SiPMs >= 100 photons   occupancy at a high threshold

    python compare.py Run150_..._requireTrigger.txt sipm_hits_PoC_..._innerRing3.txt

        -> Run150_..._vs_sipm_hits_PoC_..._compare.pdf  (+ .png)

File formats
------------
Both layouts are DETECTED from the file, not assumed.  An event is a header
line followed by one or more 4 x NRING blocks; the PHOTONS are always the LAST
block.  A header is any line whose token count is not NRING.

    data :  TrgID                                   1 token  -> 1 + 4 lines
            4 x 11 photon counts

    MC   :  theta phi z ...                         8 tokens -> 1 + 8 lines
            4 x 11 placeholder / truth block        (skipped)
            4 x 11 photon counts

Blank lines are ignored.  The MC truth header is parsed when it has >= 2
numeric fields, so theta/phi are available for reference, but nothing here
depends on it.

Streaming
---------
The MC file is routinely millions of events and gigabytes.  Nothing is held in
memory except the three scalars per event, so the whole file can be read
without trouble.  --max-events caps either side if you want a quick look.

Normalisation
-------------
The two samples have different sizes and different selections, so the top row
is normalised to unit area: what is being compared is the SHAPE.  The legend
carries the raw event counts.  The bottom row is the data/MC ratio of those
normalised shapes, with Poisson errors from the raw counts, so a flat line at
1 means the shapes agree.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NROW = 4                 # detector columns A, B, C, D
NRING = 11               # rings 15 .. 5

DATA_COLOR = "#1a1a1a"
MC_COLOR = "#2a78d6"
MC_FILL = "#2a78d6"


# ---------------------------------------------------------------- the reader
def sniff(path, nrow=NROW, nring=NRING, look=400):
    """Work out lines-per-event and how many blocks precede the photons.

    Returns (per_event, n_skip_rows, n_header_tokens).
    """
    lines = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                lines.append(ln)
            if len(lines) >= look:
                break
    if not lines:
        sys.exit("%s has no data lines" % path)

    hdr = [i for i, ln in enumerate(lines) if len(ln.split()) != nring]
    if not hdr or hdr[0] != 0:
        sys.exit("%s does not start with a header line (a line whose token "
                 "count is not %d).  Is this the right format?" % (path, nring))
    if len(hdr) < 2:
        sys.exit("%s: could not find a second event within the first %d lines"
                 % (path, look))

    per_event = hdr[1] - hdr[0]
    nblocks = (per_event - 1) // nrow
    if per_event != 1 + nblocks * nrow or nblocks < 1:
        sys.exit("%s has %d lines per event; expected 1 + k*%d"
                 % (path, per_event, nrow))
    return per_event, (nblocks - 1) * nrow, len(lines[0].split())


def stream_events(path, thresholds, max_events=0, nrow=NROW, nring=NRING):
    """Yield nothing -- return the per-event summary arrays.

    Only the scalars are kept, so a multi-GB file costs a few tens of MB.
    Returns (total, counts_above (n_thresh, N), header0 (N, k) or None, nread).
    """
    per_event, n_skip, n_hdr_tok = sniff(path, nrow, nring)

    totals = []
    above = [[] for _ in thresholds]
    heads = []
    keep_head = n_hdr_tok >= 2

    buf = []
    n = 0
    bad = 0
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            buf.append(ln)
            if len(buf) < per_event:
                continue

            rows = buf[1 + n_skip:1 + n_skip + nrow]
            try:
                img = np.array([[float(x) for x in r.split()] for r in rows],
                               dtype=np.float64)
                if img.shape != (nrow, nring):
                    raise ValueError
            except ValueError:
                bad += 1
                buf = []
                continue

            totals.append(img.sum())
            for k, t in enumerate(thresholds):
                above[k].append(int(np.count_nonzero(img >= t)))
            if keep_head:
                try:
                    heads.append([float(x) for x in buf[0].split()[:2]])
                except ValueError:
                    heads.append([np.nan, np.nan])

            buf = []
            n += 1
            if max_events and n >= max_events:
                break

    if not totals:
        sys.exit("%s: no events parsed" % path)
    if bad:
        print("  warning: %d malformed blocks skipped in %s"
              % (bad, os.path.basename(path)))

    return (np.array(totals),
            np.array(above, dtype=int),
            np.array(heads) if keep_head and heads else None,
            n)


# ------------------------------------------------------------------- drawing
def panel(ax, axr, d, m, edges, xlabel, title, logy=False):
    """One column: normalised overlay on ax, data/MC ratio on axr."""
    ctr = 0.5 * (edges[:-1] + edges[1:])
    w = np.diff(edges)

    nd, _ = np.histogram(d, bins=edges)
    nm, _ = np.histogram(m, bins=edges)
    Nd, Nm = max(nd.sum(), 1), max(nm.sum(), 1)

    # densities, so the comparison is of shape not sample size
    hd, ed_ = nd / Nd / w, np.sqrt(nd) / Nd / w
    hm, em_ = nm / Nm / w, np.sqrt(nm) / Nm / w

    ax.stairs(hm, edges, fill=True, color=MC_FILL, alpha=0.25, zorder=1)
    ax.stairs(hm, edges, color=MC_COLOR, lw=1.8, zorder=2,
              label="MC  (%d events)" % Nm)
    ax.errorbar(ctr, hd, yerr=ed_, fmt="o", ms=3.2, color=DATA_COLOR,
                capsize=0, lw=1.0, zorder=3, label="data  (%d events)" % Nd)

    ax.set_ylabel("fraction of events / bin", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlim(edges[0], edges[-1])
    if logy:
        ax.set_yscale("log")
        pos = np.r_[hd[hd > 0], hm[hm > 0]]
        if pos.size:
            ax.set_ylim(0.5 * pos.min(), 3.0 * pos.max())
    ax.legend(fontsize=8.5, loc="best")

    # ---- ratio ----
    ok = (hm > 0) & (nd > 0)
    r = np.full_like(hd, np.nan)
    re = np.full_like(hd, np.nan)
    r[ok] = hd[ok] / hm[ok]
    re[ok] = r[ok] * np.sqrt((ed_[ok] / hd[ok]) ** 2 + (em_[ok] / hm[ok]) ** 2)

    axr.axhline(1.0, color=MC_COLOR, lw=1.2, ls="--")
    axr.errorbar(ctr, r, yerr=re, fmt="o", ms=3.2, color=DATA_COLOR,
                 capsize=0, lw=1.0)
    axr.set_xlabel(xlabel, fontsize=11)
    axr.set_ylabel("data / MC", fontsize=10)
    axr.grid(alpha=0.25, lw=0.5)
    axr.set_xlim(edges[0], edges[-1])
    fin = r[np.isfinite(r)]
    hi = min(4.0, max(2.0, np.nanpercentile(fin, 95) * 1.3)) if fin.size else 2.0
    axr.set_ylim(0, hi)
    return nd, nm


def stats_line(name, v):
    return ("  %-22s N %8d   mean %9.1f   median %9.1f   "
            "[5%%, 95%%] = [%.1f, %.1f]   max %.1f"
            % (name, v.size, v.mean(), np.median(v),
               np.percentile(v, 5), np.percentile(v, 95), v.max()))


# ---------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datafile")
    p.add_argument("mcfile")
    p.add_argument("-o", "--out", default=None,
                   help="output stem (default: built from the two filenames)")
    p.add_argument("--thresholds", type=float, nargs="+", default=[50.0, 100.0],
                   metavar="PHOT",
                   help="photon thresholds for the occupancy panels "
                        "(default 50 100)")
    p.add_argument("--bins", type=int, default=50,
                   help="bins in the total-photons histogram (default 50)")
    p.add_argument("--total-max", type=float, default=None, metavar="PHOT",
                   help="upper edge of the total-photons axis (default: the "
                        "99th percentile of the two samples combined, so a "
                        "few huge events do not flatten the plot; the number "
                        "overflowing is reported)")
    p.add_argument("--max-events", type=int, default=0, metavar="N",
                   help="read at most N events from EACH file (0 = all).  The "
                        "MC file is often millions of events; the reader is "
                        "streaming so this is for speed, not memory")
    p.add_argument("--max-mc", type=int, default=0, metavar="N",
                   help="separate cap for the MC file only (overrides "
                        "--max-events for MC)")
    p.add_argument("--logy", action="store_true",
                   help="log scale on the top row")
    p.add_argument("--no-pdf", action="store_true")
    a = p.parse_args(argv)

    for f in (a.datafile, a.mcfile):
        if not os.path.isfile(f):
            sys.exit("no such file: %s" % f)

    thr = list(a.thresholds)

    print("=" * 78)
    for tag, path, cap in (("data", a.datafile, a.max_events),
                           ("MC", a.mcfile, a.max_mc or a.max_events)):
        pe, nsk, nht = sniff(path)
        print("%-5s %s" % (tag, path))
        print("      %d lines/event, header has %d token%s, %d block%s skipped "
              "before the photons"
              % (pe, nht, "" if nht == 1 else "s",
                 nsk // NROW, "" if nsk // NROW == 1 else "s"))
    print("-" * 78)

    d_tot, d_above, d_head, d_n = stream_events(a.datafile, thr, a.max_events)
    m_tot, m_above, m_head, m_n = stream_events(a.mcfile, thr,
                                                a.max_mc or a.max_events)
    print("read  %d data events, %d MC events" % (d_n, m_n))
    if m_head is not None and np.isfinite(m_head).all():
        print("      MC truth header: theta mean %.2f deg, phi range "
              "[%.1f, %.1f] deg"
              % (m_head[:, 0].mean(), m_head[:, 1].min(), m_head[:, 1].max()))
    print("-" * 78)

    print(stats_line("total photons  data", d_tot))
    print(stats_line("total photons  MC", m_tot))
    for k, t in enumerate(thr):
        print(stats_line("N >= %-3g phot  data" % t, d_above[k].astype(float)))
        print(stats_line("N >= %-3g phot  MC" % t, m_above[k].astype(float)))
    print("-" * 78)

    # ---- binning ----
    tmax = a.total_max if a.total_max else float(
        np.percentile(np.r_[d_tot, m_tot], 99.0))
    tmax = max(tmax, 1.0)
    n_over_d = int(np.sum(d_tot > tmax))
    n_over_m = int(np.sum(m_tot > tmax))
    if n_over_d or n_over_m:
        print("total-photons axis capped at %.0f: %d data (%.2f%%) and %d MC "
              "(%.2f%%) events above it are not plotted"
              % (tmax, n_over_d, 100.0 * n_over_d / d_n,
                 n_over_m, 100.0 * n_over_m / m_n))

    nmax = int(max(d_above.max(), m_above.max()))
    nmax = min(NROW * NRING, nmax + 1)

    # ---- figure ----
    ncol = 1 + len(thr)
    fig, axes = plt.subplots(2, ncol, figsize=(5.2 * ncol, 7.6),
                             gridspec_kw={"height_ratios": [2.4, 1]},
                             squeeze=False)

    panel(axes[0][0], axes[1][0], d_tot, m_tot,
          np.linspace(0, tmax, a.bins + 1),
          "total photons in the 4 x %d grid" % NRING,
          "total photons", logy=a.logy)

    for k, t in enumerate(thr):
        panel(axes[0][k + 1], axes[1][k + 1],
              d_above[k].astype(float), m_above[k].astype(float),
              np.arange(-0.5, nmax + 1.5, 1.0),
              "N SiPMs $\\geq$ %g photons" % t,
              "SiPMs above %g photons" % t, logy=a.logy)

    fig.suptitle("data  vs  MC\n%s     vs     %s"
                 % (os.path.basename(a.datafile), os.path.basename(a.mcfile)),
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955])

    stem = a.out or (os.path.splitext(os.path.basename(a.datafile))[0]
                     + "_vs_"
                     + os.path.splitext(os.path.basename(a.mcfile))[0][:40]
                     + "_compare")
    paths = [stem + ".png"] + ([] if a.no_pdf else [stem + ".pdf"])
    for q in paths:
        fig.savefig(q, dpi=150, bbox_inches="tight")
        print("wrote %s" % q)
    plt.close(fig)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
