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
output of calculatePedestalsAndNoise.py.  --ped overrides it.

Outputs
-------
    RUNFILE_LGscale.txt        one row per channel with the scale factor
    RUNFILE_LGscale.png        8x8 grid of LGcorr vs HGcorr with the fit
    RUNFILE_LGscale_map.png    the scale factor drawn on the detector map

This script imports the parser and the detector map from
calculatePedestalsAndNoise.py, so keep the two files in the same directory.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from calculatePedestalsAndNoise import (parse_list_file, run_label,
                                            draw_panel, CHMAP, NCH,
                                            UNCONNECTED)
except ImportError:
    sys.exit("calculateLGScale.py needs calculatePedestalsAndNoise.py "
             "in the same directory")

# ----------------------------------------------------------------------------
PedestalNoiseFile = "Data/Run124_list_pedestalsAndNoise.txt"
# ----------------------------------------------------------------------------

HG_SAT = 4000.0     # raw HG at or above this is saturated and is dropped
NSIGMA = 3.0        # keep events with HGcorr above this many HG pedestal RMS
CLIP = 4.0          # residual clipping, in sigma, during the fit
MINPTS = 20         # fewest surviving events for a fit to be attempted


# ----------------------------------------------------------------------------
# pedestal file
# ----------------------------------------------------------------------------
def read_pedestals(path):
    """Read calculatePedestalsAndNoise.py output -> dict of 64-length arrays."""
    hg_ped = np.full(NCH, np.nan)
    hg_rms = np.full(NCH, np.nan)
    lg_ped = np.full(NCH, np.nan)
    lg_rms = np.full(NCH, np.nan)
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 7:
                continue
            # read from the right, so a channel name containing a space
            # (an older "CW top") still parses
            try:
                c = int(p[0])
                if not 0 <= c < NCH:
                    continue
                hg_ped[c], hg_rms[c] = float(p[-5]), float(p[-4])
                lg_ped[c], lg_rms[c] = float(p[-3]), float(p[-2])
            except (ValueError, IndexError):
                continue
    if not np.isfinite(hg_ped).any():
        sys.exit("no pedestal rows parsed from %s" % path)
    return {"hg_ped": hg_ped, "hg_rms": hg_rms,
            "lg_ped": lg_ped, "lg_rms": lg_rms}


# ----------------------------------------------------------------------------
# the fit
# ----------------------------------------------------------------------------
def fit_channel(lgc, hgc, clip=CLIP):
    """Fit HGcorr = k * LGcorr + c.

    Returns a dict with
        k        slope of the ordinary least-squares fit of HG on LG
        kerr     its uncertainty
        c        intercept
        kgm      sigma(HG)/sigma(LG), the geometric-mean (Deming, delta=var
                 ratio) slope.  Immune to the regression dilution that pulls
                 the OLS slope low when LG carries real scatter of its own.
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
        name = CHMAP[c][0]
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

        tag = "ch%d  %s" % (c, name)
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


def scale_map(values, out_png, title, suptitle):
    """The per-channel scale factor drawn on the A/B/C/D detector map."""
    fig = plt.figure(figsize=(6.2, 9.2))
    gs = fig.add_gridspec(1, 1, left=0.10, right=0.90, top=0.90, bottom=0.04)
    # vmin=None: the interesting spread is around 12, not down at 0
    draw_panel(fig, gs[0, 0], values, title, "cividis", "HG per LG", vmin=None)
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
            name = CHMAP[c][0].replace(" ", "")
            if c in UNCONNECTED:
                name += "*"
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

    ped = read_pedestals(args.ped)
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
        sub = "LG to HG scale   -   %s   |   %s" % (label, cuts)
        scatter_grid(per_ch, fits, out_png, sub)
        kvals = np.array([fits[c]["k"] for c in range(NCH)])
        scale_map(kvals, out_map, "HG / LG slope",
                  "LG to HG scale   -   %s" % label)

    # ---- console summary ----
    ok = np.array([c for c in range(NCH)
                   if np.isfinite(fits[c]["k"]) and c not in UNCONNECTED])
    print("parsed %d events from %s" % (n_events, os.path.basename(src)))
    print("pedestals from %s" % args.ped)
    print("cuts: %s" % cuts)
    if ok.size:
        k = np.array([fits[c]["k"] for c in ok])
        kg = np.array([fits[c]["kgm"] for c in ok])
        nn = np.array([fits[c]["n"] for c in ok])
        print("%d channels fitted, median events used %d" % (ok.size,
                                                             int(np.median(nn))))
        print("  HG/LG slope   median %.3f   mean %.3f +- %.3f   "
              "range %.3f - %.3f" % (np.median(k), k.mean(), k.std(),
                                     k.min(), k.max()))
        print("  k_gm          median %.3f   (sigma ratio, dilution-free)"
              % np.median(kg))
        print("  -> LG * %.3f  ~  HG" % np.median(k))
        bad = [c for c in range(NCH)
               if c not in UNCONNECTED and not np.isfinite(fits[c]["k"])]
        if bad:
            print("  no fit for channels: %s" %
                  ", ".join(str(b) for b in bad))
    else:
        print("no channel could be fitted - loosen --nsigma or check the cuts")
    print("wrote %s" % out_txt)
    if not args.no_plot:
        print("wrote %s" % out_png)
        print("wrote %s" % out_map)


if __name__ == "__main__":
    main()
