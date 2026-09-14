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
    RunX_photonPeaks_all50.png                  the 50 active SiPMs together,
                                                with the photon-peak fit

The fit
-------
The all-50 spectrum is fitted with

    y(x) = p0 + p1 x + p2 x^2
           + SUM(i = 1..npeaks)  A_i exp( -0.5 ((x - i N) / sigma_i)^2 )

i.e. a 2nd-order polynomial background plus a comb of Gaussians whose means are
locked to integer multiples of one free gain N -- the single-photoelectron
spacing in HG ADC counts, which is the number this plot exists to measure.
Each peak keeps its own free width, started at 20 and bounded to (5, 50).

N starts at 35 (--n-init) and the fit runs over 20 ADC and up (--fit-xmin),
leaving the pedestal tail out of it; the plot still shows the whole range.
--n-init 0 instead seeds N by scanning it over a grid and solving the
remaining, linear, parameters exactly at each step.  --npeaks changes the
number of peaks, --no-fit switches it off.

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

HG_LO = 2.0        # x axis of every panel
HG_HI = 200.0
NBINS = 99

# ---- the photon-peak fit -----------------------------------------------------
NPEAKS = 4          # Gaussians in the comb: means at N, 2N, 3N, ... npeaks*N
N_INIT = 35.0       # starting value for the gain N
SIGMA_INIT = 20.0   # starting width of every peak
SIGMA_LO = 5.0      # and the window it is allowed to move in
SIGMA_HI = 50.0
NSCAN = 400         # grid points in the seed scan over N

# The fit starts above the pedestal tail.  Below ~20 ADC the spectrum falls far
# too steeply for a 2nd-order polynomial, and the comb gets dragged into it --
# which is how N ends up at half its true value with the odd peaks switched off.
FIT_LO = 20.0

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
# the photon-peak fit
# ----------------------------------------------------------------------------
def peak_model(x, pars, npeaks):
    """p0 + p1 x + p2 x^2 + sum_i A_i Gauss(mean = i*N, sigma = sigma_i)."""
    x = np.asarray(x, dtype=float)
    y = pars[0] + pars[1] * x + pars[2] * x * x
    gain = pars[3]
    for i in range(npeaks):
        amp = pars[4 + i]
        sig = pars[4 + npeaks + i]
        y = y + amp * np.exp(-0.5 * ((x - (i + 1) * gain) / sig) ** 2)
    return y


def _design(x, gain, sigmas, npeaks):
    """Columns of the model that multiply the linear parameters."""
    cols = [np.ones_like(x), x, x * x]
    for i in range(npeaks):
        cols.append(np.exp(-0.5 * ((x - (i + 1) * gain) / sigmas[i]) ** 2))
    return np.vstack(cols).T


def seed_gain(x, y, w, npeaks, sigma_init, sigma_lo, sigma_hi, n_lo, n_hi,
              nscan=NSCAN):
    """Scan N; at each N the rest of the model is linear, so solve it exactly.

    The comb has a local minimum at every sub-multiple of the true spacing, so
    a scan is the only reliable way in.  The width is scanned coarsely too --
    holding it at one value tilts the landscape towards combs of that width.

    Returns (chi2, N, sigma, linear coefficients) sorted best first.
    """
    widths = sorted({min(max(f * sigma_init, sigma_lo), sigma_hi)
                     for f in (0.5, 1.0, 1.5)})
    out = []
    for sg in widths:
        sig = [sg] * npeaks
        for gain in np.linspace(n_lo, n_hi, nscan):
            a = _design(x, gain, sig, npeaks)
            try:
                coef, _, _, _ = np.linalg.lstsq(a * w[:, None], y * w,
                                                rcond=None)
            except np.linalg.LinAlgError:
                continue
            r = (a @ coef - y) * w
            out.append((float(r @ r), float(gain), float(sg), coef))
    out.sort(key=lambda t: t[0])
    return out


def spread_seeds(scan, nseeds, n_lo, n_hi):
    """The best seeds, kept apart in N so they explore different combs."""
    gap = 0.04 * (n_hi - n_lo)
    picked = []
    for item in scan:
        if all(abs(item[1] - q[1]) > gap for q in picked):
            picked.append(item)
        if len(picked) >= nseeds:
            break
    return picked


def fit_photon_peaks(centres, y, yerr, npeaks=NPEAKS, sigma_init=SIGMA_INIT,
                     sigma_lo=SIGMA_LO, sigma_hi=SIGMA_HI,
                     n_init=None, n_lo=None, n_hi=None, nseeds=5):
    """Fit the comb.  Returns a dict, or None if scipy is missing."""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return {"ok": False, "why": "scipy is not installed; use --no-fit"}

    x = np.asarray(centres, dtype=float)
    y = np.asarray(y, dtype=float)
    err = np.asarray(yerr, dtype=float)
    err = np.where(np.isfinite(err) & (err > 0), err, 1.0)
    w = 1.0 / err

    # two peaks closer together than 2 sigma_min cannot be told apart, so a
    # comb finer than that is not a photon comb -- it is the model chasing
    # whatever structure the quadratic background cannot follow
    if n_lo is None:
        n_lo = max(2.0 * (x[1] - x[0]), 2.0 * sigma_lo)
    if n_hi is None:
        n_hi = (float(x[-1]) + 0.5 * (x[1] - x[0])) / npeaks
    n_lo, n_hi = float(n_lo), float(n_hi)
    if not n_hi > n_lo:
        return {"ok": False, "why": "empty N range (%.3g, %.3g)" % (n_lo, n_hi)}

    scan = seed_gain(x, y, w, npeaks, sigma_init, sigma_lo, sigma_hi,
                     n_lo, n_hi)
    if n_init is not None:
        a = _design(x, float(n_init), [sigma_init] * npeaks, npeaks)
        coef, _, _, _ = np.linalg.lstsq(a * w[:, None], y * w, rcond=None)
        seeds = [(0.0, float(n_init), sigma_init, coef)]
    else:
        seeds = spread_seeds(scan, nseeds, n_lo, n_hi)

    def model(xx, *pars):
        return peak_model(xx, pars, npeaks)

    lo = [-np.inf, -np.inf, -np.inf, n_lo] + [0.0] * npeaks \
        + [sigma_lo] * npeaks
    hi = [np.inf, np.inf, np.inf, n_hi] + [np.inf] * npeaks \
        + [sigma_hi] * npeaks
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    best = None
    for _chi2, gain, sg, coef in seeds:
        amps = np.clip(np.asarray(coef[3:], dtype=float), 1e-6, None)
        p_init = np.concatenate(([coef[0], coef[1], coef[2], gain], amps,
                                 np.full(npeaks, sg)))
        p_init = np.clip(p_init, lo + 1e-9, hi - 1e-9)
        try:
            popt, pcov = curve_fit(model, x, y, p0=p_init, sigma=err,
                                   absolute_sigma=True, bounds=(lo, hi),
                                   maxfev=200000)
        except (RuntimeError, ValueError):
            continue
        r = (model(x, *popt) - y) / err
        chi2 = float(r @ r)
        if best is None or chi2 < best[0]:
            best = (chi2, popt, pcov)

    if best is None:
        return {"ok": False, "why": "the fit did not converge from any seed"}

    chi2, popt, pcov = best
    with np.errstate(invalid="ignore"):
        perr = np.sqrt(np.abs(np.diag(pcov)))
    ndf = len(x) - len(popt)

    return {"ok": True, "npeaks": npeaks, "p": popt, "e": perr,
            "chi2": chi2, "ndf": ndf,
            "gain": float(popt[3]), "gain_err": float(perr[3]),
            "bkg": popt[:3], "bkg_err": perr[:3],
            "amp": popt[4:4 + npeaks], "amp_err": perr[4:4 + npeaks],
            "sig": popt[4 + npeaks:4 + 2 * npeaks],
            "sig_err": perr[4 + npeaks:4 + 2 * npeaks],
            "n_lo": float(n_lo), "n_hi": float(n_hi),
            "scan": [(c, g) for c, g, _s, _k
                     in spread_seeds(scan, 6, n_lo, n_hi)],
            "sigma_lo": float(sigma_lo), "sigma_hi": float(sigma_hi),
            "sigma_init": float(sigma_init)}


def fit_lines(fit):
    """The parameter block, as a list of monospaced lines."""
    if not fit or not fit.get("ok"):
        return ["fit failed: %s" % (fit or {}).get("why", "unknown")]
    b, be = fit["bkg"], fit["bkg_err"]
    lines = ["%-6s %9s %12s %11s"
             % ("peak", "mean", "amplitude", "sigma")]
    for i in range(fit["npeaks"]):
        lines.append("%-6d %9.2f %6.1f\u00b1%-5.1f %5.1f\u00b1%-5.1f"
                     % (i + 1, (i + 1) * fit["gain"],
                        fit["amp"][i], fit["amp_err"][i],
                        fit["sig"][i], fit["sig_err"][i]))
    lines.append("")
    lines.append("background  p0 = %.4g \u00b1 %.2g" % (b[0], be[0]))
    lines.append("            p1 = %.4g \u00b1 %.2g" % (b[1], be[1]))
    lines.append("            p2 = %.4g \u00b1 %.2g" % (b[2], be[2]))
    lines.append("")
    lines.append("chi2/ndf = %.1f / %d = %.2f"
                 % (fit["chi2"], fit["ndf"],
                    fit["chi2"] / fit["ndf"] if fit["ndf"] else np.nan))
    return lines


def print_fit(fit):
    """The same numbers, on the terminal."""
    print("=" * 78)
    if not fit or not fit.get("ok"):
        print("fit        FAILED: %s" % (fit or {}).get("why", "unknown"))
        print("=" * 78)
        return
    k = fit["npeaks"]
    print("fit        background(2nd order) + %d photon peaks at N, 2N, ... %dN"
          % (k, k))
    print("           y = p0 + p1 x + p2 x^2 + SUM A_i exp(-0.5 ((x-iN)/s_i)^2)")
    print("           fitted over %.4g to %.4g ADC"
          % (fit.get("fit_lo", float("nan")), fit.get("fit_hi", float("nan"))))
    print("           N in (%.3g, %.3g);  sigma start %.3g, in (%.3g, %.3g)"
          % (fit["n_lo"], fit["n_hi"], fit["sigma_init"],
             fit["sigma_lo"], fit["sigma_hi"]))
    print("-" * 78)
    print("")
    print("     >>>>   N  =  %.3f  +-  %.3f   HG ADC per photoelectron   <<<<"
          % (fit["gain"], fit["gain_err"]))
    print("")
    print("-" * 78)
    print("  %-6s %12s %16s %16s"
          % ("peak", "mean [ADC]", "amplitude", "sigma [ADC]"))
    for i in range(k):
        s = fit["sig"][i]
        flag = ""
        if fit["amp"][i] < 2.0 * fit["amp_err"][i]:
            flag = "  <- amplitude consistent with zero"
        elif s <= fit["sigma_lo"] * 1.001:
            flag = "  <- sigma at the lower bound"
        elif s >= fit["sigma_hi"] * 0.999:
            flag = "  <- sigma at the upper bound"
        print("  %-6d %12.2f %8.2f +- %-5.2f %7.2f +- %-5.2f%s"
              % (i + 1, (i + 1) * fit["gain"], fit["amp"][i],
                 fit["amp_err"][i], s, fit["sig_err"][i], flag))
    b, be = fit["bkg"], fit["bkg_err"]
    print("  %-6s p0 = %.4g +- %.3g   p1 = %.4g +- %.3g   p2 = %.4g +- %.3g"
          % ("bkg", b[0], be[0], b[1], be[1], b[2], be[2]))
    print("-" * 78)
    red = fit["chi2"] / fit["ndf"] if fit["ndf"] else float("nan")
    print("  chi2 / ndf = %.2f / %d = %.3f" % (fit["chi2"], fit["ndf"], red))
    if fit.get("scan"):
        print("  other comb spacings the seed scan found, best first:")
        print("         " + "   ".join("N=%.2f" % g for _c, g in fit["scan"]))
        print("         pick one with --n-init, or fence N in with "
              "--n-min/--n-max")
    empty = [i + 1 for i in range(k)
             if fit["amp"][i] < 2.0 * fit["amp_err"][i]]
    if empty:
        print("  WARNING  peak(s) %s have no amplitude.  A comb with empty "
              "slots is the" % ", ".join(str(i) for i in empty))
        print("           signature of N landing on a sub-multiple of the "
              "true spacing:")
        print("           if peak 1 is the empty one, the real N is probably "
              "%.2f." % (2.0 * fit["gain"]))
        print("           Reseed with --n-init, or raise --fit-xmin.")
    if red > 3.0:
        print("  NOTE   chi2/ndf is large.  Try raising --fit-xmin further, "
              "or --npeaks,")
        print("         or check that the peaks the fit chose are the ones "
              "you see.")
    print("=" * 78)


# ----------------------------------------------------------------------------
# the 50 active SiPMs in one panel
# ----------------------------------------------------------------------------
def make_all50(vals, out_paths, title, subtitle, lo, hi, nbins,
               logy=False, column="HG", show_individual=False,
               dofit=True, npeaks=NPEAKS, sigma_init=SIGMA_INIT,
               sigma_lo=SIGMA_LO, sigma_hi=SIGMA_HI,
               n_init=N_INIT, n_lo=None, n_hi=None,
               fit_lo=FIT_LO, fit_hi=None):
    """One axis: the mean spectrum of the active SiPMs, with the comb fit.

    All 50 channels have the same number of entries -- one per event -- so the
    pooled spectrum divided by 50 is the average channel.  With
    show_individual the 50 channels are drawn faintly behind it, on the same
    scale; the y axis then covers them too.

    The mean spectrum is fitted with a 2nd-order polynomial plus npeaks
    Gaussians whose means are 1, 2, ... npeaks times one free gain N.  The
    fit, its background, the individual peaks and the parameters are drawn on
    top, and the fit dict is returned alongside the usual numbers.
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

    # ------------------------------------------------------------------
    # the fit
    # ------------------------------------------------------------------
    fit = None
    flo = lo if fit_lo is None else max(lo, float(fit_lo))
    fhi = hi if fit_hi is None else min(hi, float(fit_hi))
    if dofit:
        # the mean of nch histograms of Poisson counts
        err = np.sqrt(np.maximum(pooled, 1.0)) / float(nch)
        sel = (centres >= flo) & (centres <= fhi)
        if sel.sum() < 2 * npeaks + 5:
            fit = {"ok": False, "why": "only %d bins in the fit range %g-%g"
                                       % (int(sel.sum()), flo, fhi)}
        else:
            fit = fit_photon_peaks(centres[sel], mean[sel], err[sel],
                                   npeaks=npeaks, sigma_init=sigma_init,
                                   sigma_lo=sigma_lo, sigma_hi=sigma_hi,
                                   n_init=n_init, n_lo=n_lo, n_hi=n_hi)
            if fit.get("ok"):
                fit["fit_lo"], fit["fit_hi"] = float(flo), float(fhi)

    if fit and fit.get("ok"):
        gain = fit["gain"]
        xf = np.linspace(flo, fhi, 1000)
        if flo > lo:
            ax.axvspan(lo, flo, color="0.55", alpha=0.10, zorder=0)
            ax.annotate("not fitted", xy=(0.5 * (lo + flo), 0.985),
                        xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=7.5, color="0.45")
        bkg = fit["p"][0] + fit["p"][1] * xf + fit["p"][2] * xf * xf

        # each peak, sitting on the background it was fitted over
        for i in range(fit["npeaks"]):
            g = fit["amp"][i] * np.exp(
                -0.5 * ((xf - (i + 1) * gain) / fit["sig"][i]) ** 2)
            ax.plot(xf, bkg + g, color="#2a78d6", lw=0.9, ls="--", zorder=5,
                    label="single peaks" if i == 0 else None)
            ax.axvline((i + 1) * gain, color="#2a78d6", lw=0.7, ls=":",
                       alpha=0.55, zorder=1)
            # at the foot of the line, so the labels never fight the legend
            ax.annotate("%d p.e." % (i + 1), xy=((i + 1) * gain, 0.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8,
                        color="#2a78d6", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                  edgecolor="none", alpha=0.85))

        ax.plot(xf, bkg, color="0.45", lw=1.1, ls="-.", zorder=5,
                label="background  p0 + p1 x + p2 x$^2$")
        ax.plot(xf, peak_model(xf, fit["p"], fit["npeaks"]), color="#cc3311",
                lw=2.0, zorder=6,
                label="fit: background + %d peaks" % fit["npeaks"])
        ymax = max(ymax, float(np.max(peak_model(xf, fit["p"],
                                                 fit["npeaks"]))))

    if show_individual:
        handles = [plt.Line2D([], [], color=COLCOLOUR[c], lw=1.2, alpha=0.6,
                              label="column %s" % c) for c in COLS
                   if c in seen_cols]
        handles.append(plt.Line2D([], [], color="0.1", lw=1.6,
                                  label="mean of all %d" % nch))
        if fit and fit.get("ok"):
            handles.append(plt.Line2D([], [], color="#cc3311", lw=2.0,
                                      label="fit: background + %d peaks"
                                            % fit["npeaks"]))
            handles.append(plt.Line2D([], [], color="0.45", lw=1.1, ls="-.",
                                      label="background"))
            handles.append(plt.Line2D([], [], color="#2a78d6", lw=0.9,
                                      ls="--", label="single peaks"))
        ax.legend(handles=handles, fontsize=8.5, frameon=False,
                  loc="upper right")
    else:
        ax.legend(fontsize=8.5, frameon=False, loc="upper right")

    ax.set_xlabel("%s ADC counts" % column, fontsize=11)
    ax.set_ylabel("events / bin, per SiPM", fontsize=11)
    ax.set_xlim(lo, hi)
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, YHEADROOM * ymax if ymax > 0 else 1)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=9)

    # ---- N, big, because N is the point of the plot ----
    if fit and fit.get("ok"):
        ax.text(0.985, 0.735,
                "N = %.2f $\\pm$ %.2f" % (fit["gain"], fit["gain_err"]),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=21, fontweight="bold", color="#cc3311", zorder=10,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          edgecolor="#cc3311", lw=1.6, alpha=0.96))
        ax.text(0.985, 0.652, "%s ADC counts per photoelectron" % column,
                transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
                color="#cc3311", zorder=10)
        ax.text(0.985, 0.605, "\n".join(fit_lines(fit)),
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                family="monospace", color="0.15", zorder=10, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor="0.75", lw=0.8, alpha=0.94))
    elif fit:
        ax.text(0.985, 0.735, "fit failed\n%s" % fit.get("why", ""),
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                color="#cc3311", zorder=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#cc3311", lw=1.2))

    fig.suptitle(title, fontsize=14, y=0.975)
    fig.text(0.5, 0.935, subtitle, ha="center", va="top", fontsize=9,
             color="0.25", linespacing=1.5)
    for path in out_paths:
        fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return nch, ntot, ymax, fit


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
    p.add_argument("--npeaks", type=int, default=NPEAKS,
                   help="number of photon peaks in the fit, at N, 2N, ... "
                        "(default %d)" % NPEAKS)
    p.add_argument("--no-fit", action="store_true",
                   help="do not fit the all-50 spectrum")
    p.add_argument("--sigma-init", type=float, default=SIGMA_INIT,
                   help="starting width of every peak (default %g)"
                        % SIGMA_INIT)
    p.add_argument("--sigma-min", type=float, default=SIGMA_LO,
                   help="lower bound on every peak width (default %g)"
                        % SIGMA_LO)
    p.add_argument("--sigma-max", type=float, default=SIGMA_HI,
                   help="upper bound on every peak width (default %g)"
                        % SIGMA_HI)
    p.add_argument("--n-init", type=float, default=N_INIT,
                   help="starting value for the gain N (default %g; pass 0 to "
                        "seed it by scanning N over its allowed range "
                        "instead)" % N_INIT)
    p.add_argument("--fit-xmin", type=float, default=FIT_LO,
                   help="low edge of the FIT range, independent of the plot "
                        "range (default %g, which keeps the pedestal tail out "
                        "of the fit)" % FIT_LO)
    p.add_argument("--fit-xmax", type=float, default=None,
                   help="high edge of the fit range (default: --xmax)")
    p.add_argument("--n-min", type=float, default=None,
                   help="lower bound on N (default: 2 bin widths)")
    p.add_argument("--n-max", type=float, default=None,
                   help="upper bound on N (default: xmax / npeaks, so the "
                        "last peak stays inside the range)")
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
        nch, nent, top50, fit = make_all50(
            vals, paths, "%s   photon peaks   all %d active SiPMs"
            % (tag, len(ACTIVE_CH)),
            base + ("  |  every active SiPM faint, their mean in black"
                    if args.all50_individual
                    else "  |  mean of the active SiPMs") + stamp,
            args.xmin, args.xmax, args.nbins, logy=args.logy,
            column=args.column, show_individual=args.all50_individual,
            dofit=not args.no_fit, npeaks=args.npeaks,
            sigma_init=args.sigma_init, sigma_lo=args.sigma_min,
            sigma_hi=args.sigma_max,
            n_init=(args.n_init if args.n_init and args.n_init > 0 else None),
            n_lo=args.n_min, n_hi=args.n_max,
            fit_lo=args.fit_xmin, fit_hi=args.fit_xmax)
        print("all50      %d SiPMs, %d entries pooled (%d per SiPM)"
              % (nch, nent, nent // nch if nch else 0))
        if not args.logy:
            print("           highest bin %.1f -> y axis top %.1f"
                  % (top50, YHEADROOM * top50))
        written += paths
        if fit is not None:
            print_fit(fit)

    print("-" * 78)
    for path in written:
        print("wrote %s" % path)


if __name__ == "__main__":
    main()
