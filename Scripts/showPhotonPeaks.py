#!/usr/bin/env python3
"""
showPhotonPeaks.py

Single-photoelectron spectra for the 64 channels of a CAEN Janus / DT5202 run.

Usage
-----
    python showPhotonPeaks.py RUNFILE.txt [options]

The input file may cover several runs.  A name of the form

    Run132_133_137_list.txt

is read as runs 132, 133 and 137: the title says "Runs 132, 133, 137" and every
output keeps the full list, e.g. Run132_133_137_photonPeaks.png.  A single run
behaves as before.

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
                                                with the photon-peak fit and a
                                                data/fit ratio panel under it
                                                (--no-ratio drops the panel)
    RunX_photonPeaks_fit.txt                    the fit report as plain text

The fit
-------
The all-50 spectrum is fitted with

    y(x) = bkg(x)
           + SUM(i = 1..npeaks)  A_i exp( -0.5 ((x - (P + i M)) / sigma_i)^2 )

a falling background plus a comb of Gaussians sitting at

    P + M,  P + 2M,  P + 3M,  ...

P is the PEDESTAL -- the ADC value of a channel with no avalanche -- and M is
the single-photoelectron spacing, the number this plot exists to measure.  Both
are free parameters, so the comb is allowed to sit on a non-zero baseline
instead of being forced through the origin.  P is reported on the plot, in the
text file and on the terminal, and is drawn as a labelled marker when it falls
inside the x range.

P is bounded to (0, the low edge of the fit range) by default: the pedestal is
a positive ADC value that lies below where the fit starts.  --ped-min /
--ped-max move that window, --ped-init reseeds it and --fix-ped holds P at a
value you already know, e.g. one measured from a random-trigger run.

--ped-peak additionally puts the 0-photoelectron Gaussian at P itself, so the
comb runs P, P+M, P+2M, ...  Use it with --fit-xmin low enough to include the
pedestal peak.

Each peak keeps its own free width, started at 20 and bounded to (5, 50).

The background
--------------
By default the background is held NON-NEGATIVE and FALLING across the fit
range -- a spectrum's continuum does not rise with energy, and letting it do so
is how the comb gets dragged off the real peaks.

Box bounds on p0, p1, p2 cannot say "this quadratic never rises", so the
background is fitted in a basis of the fit interval whose members are each
non-negative and non-increasing,

    bkg(x) = d0 (1-t)^2 + d1 (1 - t^2) + d2 ,    t = (x - a) / (b - a)

with d0, d1, d2 >= 0.  Every such curve has

    d bkg / dt = -2 d0 (1-t) - 2 d1 t  <=  0        on [a, b]

so an inequality on the CURVE has become three box bounds on the PARAMETERS,
which is what curve_fit can enforce.  (This is the Bernstein quadratic with its
coefficients ordered c0 >= c1 >= c2 >= 0.)  The constraint is non-increasing
rather than strictly decreasing -- d0 = d1 = 0 leaves a flat background -- since
a strict inequality is not a closed constraint.  The fitted steepest slope is
reported so you can see how close to flat it ended up.

The fitted d are converted back to p0, p1, p2 for the report, so nothing
downstream changes.  --allow-rising-bkg relaxes to non-negative only (the old
Bernstein basis), --allow-negative-bkg restores the fully unconstrained
polynomial, and --nobkg removes it altogether: the three coefficients are
dropped from the parameter vector rather than set to zero, so they cost no
degrees of freedom and ndf reflects what was actually fitted.

M starts at 35 (--n-init) and the fit runs over 20 ADC and up (--fit-xmin),
leaving the pedestal tail out of it; the plot still shows the whole range.
--n-init 0 instead seeds M by scanning it over a grid and solving the
remaining, linear, parameters exactly at each step.  --npeaks changes the
number of peaks, --no-fit switches it off.

    A matching .pdf is written beside each png; --no-pdf skips them.
    The fit report is also written as a .txt; --no-txt skips it.
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
NPEAKS = 5          # Gaussians in the comb: means at P+M, P+2M, ... P+npeaks*M
N_INIT = 35.0       # starting value for the gain M
SIGMA_INIT = 20.0   # starting width of every peak
SIGMA_LO = 5.0      # and the window it is allowed to move in
SIGMA_HI = 50.0
NSCAN = 400         # grid points in the seed scan over M

# A "peak" wider than about half the spacing is not a peak -- adjacent ones
# merge and the Gaussian starts doing the background's job, which is how the
# comb slides away from its seed.  Widths are therefore fitted as a FRACTION
# of M, bounded here, instead of as free absolute numbers.
SIG_FRAC_LO = 0.05
SIG_FRAC_HI = 0.50

# The fit starts above the pedestal tail.  Below ~20 ADC the spectrum falls far
# too steeply for a 2nd-order polynomial, and the comb gets dragged into it --
# which is how M ends up at half its true value with the odd peaks switched off.
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

# background modes, loosest last
BKG_NONE = "none"
BKG_FALLING = "falling"     # >= 0 and non-increasing        (the default)
BKG_NONNEG = "nonneg"       # >= 0, free to rise
BKG_FREE = "free"           # unconstrained quadratic

BKG_LABEL = {BKG_FALLING: "falling, >= 0",
             BKG_NONNEG: ">= 0",
             BKG_FREE: "unconstrained",
             BKG_NONE: "none"}


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


# ----------------------------------------------------------------------------
# run names: one file may hold several runs
# ----------------------------------------------------------------------------
def run_numbers(path):
    """'Run132_133_137_list.txt' -> ['132', '133', '137'].

    The run number may be followed by any number of further numbers joined by
    '_' or '-'; the chain stops at the first non-numeric field ('_list'), so
    the usual single-run names are unaffected.  'Runs134_137_list.txt', as
    written by combineRuns.py, is understood the same way.
    """
    base = os.path.basename(path)
    m = re.search(r"runs?[_\-\s]*0*(\d+(?:[_\-]0*\d+)*)", base, re.IGNORECASE)
    if not m:
        return []
    out = []
    for piece in re.split(r"[_\-]", m.group(1)):
        piece = piece.lstrip("0") or "0"
        if piece not in out:
            out.append(piece)
    return out


def run_stem(path):
    """The file-name stem: 'Run125', 'Runs132_133_137', or the bare name.

    Plural for several runs, so the outputs line up with the combined list
    file combineRuns.py writes ('Runs132_133_137_list.txt').
    """
    nums = run_numbers(path)
    if nums:
        return ("Run" if len(nums) == 1 else "Runs") + "_".join(nums)
    return os.path.splitext(os.path.basename(path))[0]


def run_label(path):
    """The title text: 'Run125', or 'Runs 132, 133, 137' for several."""
    nums = run_numbers(path)
    if not nums:
        return os.path.splitext(os.path.basename(path))[0]
    if len(nums) == 1:
        return "Run%s" % nums[0]
    return "Runs " + ", ".join(nums)


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
def _peaks_only(x, pars, npeaks, ped=0.0, i0=1):
    """The comb alone: sum_i A_i Gauss(mean = P + i*M, sigma = sigma_i)."""
    y = np.zeros_like(np.asarray(x, dtype=float))
    gain = pars[3]
    for i in range(npeaks):
        amp = pars[4 + i]
        sig = pars[4 + npeaks + i]
        y = y + amp * np.exp(-0.5 * ((x - (ped + (i0 + i) * gain))
                                     / sig) ** 2)
    return y


def peak_model(x, pars, npeaks, ped=0.0, i0=1):
    """p0 + p1 x + p2 x^2 + the comb.  pars[:3] are monomial coefficients."""
    x = np.asarray(x, dtype=float)
    return pars[0] + pars[1] * x + pars[2] * x * x \
        + _peaks_only(x, pars, npeaks, ped, i0)


# ---- the constrained backgrounds -------------------------------------------
# Two bases of the fit interval [a, b], both chosen so that a NON-NEGATIVE
# combination of their members automatically satisfies the constraint we want,
# turning an inequality on the curve into box bounds on the parameters.
#
#   nonneg   Bernstein:  (1-t)^2 , 2t(1-t) , t^2
#            every member >= 0, so the curve is >= 0 on [a, b].
#
#   falling  (1-t)^2 , 1 - t^2 , 1
#            every member is >= 0 AND non-increasing, so the curve is >= 0 and
#            never rises:  d/dt = -2 d0 (1-t) - 2 d1 t <= 0 on [0, 1].
#            (Equivalently the Bernstein coefficients obey c0 >= c1 >= c2 >= 0.)
def _bern_basis(x, a, b):
    t = (np.asarray(x, dtype=float) - a) / float(b - a)
    return np.vstack([(1.0 - t) ** 2, 2.0 * t * (1.0 - t), t * t]).T


def _fall_basis(x, a, b):
    t = (np.asarray(x, dtype=float) - a) / float(b - a)
    return np.vstack([(1.0 - t) ** 2, 1.0 - t * t, np.ones_like(t)]).T


# coefficient vector -> monomial in t, for each basis
_M_BERN = np.array([[1.0, 0.0, 0.0],
                    [-2.0, 2.0, 0.0],
                    [1.0, -2.0, 1.0]])
_M_FALL = np.array([[1.0, 1.0, 1.0],
                    [-2.0, 0.0, 0.0],
                    [1.0, -1.0, 0.0]])


def _bkg_basis(mode, x, a, b):
    if mode == BKG_FALLING:
        return _fall_basis(x, a, b)
    if mode == BKG_NONNEG:
        return _bern_basis(x, a, b)
    x = np.asarray(x, dtype=float)
    return np.vstack([np.ones_like(x), x, x * x]).T


def _bkg_curve(mode, x, c, a, b):
    return _bkg_basis(mode, x, a, b) @ np.asarray(c, dtype=float)


def _to_poly(mode, c, a, b):
    """Basis coefficients -> (p0, p1, p2), and the Jacobian of that map."""
    c = np.asarray(c, dtype=float)
    if mode not in (BKG_FALLING, BKG_NONNEG):
        return c, np.eye(3)
    L = float(b - a)
    m1 = _M_FALL if mode == BKG_FALLING else _M_BERN
    m2 = np.array([[1.0, -a / L, a * a / (L * L)],
                   [0.0, 1.0 / L, -2.0 * a / (L * L)],
                   [0.0, 0.0, 1.0 / (L * L)]])
    j = m2 @ m1
    return j @ c, j


def _design(x, gain, sigmas, npeaks, mode, a=None, b=None, ped=0.0, i0=1):
    """Columns of the model that multiply the linear parameters.

    For the constrained modes the first three columns are the basis of [a, b]
    whose non-negative combinations obey the constraint, so every linear
    coefficient of the model -- background and peak amplitudes alike -- is one
    that must come out non-negative.  mode='none' drops the background columns,
    leaving the comb on its own.
    """
    x = np.asarray(x, dtype=float)
    if mode == BKG_NONE:
        cols = []
    else:
        cols = list(_bkg_basis(mode, x, a, b).T)
    for i in range(npeaks):
        cols.append(np.exp(-0.5 * ((x - (ped + (i0 + i) * gain))
                                   / sigmas[i]) ** 2))
    return np.vstack(cols).T


def _solve_linear(a, y, w, nonneg):
    """Weighted linear solve, constrained to non-negative coefficients."""
    aw, yw = a * w[:, None], y * w
    if nonneg:
        try:
            from scipy.optimize import nnls
            coef, _ = nnls(aw, yw)
            return coef
        except (ImportError, RuntimeError, ValueError):
            coef, _, _, _ = np.linalg.lstsq(aw, yw, rcond=None)
            return np.clip(coef, 0.0, None)
    coef, _, _, _ = np.linalg.lstsq(aw, yw, rcond=None)
    return coef


def seed_gain(x, y, w, npeaks, sigma_init, sigma_lo, sigma_hi, n_lo, n_hi,
              nscan=NSCAN, mode=BKG_FALLING, a=None, b=None, ped=0.0, i0=1):
    """Scan M; at each M the rest of the model is linear, so solve it exactly.

    The comb has a local minimum at every sub-multiple of the true spacing, so
    a scan is the only reliable way in.  The width is scanned coarsely too --
    holding it at one value tilts the landscape towards combs of that width.

    Returns (chi2, M, sigma, linear coefficients) sorted best first.
    """
    nonneg = mode in (BKG_FALLING, BKG_NONNEG, BKG_NONE)
    widths = sorted({min(max(f * sigma_init, sigma_lo), sigma_hi)
                     for f in (0.5, 1.0, 1.5)})
    out = []
    for sg in widths:
        sig = [sg] * npeaks
        for gain in np.linspace(n_lo, n_hi, nscan):
            dm = _design(x, gain, sig, npeaks, mode, a, b, ped, i0)
            try:
                coef = _solve_linear(dm, y, w, nonneg)
            except np.linalg.LinAlgError:
                continue
            r = (dm @ coef - y) * w
            out.append((float(r @ r), float(gain), float(sg), coef))
    out.sort(key=lambda t: t[0])
    return out


def spread_seeds(scan, nseeds, n_lo, n_hi):
    """The best seeds, kept apart in M so they explore different combs."""
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
                     n_init=None, n_lo=None, n_hi=None, nseeds=5,
                     bkg_mode=BKG_FALLING, sig_frac_lo=SIG_FRAC_LO,
                     sig_frac_hi=SIG_FRAC_HI, i0=1,
                     ped_init=None, ped_lo=None, ped_hi=None, fix_ped=None):
    """Fit the comb at P + M, P + 2M, ...  Returns a dict, or None on failure.

    P (the pedestal) and M (the spacing) are both free unless fix_ped pins P.
    """
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return {"ok": False, "why": "scipy is not installed; use --no-fit"}

    x = np.asarray(centres, dtype=float)
    y = np.asarray(y, dtype=float)
    err = np.asarray(yerr, dtype=float)
    err = np.where(np.isfinite(err) & (err > 0), err, 1.0)
    w = 1.0 / err

    # the constrained bases live on the fit interval, edge to edge
    half = 0.5 * (x[1] - x[0])
    xa, xb = float(x[0] - half), float(x[-1] + half)

    # ---- the pedestal P --------------------------------------------------
    # default window: P is a positive ADC value BELOW where the fit starts,
    # which is also what stops the comb re-indexing itself (P -> P + M with the
    # same M is the same comb shifted by one peak).  With the pedestal peak in
    # the comb, P must instead lie inside the fit range, near its start.
    if ped_lo is None:
        ped_lo = xa if i0 == 0 else 0.0
    if ped_hi is None:
        ped_hi = (xa + (xb - xa) / float(npeaks + 1)) if i0 == 0 \
            else max(xa, 1.0)
    ped_lo, ped_hi = float(ped_lo), float(ped_hi)
    if not ped_hi > ped_lo:
        ped_hi = ped_lo + 1.0
    if ped_init is None:
        if i0 == 0:
            # the pedestal peak is the tallest thing near the start of the
            # range, so seed P there rather than at the middle of the window
            near = x <= xa + 0.3 * (xb - xa)
            ped_init = float(x[near][int(np.argmax(y[near]))]) if near.any() \
                else 0.5 * (ped_lo + ped_hi)
        else:
            ped_init = 0.5 * (ped_lo + ped_hi)
    ped_init = float(min(max(ped_init, ped_lo), ped_hi))

    fit_ped = fix_ped is None
    ped_fixed = ped_init if fit_ped else float(fix_ped)
    nped = 1 if fit_ped else 0

    # two peaks closer together than 2 sigma_min cannot be told apart, so a
    # comb finer than that is not a photon comb -- it is the model chasing
    # whatever structure the quadratic background cannot follow
    if n_lo is None:
        n_lo = max(2.0 * (x[1] - x[0]), 2.0 * sigma_lo)
    if n_hi is None:                      # keep the last peak inside the range
        n_hi = (xb - ped_fixed) / float(npeaks + (1 if i0 == 0 else 0))
    n_lo, n_hi = float(n_lo), float(n_hi)
    if not n_hi > n_lo:
        return {"ok": False, "why": "empty M range (%.3g, %.3g)" % (n_lo, n_hi)}

    # with --nobkg the three polynomial coefficients are not fitted at all --
    # they are dropped from the parameter vector, so they cost no degrees of
    # freedom, and padded back with zeros afterwards
    nbkg = 0 if bkg_mode == BKG_NONE else 3
    nonneg_lin = bkg_mode in (BKG_FALLING, BKG_NONNEG, BKG_NONE)

    scan = seed_gain(x, y, w, npeaks, sigma_init, sigma_lo, sigma_hi,
                     n_lo, n_hi, mode=bkg_mode, a=xa, b=xb,
                     ped=ped_fixed, i0=i0)
    if n_init is not None:
        dm = _design(x, float(n_init), [sigma_init] * npeaks, npeaks,
                     bkg_mode, xa, xb, ped_fixed, i0)
        coef = _solve_linear(dm, y, w, nonneg_lin)
        seeds = [(0.0, float(n_init), sigma_init, coef)]
    else:
        seeds = spread_seeds(scan, nseeds, n_lo, n_hi)

    # index of each block in the padded parameter vector
    i_gain = 3
    i_amp = 4
    i_sig = 4 + npeaks
    i_ped = 4 + 2 * npeaks
    n_par = 4 + 2 * npeaks

    # the width parameters carried through the fit are r_i = sigma_i / M
    def _abs_pars(pars):
        pars = np.asarray(pars, dtype=float)
        out = pars.copy()
        out[i_sig:i_sig + npeaks] = pars[i_sig:i_sig + npeaks] * pars[i_gain]
        return out

    def _full(pars):
        """Pad back out to p0, p1, p2, M, A..., r..., [P]."""
        pars = np.asarray(pars, dtype=float)
        return pars if nbkg == 3 else np.concatenate((np.zeros(3), pars))

    def model(xx, *pars):
        xx = np.asarray(xx, dtype=float)
        fp = _full(pars)
        ped_v = fp[i_ped] if nped else ped_fixed
        if nbkg == 0:
            bkg = 0.0
        else:
            bkg = _bkg_curve(bkg_mode, xx, fp[:3], xa, xb)
        return bkg + _peaks_only(xx, _abs_pars(fp), npeaks, ped_v, i0)

    if nbkg == 0:
        bkg_lo, bkg_hi = [], []
    elif bkg_mode == BKG_FREE:
        bkg_lo, bkg_hi = [-np.inf] * 3, [np.inf] * 3
    else:
        bkg_lo, bkg_hi = [0.0] * 3, [np.inf] * 3

    # the absolute sigma bounds still apply, as a fraction of the widest /
    # narrowest gain the fit is allowed to reach
    r_lo = max(sig_frac_lo, sigma_lo / n_hi)
    r_hi = min(sig_frac_hi, sigma_hi / n_lo)
    if not r_hi > r_lo:
        r_lo, r_hi = sig_frac_lo, sig_frac_hi

    lo = bkg_lo + [n_lo] + [0.0] * npeaks + [r_lo] * npeaks
    hi = bkg_hi + [n_hi] + [np.inf] * npeaks + [r_hi] * npeaks
    if nped:
        lo = lo + [ped_lo]
        hi = hi + [ped_hi]
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    best = None
    for _chi2, gain, sg, coef in seeds:
        coef = np.asarray(coef, dtype=float)
        amps = np.clip(coef[nbkg:], 1e-6, None)
        p_init = np.concatenate((coef[:nbkg], [gain], amps,
                                 np.full(npeaks, sg / max(gain, 1e-9)),
                                 [ped_init] if nped else []))
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
    ndf = len(x) - len(popt)        # the dropped background costs nothing

    if nbkg == 0:                   # pad the zero background back in
        popt = np.concatenate((np.zeros(3), popt))
        big = np.zeros((len(popt), len(popt)))
        big[3:, 3:] = pcov
        pcov = big

    # sigma_i = r_i * M: convert back and push the covariance through
    jr = np.eye(len(popt))
    gain_v = popt[i_gain]
    for i in range(npeaks):
        k = i_sig + i
        jr[k, i_gain] = popt[k]     # d sigma_i / dM  = r_i
        jr[k, k] = gain_v           # d sigma_i / dr_i = M
    popt = popt.copy()
    popt[i_sig:i_sig + npeaks] *= gain_v
    pcov = jr @ pcov @ jr.T

    ped_val = float(popt[i_ped]) if nped else float(ped_fixed)
    ped_err = float(np.sqrt(abs(pcov[i_ped, i_ped]))) if nped else 0.0
    popt, pcov = popt[:n_par], pcov[:n_par, :n_par]

    # hand back monomial coefficients, so peak_model and every printout keep
    # working unchanged -- transforming the covariance with them, not after
    if bkg_mode in (BKG_FALLING, BKG_NONNEG) and nbkg == 3:
        pc, jac = _to_poly(bkg_mode, popt[:3], xa, xb)
        jfull = np.eye(len(popt))
        jfull[:3, :3] = jac
        popt = np.concatenate((pc, popt[3:]))
        pcov = jfull @ pcov @ jfull.T
    with np.errstate(invalid="ignore"):
        perr = np.sqrt(np.abs(np.diag(pcov)))

    bkg_min = bkg_slope_max = None
    if nbkg == 3:
        xs = np.linspace(xa, xb, 1024)
        bk = popt[0] + popt[1] * xs + popt[2] * xs * xs
        bkg_min = float(np.min(bk))
        bkg_slope_max = float(np.max(popt[1] + 2.0 * popt[2] * xs))

    return {"ok": True, "npeaks": npeaks, "p": popt, "e": perr,
            "ped": ped_val, "ped_err": ped_err, "ped_fitted": bool(nped),
            "ped_lo": ped_lo, "ped_hi": ped_hi, "i0": int(i0),
            "bkg_mode": bkg_mode, "bkg_min": bkg_min,
            "bkg_slope_max": bkg_slope_max,
            "nobkg": bkg_mode == BKG_NONE,
            "bern_a": xa, "bern_b": xb,
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
            "sigma_init": float(sigma_init),
            "n_seed": float(n_init) if n_init is not None else None,
            "r_lo": float(r_lo), "r_hi": float(r_hi)}


def fit_lines(fit):
    """The parameter block, as a list of monospaced lines."""
    if not fit or not fit.get("ok"):
        return ["fit failed: %s" % (fit or {}).get("why", "unknown")]
    b, be = fit["bkg"], fit["bkg_err"]
    lines = ["%-6s %9s %12s %11s"
             % ("peak", "mean", "amplitude", "sigma")]
    for i in range(fit["npeaks"]):
        lines.append("%-6d %9.2f %6.1f±%-5.1f %5.1f±%-5.1f"
                     % (fit.get("i0", 1) + i,
                        fit["ped"] + (fit.get("i0", 1) + i) * fit["gain"],
                        fit["amp"][i], fit["amp_err"][i],
                        fit["sig"][i], fit["sig_err"][i]))
    lines.append("")
    if fit.get("ped_fitted"):
        lines.append("pedestal    P  = %.2f ± %.2f ADC"
                     % (fit["ped"], fit["ped_err"]))
    else:
        lines.append("pedestal    P  = %.2f ADC   (fixed)" % fit["ped"])
    lines.append("spacing     M  = %.2f ± %.2f ADC"
                 % (fit["gain"], fit["gain_err"]))
    lines.append("")
    if fit.get("nobkg"):
        lines.append("background  none (--nobkg)")
    else:
        lines.append("background  p0 = %.4g ± %.2g" % (b[0], be[0]))
        lines.append("            p1 = %.4g ± %.2g" % (b[1], be[1]))
        lines.append("            p2 = %.4g ± %.2g" % (b[2], be[2]))
        if fit.get("bkg_min") is not None:
            lines.append("            min   = %.4g" % fit["bkg_min"])
        if fit.get("bkg_slope_max") is not None:
            lines.append("            max d/dx = %.3g%s"
                         % (fit["bkg_slope_max"],
                            "  (<= 0)"
                            if fit.get("bkg_mode") == BKG_FALLING else ""))
    lines.append("")
    lines.append("chi2/ndf = %.1f / %d = %.2f"
                 % (fit["chi2"], fit["ndf"],
                    fit["chi2"] / fit["ndf"] if fit["ndf"] else np.nan))
    return lines


def fit_report(fit):
    """The full report, as a list of lines (printed and written to .txt)."""
    out = []
    add = out.append
    add("=" * 78)
    if not fit or not fit.get("ok"):
        add("fit        FAILED: %s" % (fit or {}).get("why", "unknown"))
        add("=" * 78)
        return out
    k = fit["npeaks"]
    i0 = fit.get("i0", 1)
    add("fit        %s%d photon peaks at P+%sM, ... P+%dM"
        % ("" if fit.get("nobkg") else "background + ", k,
           "0" if i0 == 0 else "1", i0 + k - 1))
    add("           y = %sSUM A_i exp(-0.5 ((x-(P+iM))/s_i)^2)"
        % ("" if fit.get("nobkg") else "p0 + p1 x + p2 x^2 + "))
    add("           background: %s" % BKG_LABEL.get(fit.get("bkg_mode"), "?"))
    add("           fitted over %.4g to %.4g ADC"
        % (fit.get("fit_lo", float("nan")), fit.get("fit_hi", float("nan"))))
    add("           M in (%.3g, %.3g), seeded at %s;  sigma/M in "
        "(%.3g, %.3g) -> sigma in (%.3g, %.3g) at this M"
        % (fit["n_lo"], fit["n_hi"],
           "%.4g" % fit["n_seed"] if fit.get("n_seed") else "the scan best",
           fit["r_lo"], fit["r_hi"],
           fit["r_lo"] * fit["gain"], fit["r_hi"] * fit["gain"]))
    if fit.get("ped_fitted"):
        add("           P free in (%.3g, %.3g)"
            % (fit["ped_lo"], fit["ped_hi"]))
    else:
        add("           P held fixed at %.4g" % fit["ped"])
    add("-" * 78)
    add("")
    add("     >>>>   M  =  %.3f  +-  %.3f   HG ADC per photoelectron   <<<<"
        % (fit["gain"], fit["gain_err"]))
    add("     >>>>   P  =  %.3f  +-  %.3f   HG ADC pedestal            <<<<"
        % (fit["ped"], fit["ped_err"]))
    add("")
    add("-" * 78)
    add("  %-6s %12s %16s %16s"
        % ("peak", "mean [ADC]", "amplitude", "sigma [ADC]"))
    for i in range(k):
        s = fit["sig"][i]
        flag = ""
        r = s / fit["gain"] if fit["gain"] else float("nan")
        if fit["amp"][i] < 2.0 * fit["amp_err"][i]:
            flag = "  <- amplitude consistent with zero"
        elif r <= fit["r_lo"] * 1.001:
            flag = "  <- sigma/M at the lower bound"
        elif r >= fit["r_hi"] * 0.999:
            flag = "  <- sigma/M at the upper bound"
        add("  %-6d %12.2f %8.2f +- %-5.2f %7.2f +- %-5.2f%s"
            % (i0 + i, fit["ped"] + (i0 + i) * fit["gain"], fit["amp"][i],
               fit["amp_err"][i], s, fit["sig_err"][i], flag))
    b, be = fit["bkg"], fit["bkg_err"]
    if fit.get("nobkg"):
        add("  %-6s none: the polynomial was dropped from the fit (--nobkg)"
            % "bkg")
    else:
        add("  %-6s p0 = %.4g +- %.3g   p1 = %.4g +- %.3g   p2 = %.4g +- %.3g"
            % ("bkg", b[0], be[0], b[1], be[1], b[2], be[2]))
        if fit.get("bkg_min") is not None:
            note = {BKG_FALLING: "   (held >= 0)",
                    BKG_NONNEG: "   (held >= 0)"}.get(fit.get("bkg_mode"), "")
            if not note:
                note = ("   <- NEGATIVE; drop --allow-negative-bkg"
                        if fit["bkg_min"] < 0.0
                        else "   (came out >= 0 on its own)")
            add("  %-6s lowest value over the fit range: %.4g%s"
                % ("", fit["bkg_min"], note))
        if fit.get("bkg_slope_max") is not None:
            if fit.get("bkg_mode") == BKG_FALLING:
                note = "   (held <= 0: the background never rises)"
            elif fit["bkg_slope_max"] > 0.0:
                note = "   <- RISES somewhere; this is the default's job"
            else:
                note = "   (came out falling on its own)"
            add("  %-6s steepest slope over the fit range: %.4g%s"
                % ("", fit["bkg_slope_max"], note))
    add("-" * 78)
    red = fit["chi2"] / fit["ndf"] if fit["ndf"] else float("nan")
    add("  chi2 / ndf = %.2f / %d = %.3f" % (fit["chi2"], fit["ndf"], red))
    if fit.get("scan"):
        add("  other comb spacings the seed scan found, best first:")
        add("         " + "   ".join("M=%.2f" % g for _c, g in fit["scan"]))
        add("         pick one with --n-init, or fence M in with "
            "--n-min/--n-max")
    empty = [i0 + i for i in range(k)
             if fit["amp"][i] < 2.0 * fit["amp_err"][i]]
    if empty:
        add("  WARNING  peak(s) %s have no amplitude.  A comb with empty slots "
            "is the" % ", ".join(str(i) for i in empty))
        add("           signature of M landing on a sub-multiple of the true "
            "spacing:")
        add("           if peak 1 is the empty one, the real M is probably "
            "%.2f." % (2.0 * fit["gain"]))
        add("           Reseed with --n-init, or raise --fit-xmin.")
    if fit.get("ped_fitted"):
        for edge, name in ((fit["ped_lo"], "lower"), (fit["ped_hi"], "upper")):
            if abs(fit["ped"] - edge) < 1e-3 * max(1.0, abs(edge)):
                add("  WARNING  P is sitting on its %s bound (%.4g).  Move it "
                    "with --ped-min/--ped-max." % (name, edge))
    if red > 3.0:
        add("  NOTE   chi2/ndf is large.  Try raising --fit-xmin further, or "
            "--npeaks,")
        add("         or check that the peaks the fit chose are the ones you "
            "see.")
    add("=" * 78)
    return out


def print_fit(fit):
    for line in fit_report(fit):
        print(line)


# ----------------------------------------------------------------------------
# the 50 active SiPMs in one panel
# ----------------------------------------------------------------------------
def _extent(fig, ax, artist):
    """Axes-fraction bbox of an artist, including its bbox patch if it has one.

    A Text drawn with va='top' puts its TEXT top at y, but the rounded box
    around it extends further up by the box padding -- which is how a block
    placed just under the legend still lands on top of it.  Measuring the patch
    is what makes the stacking below honest.
    """
    fig.canvas.draw()
    patch = getattr(artist, "get_bbox_patch", lambda: None)()
    win = (patch if patch is not None else artist).get_window_extent()
    return win.transformed(ax.transAxes.inverted())


def _bottom_of(fig, ax, artist, pad=0.0):
    """Axes-fraction y of the bottom of an already-drawn artist, minus pad."""
    return _extent(fig, ax, artist).y0 - pad


def _place_under(fig, ax, ytop, make):
    """Draw a text with make(y) and slide it down until nothing sticks above.

    Returns the artist, so the next thing can be stacked under its true bottom.
    """
    t = make(ytop)
    over = _extent(fig, ax, t).y1 - ytop
    if over > 1e-4:
        t.set_y(ytop - over)
    return t


def make_all50(vals, out_paths, title, subtitle, lo, hi, nbins,
               logy=False, column="HG", show_individual=False,
               dofit=True, npeaks=NPEAKS, sigma_init=SIGMA_INIT,
               sigma_lo=SIGMA_LO, sigma_hi=SIGMA_HI,
               n_init=N_INIT, n_lo=None, n_hi=None,
               fit_lo=FIT_LO, fit_hi=None, bkg_mode=BKG_FALLING,
               sig_frac_lo=SIG_FRAC_LO, sig_frac_hi=SIG_FRAC_HI,
               ped_peak=False, ped_init=None, ped_lo=None, ped_hi=None,
               fix_ped=None, ratio=True):
    """The mean spectrum of the active SiPMs, the comb fit, and data/fit.

    All 50 channels have the same number of entries -- one per event -- so the
    pooled spectrum divided by 50 is the average channel.  With
    show_individual the 50 channels are drawn faintly behind it, on the same
    scale; the y axis then covers them too.

    The mean spectrum is fitted with a falling background plus npeaks Gaussians
    at P + M, P + 2M, ... P + npeaks*M, with the pedestal P and the spacing M
    both free.  The fit, its background, the individual peaks, P and the
    parameters are drawn on top, and a data/fit ratio panel is added underneath
    over the fitted range.  Returns the usual numbers alongside the fit dict.
    """
    edges = np.linspace(lo, hi, nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # ---- accumulate first: whether there is a ratio panel depends on the fit
    pooled = np.zeros(nbins, dtype=float)
    per_ch = []
    nch = 0
    ntot = 0
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
        per_ch.append((col, h))
        if col not in seen_cols:
            seen_cols.append(col)

    if nch == 0:
        sys.exit("no active channels had any entries")

    mean = pooled / float(nch)
    # the mean of nch histograms of Poisson counts
    err = np.sqrt(np.maximum(pooled, 1.0)) / float(nch)

    # the envelope of everything that gets drawn, for the head-room step below
    env = mean.copy()
    if show_individual:
        for _c, _h in per_ch:
            env = np.maximum(env, _h)

    # ---- the fit -----------------------------------------------------------
    fit = None
    flo = lo if fit_lo is None else max(lo, float(fit_lo))
    fhi = hi if fit_hi is None else min(hi, float(fit_hi))
    if dofit:
        sel = (centres >= flo) & (centres <= fhi)
        ngauss = npeaks + 1 if ped_peak else npeaks
        i0 = 0 if ped_peak else 1
        npar = 2 * ngauss + 6 + (0 if fix_ped is not None else 1)
        if sel.sum() < npar:
            fit = {"ok": False, "why": "only %d bins in the fit range %g-%g"
                                       % (int(sel.sum()), flo, fhi)}
        else:
            fit = fit_photon_peaks(centres[sel], mean[sel], err[sel],
                                   npeaks=ngauss, sigma_init=sigma_init,
                                   sigma_lo=sigma_lo, sigma_hi=sigma_hi,
                                   n_init=n_init, n_lo=n_lo, n_hi=n_hi,
                                   bkg_mode=bkg_mode,
                                   sig_frac_lo=sig_frac_lo,
                                   sig_frac_hi=sig_frac_hi, i0=i0,
                                   ped_init=ped_init, ped_lo=ped_lo,
                                   ped_hi=ped_hi, fix_ped=fix_ped)
            if fit.get("ok"):
                fit["fit_lo"], fit["fit_hi"] = float(flo), float(fhi)

    ok = bool(fit and fit.get("ok"))
    show_ratio = bool(ratio and ok)

    # ---- the figure: one panel, or spectrum over data/fit -------------------
    if show_ratio:
        fig = plt.figure(figsize=(11.0, 8.6))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.6, 1.0], hspace=0.07,
                              left=0.085, right=0.98, top=0.885, bottom=0.082)
        ax = fig.add_subplot(gs[0, 0])
        axr = fig.add_subplot(gs[1, 0], sharex=ax)
        ax.tick_params(labelbottom=False)
        sup_y, sub_y = 0.978, 0.950
    else:
        fig, ax = plt.subplots(figsize=(11.0, 7.0))
        fig.subplots_adjust(left=0.085, right=0.98, top=0.855, bottom=0.095)
        axr = None
        sup_y, sub_y = 0.975, 0.935

    ymax = 0.0
    if show_individual:
        for col, h in per_ch:
            ax.step(centres, h, where="mid", color=COLCOLOUR[col],
                    lw=LINEWIDTH, solid_joinstyle="miter", alpha=0.30,
                    zorder=2)
            ymax = max(ymax, float(h.max()) if h.size else 0.0)

    ymax = max(ymax, float(mean.max()))
    ax.fill_between(centres, mean, step="mid", color="0.2", alpha=0.10,
                    zorder=3)
    ax.step(centres, mean, where="mid", color="0.1", lw=1.6,
            solid_joinstyle="miter", zorder=4,
            label="mean of the %d SiPMs  (%d entries / %d)" % (nch, ntot, nch))

    if ok:
        gain = fit["gain"]
        ped = fit["ped"]
        i0 = fit.get("i0", 1)
        means = [ped + (i0 + i) * gain for i in range(fit["npeaks"])]
        xf = np.linspace(flo, fhi, 1000)
        if flo > lo:
            ax.axvspan(lo, flo, color="0.55", alpha=0.10, zorder=0)
            ax.annotate("not fitted", xy=(0.5 * (lo + flo), 0.985),
                        xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=7.5, color="0.45")
        if fhi < hi:
            ax.axvspan(fhi, hi, color="0.55", alpha=0.10, zorder=0)
        bkg = fit["p"][0] + fit["p"][1] * xf + fit["p"][2] * xf * xf

        # each peak, sitting on the background it was fitted over
        for i, mu in enumerate(means):
            g = fit["amp"][i] * np.exp(-0.5 * ((xf - mu) / fit["sig"][i]) ** 2)
            ax.plot(xf, bkg + g, color="#2a78d6", lw=0.9, ls="--", zorder=5,
                    label="single peaks" if i == 0 else None)
            ax.axvline(mu, color="#2a78d6", lw=0.7, ls=":", alpha=0.55,
                       zorder=1)
            # at the foot of the line, so the labels never fight the legend
            ax.annotate("%d p.e." % (i0 + i), xy=(mu, 0.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8,
                        color="#2a78d6", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                  edgecolor="none", alpha=0.85))

        # ---- the pedestal, where the comb starts from ----
        if lo <= ped <= hi:
            ax.axvline(ped, color="#117733", lw=1.2, ls="--", alpha=0.9,
                       zorder=6)
            ax.annotate("P = %.2f" % ped, xy=(ped, 0.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(3, 24), textcoords="offset points",
                        ha="left", va="bottom", fontsize=8.5,
                        color="#117733", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.22",
                                  facecolor="white", edgecolor="#117733",
                                  lw=0.8, alpha=0.92))

        if not fit.get("nobkg"):
            ax.plot(xf, bkg, color="0.45", lw=1.1, ls="-.", zorder=5,
                    label="background (%s)"
                          % BKG_LABEL.get(fit.get("bkg_mode"), "?"))
        ax.plot(xf, peak_model(xf, fit["p"], fit["npeaks"], ped, i0),
                color="#cc3311", lw=2.0, zorder=6,
                label=("fit: %d peaks" if fit.get("nobkg")
                       else "fit: background + %d peaks") % fit["npeaks"])
        ymax = max(ymax, float(np.max(peak_model(
            xf, fit["p"], fit["npeaks"], ped, i0))))

    if show_individual:
        handles = [plt.Line2D([], [], color=COLCOLOUR[c], lw=1.2, alpha=0.6,
                              label="column %s" % c) for c in COLS
                   if c in seen_cols]
        handles.append(plt.Line2D([], [], color="0.1", lw=1.6,
                                  label="mean of all %d" % nch))
        if ok:
            handles.append(plt.Line2D([], [], color="#cc3311", lw=2.0,
                                      label="fit: background + %d peaks"
                                            % fit["npeaks"]))
            if not fit.get("nobkg"):
                handles.append(plt.Line2D([], [], color="0.45", lw=1.1,
                                          ls="-.", label="background (%s)"
                                          % BKG_LABEL.get(
                                              fit.get("bkg_mode"), "?")))
            handles.append(plt.Line2D([], [], color="#2a78d6", lw=0.9,
                                      ls="--", label="single peaks"))
            handles.append(plt.Line2D([], [], color="#117733", lw=1.2,
                                      ls="--", label="pedestal P"))
        leg = ax.legend(handles=handles, fontsize=8.0, frameon=False,
                        loc="upper right", labelspacing=0.3)
    else:
        leg = ax.legend(fontsize=8.0, frameon=False, loc="upper right",
                        labelspacing=0.3)

    ax.set_ylabel("events / bin, per SiPM", fontsize=11)
    ax.set_xlim(lo, hi)
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, YHEADROOM * ymax if ymax > 0 else 1)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=9)
    if axr is None:
        ax.set_xlabel("%s ADC counts" % column, fontsize=11)

    # ---- the parameter box top right; the legend, M and P just left of it --
    # the stats box is placed and measured FIRST, and everything else is then
    # right-aligned to its left edge, so the two blocks sit side by side in the
    # middle of the axis instead of one landing on the other
    guards = []
    if ok:
        block = "\n".join(fit_lines(fit))
        for fs in (8.0, 7.5, 7.0, 6.5, 6.0, 5.5):
            t = _place_under(fig, ax, 0.982, lambda yy, f=fs: ax.text(
                0.985, yy, block, transform=ax.transAxes, ha="right",
                va="top", fontsize=f, family="monospace", color="0.15",
                zorder=10, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor="0.75", lw=0.8, alpha=0.94)))
            if _bottom_of(fig, ax, t) > 0.015 or fs == 5.5:
                break
            t.remove()
        stats = _extent(fig, ax, t)
        guards.append(stats)

        # right edge of everything else: hard against the stats box, no overlap
        xr = min(max(stats.x0 - 0.022, 0.32), 0.97)
        leg.set_bbox_to_anchor((xr, 0.995), transform=ax.transAxes)
        guards.append(_extent(fig, ax, leg))

        y = _bottom_of(fig, ax, leg, 0.030)
        t = _place_under(fig, ax, y, lambda yy: ax.text(
            xr, yy, "M = %.2f $\\pm$ %.2f" % (fit["gain"], fit["gain_err"]),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=19, fontweight="bold", color="#cc3311", zorder=10,
            bbox=dict(boxstyle="round,pad=0.38", facecolor="white",
                      edgecolor="#cc3311", lw=1.6, alpha=0.96)))
        y = _bottom_of(fig, ax, t, 0.010)
        t = _place_under(fig, ax, y, lambda yy: ax.text(
            xr - 0.005, yy, "%s ADC counts per photoelectron" % column,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, color="#cc3311", zorder=10))
        y = _bottom_of(fig, ax, t, 0.014)
        t = _place_under(fig, ax, y, lambda yy: ax.text(
            xr - 0.005, yy,
            ("P = %.2f $\\pm$ %.2f ADC   (pedestal)"
             % (fit["ped"], fit["ped_err"])) if fit.get("ped_fitted")
            else "P = %.2f ADC   (pedestal, fixed)" % fit["ped"],
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            fontweight="bold", color="#117733", zorder=10))
        guards.append(_extent(fig, ax, t))
    elif fit:
        guards.append(_extent(fig, ax, leg))
        t = _place_under(fig, ax, _bottom_of(fig, ax, leg, 0.030), lambda yy:
                         ax.text(0.985, yy, "fit failed\n%s"
                                 % fit.get("why", ""),
                                 transform=ax.transAxes, ha="right", va="top",
                                 fontsize=10, color="#cc3311", zorder=10,
                                 bbox=dict(boxstyle="round,pad=0.4",
                                           facecolor="white",
                                           edgecolor="#cc3311", lw=1.2)))
        guards.append(_extent(fig, ax, t))
    else:
        guards.append(_extent(fig, ax, leg))

    # ---- head room: lift the y axis until the histogram clears the text ----
    # the boxes are placed in axes fractions, so this cannot move them; it only
    # opens space underneath, which beats letting the spectrum run into them
    if not logy:
        top = YHEADROOM * ymax if ymax > 0 else 1.0
        for bb in guards:
            xlo = lo + max(bb.x0, 0.0) * (hi - lo)
            xhi = lo + min(bb.x1, 1.0) * (hi - lo)
            m = (centres >= xlo) & (centres <= xhi)
            if not m.any() or bb.y0 <= 0.10:
                continue
            need = float(env[m].max()) / (bb.y0 - 0.025)
            top = max(top, min(need, 1.5 * ymax if ymax > 0 else need))
        ax.set_ylim(0, top)

    # ---- data / fit --------------------------------------------------------
    if show_ratio:
        sel = (centres >= flo) & (centres <= fhi)
        mod = peak_model(centres[sel], fit["p"], fit["npeaks"], fit["ped"],
                         fit.get("i0", 1))
        good = mod > 0
        xr = centres[sel][good]
        rr = mean[sel][good] / mod[good]
        re_ = err[sel][good] / mod[good]

        if flo > lo:
            axr.axvspan(lo, flo, color="0.55", alpha=0.10, zorder=0)
        if fhi < hi:
            axr.axvspan(fhi, hi, color="0.55", alpha=0.10, zorder=0)
        for mu in means:
            axr.axvline(mu, color="#2a78d6", lw=0.7, ls=":", alpha=0.55,
                        zorder=1)
        axr.axhline(1.0, color="#cc3311", lw=1.5, zorder=3)
        axr.errorbar(xr, rr, yerr=re_, fmt="o", ms=2.8, lw=0.0, elinewidth=0.8,
                     capsize=0, color="0.12", ecolor="0.45", zorder=4)

        # a window that shows the structure without one outlier setting it
        spread = float(np.percentile(np.abs(rr - 1.0), 98)) if rr.size else 0.1
        half = min(max(1.35 * spread, 0.03), 0.60)
        axr.set_ylim(1.0 - half, 1.0 + half)
        for lev in (1.0 - 0.5 * half, 1.0 + 0.5 * half):
            axr.axhline(lev, color="0.7", lw=0.6, ls=":", zorder=2)
        axr.set_ylabel("data / fit", fontsize=10)
        axr.set_xlabel("%s ADC counts" % column, fontsize=11)
        axr.set_xlim(lo, hi)
        axr.grid(alpha=0.25, lw=0.5)
        axr.tick_params(labelsize=9)
        nout = int(np.sum(np.abs(rr - 1.0) > half))
        if nout:
            axr.annotate("%d point%s off scale" % (nout, "" if nout == 1
                                                   else "s"),
                         xy=(0.995, 0.06), xycoords="axes fraction",
                         ha="right", va="bottom", fontsize=7.5, color="0.45")

    fig.suptitle(title, fontsize=14, y=sup_y)
    fig.text(0.5, sub_y, subtitle, ha="center", va="top", fontsize=9,
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
    p.add_argument("listfile", help="Janus list text file.  A name like "
                                    "Run132_133_137_list.txt is read as "
                                    "several runs")
    p.add_argument("-o", "--out", default=None,
                   help="output png (default: RunX_photonPeaks.png next to "
                        "the input file, keeping every run number in the name)")
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
                   help="number of photon peaks in the fit, at P+M, P+2M, ... "
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
                   help="starting value for the spacing M (default %g; pass 0 "
                        "to seed it by scanning M over its allowed range "
                        "instead)" % N_INIT)
    p.add_argument("--fit-xmin", type=float, default=FIT_LO,
                   help="low edge of the FIT range, independent of the plot "
                        "range (default %g, which keeps the pedestal tail out "
                        "of the fit)" % FIT_LO)
    p.add_argument("--fit-xmax", type=float, default=None,
                   help="high edge of the fit range (default: --xmax)")
    p.add_argument("--n-min", type=float, default=None,
                   help="lower bound on M (default: 2 bin widths)")
    p.add_argument("--n-max", type=float, default=None,
                   help="upper bound on M (default: (xmax - P) / npeaks, so "
                        "the last peak stays inside the range)")
    p.add_argument("--ped-init", type=float, default=None,
                   help="starting value for the pedestal P (default: the "
                        "middle of its allowed window)")
    p.add_argument("--ped-min", type=float, default=None,
                   help="lower bound on P (default 0)")
    p.add_argument("--ped-max", type=float, default=None,
                   help="upper bound on P (default: the low edge of the fit "
                        "range -- the pedestal sits below where the fit "
                        "starts)")
    p.add_argument("--fix-ped", type=float, default=None,
                   help="hold P at this value instead of fitting it, e.g. a "
                        "pedestal measured from a random-trigger run")
    p.add_argument("--nobkg", action="store_true",
                   help="drop the polynomial background: the model becomes the "
                        "comb alone.  The three coefficients are removed from "
                        "the fit, not just set to zero, so they cost no "
                        "degrees of freedom")
    p.add_argument("--ped-peak", action="store_true",
                   help="include the 0-avalanche (pedestal) peak in the comb, "
                        "so the means run P, P+M, P+2M, ...  In a dark run the "
                        "pedestal peak is the tallest, narrowest feature, so "
                        "it anchors the comb and M comes out as the true "
                        "SPACING.  Use --fit-xmin 2 with it")
    p.add_argument("--sigma-max-frac", type=float, default=SIG_FRAC_HI,
                   help="widest peak allowed, as a fraction of M (default "
                        "%g).  A peak wider than about half the spacing is "
                        "not a peak, it is the fit using a Gaussian as "
                        "background" % SIG_FRAC_HI)
    p.add_argument("--sigma-min-frac", type=float, default=SIG_FRAC_LO,
                   help="narrowest peak allowed, as a fraction of M "
                        "(default %g)" % SIG_FRAC_LO)
    p.add_argument("--allow-rising-bkg", action="store_true",
                   help="let the background rise.  By default it is held "
                        "non-negative AND non-increasing across the fit range; "
                        "this relaxes it to non-negative only")
    p.add_argument("--allow-negative-bkg", action="store_true",
                   help="let the polynomial background go negative as well as "
                        "rise: a completely unconstrained quadratic")
    p.add_argument("--no-ratio", action="store_true",
                   help="do not add the data/fit ratio panel under the "
                        "all-50 spectrum")
    p.add_argument("--no-pdf", action="store_true",
                   help="only write the pngs, not the matching pdfs")
    p.add_argument("--no-txt", action="store_true",
                   help="do not write the fit report as a text file")
    args = p.parse_args(argv)

    if not os.path.exists(args.listfile):
        sys.exit("no such list file: %s" % args.listfile)

    if args.allow_negative_bkg:
        bkg_mode = BKG_FREE
    elif args.allow_rising_bkg:
        bkg_mode = BKG_NONNEG
    else:
        bkg_mode = BKG_FALLING
    if args.nobkg:
        bkg_mode = BKG_NONE

    header, vals, nused, nseen = read_hg(
        args.listfile, skip_events=args.skip_events,
        max_events=args.nevents, column=args.column)

    if nused == 0:
        sys.exit("no events read from %s (%d seen, %d skipped)"
                 % (args.listfile, nseen, args.skip_events))

    stem_tag = run_stem(args.listfile)          # Run132_133_137
    label = run_label(args.listfile)            # Runs 132, 133, 137
    runs = run_numbers(args.listfile)
    out_png = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.listfile)),
        "%s_photonPeaks.png" % stem_tag)

    active_names = [CHMAP[ch][0] for ch in ACTIVE_CH]

    # ---- what was read ----
    print("=" * 78)
    print("file        %s" % args.listfile)
    print("runs        %s" % (", ".join(runs) if runs else "(none parsed)"))
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
             "%s   photon peaks" % label,
             base + "  |  A black, B red, C blue, D green" + stamp)]

    stem, ext = os.path.splitext(out_png)
    if not args.no_per_column:
        for col in COLS:
            jobs.append(([col], "%s_%s%s" % (stem, col, ext),
                         "%s   photon peaks   column %s" % (label, col),
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
            % (label, len(ACTIVE_CH)),
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
            fit_lo=args.fit_xmin, fit_hi=args.fit_xmax,
            bkg_mode=bkg_mode,
            sig_frac_lo=args.sigma_min_frac,
            sig_frac_hi=args.sigma_max_frac, ped_peak=args.ped_peak,
            ped_init=args.ped_init, ped_lo=args.ped_min,
            ped_hi=args.ped_max, fix_ped=args.fix_ped,
            ratio=not args.no_ratio)
        print("all50      %d SiPMs, %d entries pooled (%d per SiPM)"
              % (nch, nent, nent // nch if nch else 0))
        if not args.logy:
            print("           highest bin %.1f -> y axis top %.1f"
                  % (top50, YHEADROOM * top50))
        written += paths
        if fit is not None:
            report = fit_report(fit)
            for line in report:
                print(line)
            if not args.no_txt:
                txt = "%s_fit.txt" % stem
                with open(txt, "w") as f:
                    f.write("file   %s\n" % os.path.abspath(args.listfile))
                    f.write("runs   %s\n"
                            % (", ".join(runs) if runs else "(none parsed)"))
                    f.write("events %d used of %d seen (first %d skipped)\n"
                            % (nused, nseen, args.skip_events))
                    if "Run start time" in header:
                        f.write("start  %s\n" % header["Run start time"])
                    f.write("\n")
                    f.write("\n".join(report))
                    f.write("\n")
                written.append(txt)

    print("-" * 78)
    for path in written:
        print("wrote %s" % path)


if __name__ == "__main__":
    main()
