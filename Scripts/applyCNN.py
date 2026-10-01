#!/usr/bin/env python3
"""
applyCNN.py

Run a trained muon-direction network over a formatted DATA file and write out
the predicted (theta, phi) per event, plus a one-page figure with the theta
distribution, the phi distribution and a circular theta-vs-phi radiograph.

    python applyCNN.py Run152_list_formatted_1000photons_innerRing3.txt \
        --modelType FullSC_wphi100

        -> Run152_list_formatted_1000photons_innerRing3_FullSC_wphi100_CNN.txt
           Run152_list_formatted_1000photons_innerRing3_FullSC_wphi100_CNN.png
           (+ .pdf, and _CNN_displays/ if per-event output is asked for)

--modelType is one of

    FullSC (default)   FullSC_wphi100
    Simple
    SimpleWrap         SimpleWrap_eqdeg
    UnitSC             UnitSC_wphi100

and is also the label in every output filename, so runs of different models
on the same file never overwrite each other.  --model-dir overrides it with
an explicit path; the label is then built from that run directory's name
(ResNet2_<variant>_..._<extra tags>_..., keeping the variant and the extra
tags such as wphi100 or eqdeg).

Input
-----
The output of convertDataFile.py: one event is a TrgID on its own line
followed by the 4 x 11 photon counts.  A file written with --placeholder
(TrgID, then a 4 x 11 block of -999999, then the counts) is also accepted --
the layout is detected from the file, not assumed.

Model
-----
MODEL_BASE / MODEL_TYPES below, chosen with --modelType, or --model-dir.
Only two files are needed from it:

    model_scripted.pt   the network (TorchScript: no class definitions, no
                        training script)
    model_config.json   img_mean / img_std, theta_mean / theta_std, and the
                        phi decoding constants for the head in question

The head is DETERMINED, not assumed.  Before any data is touched the network
is run once on a blank 4 x 11 image to measure its output width, and that is
combined with the config to pick the decoder:

    width 3, (theta_norm, s, c)       sincos   (FullSC models)
        phi = atan2(s, c),  conf = |(s, c)|, a real per-event confidence
    width 3, (theta_norm, s, c)       unitSC   (model.variant says unitSC)
        phi = atan2(s, c),  conf = |(s, c)|, but the loss pins it near 1, so
        it is a health check on the event rather than a confidence
    width 2, (theta_norm, phi_rad)    simplePhi  (Simple models; no phi_std)
        phi = degrees(out[1]),  no confidence
    width 2, (theta_norm, phi_norm)   simpleWrap (postprocessing has phi_std)
        phi = out[1] * phi_std + phi_mean -- a RESCALE in degrees, NOT
        radians,  no confidence

The two width-2 heads cannot be told apart by width, which is why the config
decides between them.  If model.variant in the config names a head that
disagrees with the measured width, a warning is printed and the width wins.

theta is the same on every head: theta_norm * theta_std + theta_mean.

On a head with no confidence, everything that used it degrades cleanly:
--min-conf is refused, the scatter radiograph draws plain dots, the phi_conf
column of the text output is zeros (with a header line saying so) and the
confidence summary is not printed.

Units
-----
The network was trained on the MC file whose name the model directory carries,
i.e. on PHOTON COUNTS on the same 4 x 11 grid, and convertDataFile.py produces
photon counts, so the two match.  What does NOT automatically match is the
DISTRIBUTION: img_mean / img_std were measured on MC, and the data's own mean
and spread are printed next to them at startup.  A large disagreement there is
domain shift and the predictions should not be trusted until it is understood.

The ring ordering is the other thing to check: convertDataFile.py writes ring
15 in column 0, and the mean column profile of the data is printed so it can
be compared with the MC.  A profile that looks mirrored means theta will come
out backwards.

Output text file
----------------
    event  trgid  theta_deg  phi_deg  phi_conf  total_photons  train_cut
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", FutureWarning)

# ----------------------------------------------------------------------------
#MODEL_DIR = ("../ScintillatorAI/plots/sipm_hits_PoC_4p85Mevents_runs1-485_CNN_1000photons_innerRing3/ResNet2_sincos_nTiles2_nSiPM2_mV10_5epochs/model")
#MODEL_DIR = ("/Users/muonix/ScintillatorAI/IllinoisComputes/plots2/sipm_hits_PoC_14p85Mevents_runs1-485_1001-2000_CNN_1000photons_innerRing3/ResNet2_unitSC_nTiles2_nSiPM2_mV10_20epochs/model")
#MODEL_DIR = ("/Users/muonix/ScintillatorAI/plots/sipm_hits_PoC_14p85Mevents_runs1-485_1001-2000_CNN_1000photons_innerRing3/ResNet2_simplePhi_nTiles2_nSiPM2_mV10_5epochs_Take2/model")
#MODEL_DIR = ("/Users/muonix/ScintillatorAI/IllinoisComputes/plots3/sipm_hits_PoC_14p85Mevents_runs1-485_1001-2000_CNN_1000photons_innerRing3/ResNet2_simpleWrap_nTiles2_nSiPM2_mV10_tiles11_phi135-225_20epochs/model")
#MODEL_DIR = ("/Users/muonix/ScintillatorAI/IllinoisComputes/plots4/sipm_hits_PoC_14p85Mevents_runs1-485_1001-2000_CNN_1000photons_innerRing3/ResNet2_simpleWrap_nTiles2_nSiPM2_mV10_tiles11_phi0-360_5epochs/model")
#MODEL_DIR = ("/Users/muonix/ScintillatorAI/IllinoisComputes/plots5/sipm_hits_PoC_14p85Mevents_runs1-485_1001-2000_CNN_1000photons_innerRing3/ResNet2_simpleWrap_nTiles2_nSiPM2_mV10_tiles11_phi0-360_eqdeg_20epochs/model")
#MODEL_DIR = ("/Users/muonix/ScintillatorAI/IllinoisComputes/plots6/sipm_hits_PoC_14p85Mevents_runs1-485_1001-2000_CNN_1000photons_innerRing3/ResNet2_simpleWrap_nTiles2_nSiPM2_mV10_tiles11_phi0-360_20epochs/model")

# --modelType picks one of these run directories under MODEL_BASE; the model
# itself is in <run>/model.  The key is also the label that goes into every
# output filename, so outputs from different models never collide.
# --model-dir still overrides all of this with an explicit path.
MODEL_BASE = ("/Users/muonix/ScintillatorAI/IllinoisComputes/"
              "sipm_hits_PoC_10Mevents_CNN_v2p4_1000photons_innerRing3")
MODEL_TYPES = {
    "FullSC":          "ResNet2_FullSC_nTiles2_nSiPM2_mV10_tiles11_50epochs_2026-09-30_19h01m",
    "FullSC_wphi100":  "ResNet2_FullSC_nTiles2_nSiPM2_mV10_tiles11_wphi100_50epochs_2026-09-30_19h02m",
    "Simple":          "ResNet2_Simple_nTiles2_nSiPM2_mV10_tiles11_50epochs_2026-09-30_19h01m",
    "SimpleWrap":      "ResNet2_SimpleWrap_nTiles2_nSiPM2_mV10_tiles11_50epochs_2026-09-30_19h01m",
    "SimpleWrap_eqdeg": "ResNet2_SimpleWrap_nTiles2_nSiPM2_mV10_tiles11_eqdeg_50epochs_2026-09-30_19h02m",
    "UnitSC":          "ResNet2_UnitSC_nTiles2_nSiPM2_mV10_tiles11_50epochs_2026-09-30_19h01m",
    "UnitSC_wphi100":  "ResNet2_UnitSC_nTiles2_nSiPM2_mV10_tiles11_wphi100_50epochs_2026-09-30_19h02m",
}
DEFAULT_MODEL_TYPE = "FullSC"


# ----------------------------------------------------------------------------

NROW = 4                # detector columns A, B, C, D
NRING = 11              # rings 15 (output column 0) down to 5
COLS = ["A", "B", "C", "D"]
RING_HI = 15
RING_LO = RING_HI - NRING + 1       # 5

# --showEventDisplays: drawing one png per event gets out of hand fast, so
# there is a cap.  Raise it with --max-displays.
MAX_DISPLAYS = 50

CMAP = "inferno"
SCATTER_BELOW = 1500    # fewer events than this: draw points, not a density

# The theta axis is FIXED, on both the 1D spectrum and the radiograph, so
# plots from different runs can be laid side by side.  --theta-max overrides.
# Events above it are not plotted; how many were clipped is always reported.
THETA_MAX_DEG = 80.0

# --showTarget: the region the muons are being aimed at.  theta between
# TARGET_THETA, phi within TARGET_PHI of straight ahead.
TARGET_THETA = (40.0, 60.0)
TARGET_PHI = (-8.5, 8.5)
TARGET_FILL = "#a8d3f0"     # light blue
TARGET_EDGE = "#3f88c5"
DOT_COLOR = "#cc3311"       # the one red the dots take under --showTarget

# names a head goes by (in model.variant or the run directory), lower case,
# -> the canonical head.  Matched as WHOLE tokens, so "simple" does not
# swallow "simplewrap".
VARIANT_ALIASES = {"sincos": "sincos", "fullsc": "sincos",
                   "unitsc": "unitsc",
                   "simplephi": "simplephi", "simple": "simplephi",
                   "simplewrap": "simplewrap"}
# canonical head -> the output width it implies
VARIANT_WIDTH = {"sincos": 3, "unitsc": 3, "simplephi": 2, "simplewrap": 2}

# run-directory tokens that are the same for every model and say nothing
# about which one it is; everything else after "ResNet2" goes in the label
_COMMON_TOKEN = r"^(nTiles\d+|nSiPM\d+|mV\d+|tiles\d+|\d+epochs|" \
                r"\d{4}-\d{2}-\d{2}|\d+h\d+m)$"


def model_label(model_dir):
    """Short, unique label for a model, from its run directory name.

    ResNet2_FullSC_nTiles2_nSiPM2_mV10_tiles11_wphi100_50epochs_<date>_<time>
        -> FullSC_wphi100
    """
    import re
    run = os.path.basename(os.path.dirname(os.path.abspath(model_dir)))
    toks = run.split("_")
    if toks and toks[0].lower().startswith("resnet"):
        toks = toks[1:]
    keep = [t for t in toks if t and not re.match(_COMMON_TOKEN, t)]
    return "_".join(keep) or run


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
def load_model(model_dir):
    import torch
    cfg_path = os.path.join(model_dir, "model_config.json")
    ts_path = os.path.join(model_dir, "model_scripted.pt")
    if not os.path.isdir(model_dir):
        sys.exit("no such model directory: %s\n"
                 "Point --model-dir at the 'model' folder written next to the "
                 "training plots." % model_dir)
    if not os.path.isfile(cfg_path):
        sys.exit("no model_config.json in %s" % model_dir)
    if not os.path.isfile(ts_path):
        sys.exit("no model_scripted.pt in %s" % model_dir)
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = torch.jit.load(ts_path, map_location="cpu")
    model.eval()
    return model, cfg


def _variant_from(cfg, model_dir):
    """The head the config claims, as one of VARIANT_WIDTH's keys, or ''.

    model.variant is the authority.  If it is missing, the run directory name
    (ResNet2_sincos_..., ResNet2_SimpleWrap_...) is used as a fallback hint,
    and the result says which it came from.
    """
    import re
    v = str(cfg.get("model", {}).get("variant", "")).lower()
    for tok in re.split(r"[^a-z0-9]+", v):
        if tok in VARIANT_ALIASES:
            return VARIANT_ALIASES[tok], "model.variant"
    run = os.path.basename(os.path.dirname(os.path.abspath(model_dir))).lower()
    for tok in run.split("_"):
        if tok in VARIANT_ALIASES:
            return VARIANT_ALIASES[tok], "run directory name"
    return "", ""


def detect_head(model, cfg, model_dir):
    """Work out which output layout this model has, before any data is run.

    The width is MEASURED by running the network once on a blank image; the
    config then picks between heads of the same width.  Returns a dict:

        name       sincos | unitSC | simplePhi | simpleWrap
        nout       measured output width
        has_conf   whether |(s,c)| exists for this head
        decode     the formula, for the log and the output header
        conf_note  what phi_conf means on this head
    """
    import torch
    post = cfg["postprocessing"]
    with torch.no_grad():
        o = model(torch.zeros((1, 1, NROW, NRING), dtype=torch.float32))
    o = o.cpu().numpy()
    if o.ndim != 2 or o.shape[0] != 1:
        sys.exit("unexpected model output shape %s for a single image; "
                 "expected (1, 2) or (1, 3)" % (tuple(o.shape),))
    nout = int(o.shape[1])

    claimed, source = _variant_from(cfg, model_dir)

    if nout == 3:
        if claimed == "unitsc" or "unit" in str(
                post.get("confidence_formula", "")).lower():
            head = dict(
                name="unitSC", has_conf=True,
                decode="phi = atan2(s, c)",
                conf_note="|(s,c)|; the unitSC loss anchors it near 1, so "
                          "it is a health check on the event, not a "
                          "confidence")
        else:
            head = dict(
                name="sincos", has_conf=True,
                decode="phi = atan2(s, c)",
                conf_note="|(s,c)|, the network's confidence in phi "
                          "(1 = sure, 0 = azimuthally ambiguous)")
    elif nout == 2:
        if "phi_std" in post or claimed == "simplewrap":
            if "phi_std" not in post or "phi_mean" not in post:
                sys.exit("this is a simpleWrap model (%s) but its "
                         "postprocessing has no phi_mean/phi_std -- cannot "
                         "decode phi.  Fix model_config.json."
                         % (source or "phi_std present"))
            head = dict(
                name="simpleWrap", has_conf=False,
                decode="phi = out[1] * phi_std + phi_mean  (standardised "
                       "degrees, NOT radians)",
                conf_note="none on this head; phi_conf is written as 0")
        else:
            head = dict(
                name="simplePhi", has_conf=False,
                decode="phi = degrees(out[1])",
                conf_note="none on this head; phi_conf is written as 0")
    else:
        sys.exit("this model has %d outputs; expected 3 (theta_norm, s, c) "
                 "or 2 (theta_norm, phi_rad | phi_norm)" % nout)

    head["nout"] = nout
    head["claimed"] = claimed
    head["claimed_from"] = source
    head["mismatch"] = bool(claimed) and VARIANT_WIDTH[claimed] != nout
    return head


def predict(model, cfg, head, images, batch_size=256):
    """-> (theta_deg, phi_deg, conf).  conf is zeros on a no-confidence head,
    so every caller keeps the same shape -- but nothing should interpret it."""
    import torch
    pre, post = cfg["preprocessing"], cfg["postprocessing"]
    x = (images - pre["img_mean"]) / pre["img_std"]
    outs = []
    with torch.no_grad():
        for k in range(0, x.shape[0], batch_size):
            t = torch.from_numpy(x[k:k + batch_size][:, None, :, :]).float()
            outs.append(model(t).cpu().numpy())
    o = np.concatenate(outs, axis=0)
    if o.shape[1] != head["nout"]:
        sys.exit("model returned %d outputs on data but %d on the probe"
                 % (o.shape[1], head["nout"]))

    theta = o[:, 0] * post["theta_std"] + post["theta_mean"]

    name = head["name"]
    if name in ("sincos", "unitSC"):
        s, c = o[:, 1], o[:, 2]
        phi = np.degrees(np.arctan2(s, c))
        conf = np.hypot(s, c)
    elif name == "simpleWrap":
        phi = o[:, 1] * post["phi_std"] + post["phi_mean"]
        conf = np.zeros_like(phi)
    else:                                   # simplePhi
        phi = np.degrees(o[:, 1])
        conf = np.zeros_like(phi)

    # phi_lo is only present on the heads that define one; default [0, 360)
    phi_lo = float(post.get("phi_lo", 0.0))
    phi = (phi - phi_lo) % 360.0 + phi_lo
    return theta, phi, conf


# ----------------------------------------------------------------------------
# the formatted data file
# ----------------------------------------------------------------------------
def read_formatted(path, nrow=NROW, nring=NRING):
    """Read convertDataFile.py output.

    Returns (trgid (N,), images (N, nrow, nring) float32, nblocks).  Works
    whether or not the file carries the --placeholder block: the number of
    lines per event is measured from the file.
    """
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    if not lines:
        sys.exit("%s has no data lines" % path)

    # a line with a single token starts an event; the gap to the next one is
    # the block size
    starts = [i for i, ln in enumerate(lines) if len(ln.split()) == 1]
    if not starts or starts[0] != 0:
        sys.exit("%s does not start with a TrgID line; is it the output of "
                 "convertDataFile.py?" % path)
    per_event = starts[1] - starts[0] if len(starts) > 1 else len(lines)
    nblocks = (per_event - 1) // nrow
    if per_event != 1 + nblocks * nrow or nblocks not in (1, 2):
        sys.exit("%s has %d lines per event; expected %d (TrgID + %d x %d) "
                 "or %d (with the --placeholder block)"
                 % (path, per_event, 1 + nrow, nrow, nring, 1 + 2 * nrow))
    if len(lines) % per_event:
        sys.exit("%s has %d data lines, not a multiple of %d"
                 % (path, len(lines), per_event))

    # with a placeholder block the counts are the SECOND block
    skip = (nblocks - 1) * nrow

    trgid, images = [], []
    for i in range(0, len(lines), per_event):
        trgid.append(int(float(lines[i])))
        rows = []
        for r in range(nrow):
            v = lines[i + 1 + skip + r].split()
            if len(v) != nring:
                sys.exit("line %d: %d values, expected %d"
                         % (i + 1 + skip + r + 1, len(v), nring))
            rows.append([float(x) for x in v])
        images.append(rows)

    return (np.asarray(trgid, dtype=np.int64),
            np.asarray(images, dtype=np.float32),
            nblocks)


def training_cut(images, cfg):
    """The event cut the network was trained behind, as a 0/1 flag."""
    sel = cfg.get("selection", {})
    thr = float(sel.get("volt_threshold_V", 0.0))
    nsipm = int(sel.get("min_sipms_per_tile", 0))
    ntile = int(sel.get("min_tiles_per_event", 0))
    above = images >= thr                       # (N, 4, 11)
    per_tile = above.sum(axis=1)                # (N, 11)
    return ((per_tile >= nsipm).sum(axis=1) >= ntile).astype(int)


# ----------------------------------------------------------------------------
# --showEventDisplays: one detector image per surviving event
# ----------------------------------------------------------------------------
def _text_colour(rgba):
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "black" if lum > 0.55 else "white"


def image_to_grid(img):
    """(4, 11) as stored -> (11, 4) as the detector is drawn.

    convertDataFile.py writes rows = detector columns A,B,C,D and output
    column 0 = ring 15, so getting back to a picture of the detector means
    transposing and flipping: ring 15 ends up on the TOP row, ring 5 on the
    bottom, columns A B C D left to right -- the same orientation
    makeEventDisplays.py uses.
    """
    return np.asarray(img, dtype=float).T[::-1, :]


def write_event_text(img, out_txt, trgid, theta, phi, conf):
    """One event as a small text file: a header line, then the 4 x 11 array.

        TrigID  theta  phi  phi_confidence
        <4 rows of 11 photon counts>

    The array is written in the SAME orientation convertDataFile.py used --
    rows are detector columns A, B, C, D and array column 0 is ring 15 -- so
    it can be read back with the same code that reads the input file.  Note
    that is the transpose of the picture draw_photon_event() makes.

    No comment lines: the header is exactly the four numbers, so the file
    parses with a bare loadtxt/split.  On a no-confidence head the fourth
    number is always 0 -- kept so the format does not change between models.
    """
    a = np.asarray(img, dtype=float)
    with open(out_txt, "w") as f:
        f.write("%d %.4f %.4f %.4f\n" % (int(trgid), theta, phi, conf))
        for row in a:
            f.write(" ".join("%d" % int(round(v)) for v in row) + "\n")


def draw_photon_event(img, out_png, title, subtitle, vmin=0.0, vmax=None,
                      cmap_name=CMAP, logscale=False, mark_peak=True):
    """One event's 4 x 11 photon image, in makeEventDisplays.py's style.

    These are the PHOTON COUNTS the network was handed, not raw ADC, and the
    CosmicWatch channels are not in them: convertDataFile.py drops ch0-5 and
    rings 0-4 before writing.  So there are no CW strips above and below the
    grid and no saturation flags -- by this point the LG substitution has
    already happened upstream.
    """
    grid = image_to_grid(img)
    if vmax is None:
        vmax = float(np.nanmax(grid)) if np.isfinite(grid).any() else 1.0
    vmax = max(float(vmax), vmin + 1.0)

    fig = plt.figure(figsize=(6.6, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[24, 1], wspace=0.05,
                          left=0.11, right=0.87, top=0.855, bottom=0.055)
    ax = fig.add_subplot(gs[0, 0])
    ax_cb = fig.add_subplot(gs[0, 1])

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#e8e8e8")
    if logscale:
        norm = matplotlib.colors.LogNorm(vmin=max(vmin, 1.0), vmax=vmax)
        shown = np.clip(grid, max(vmin, 1.0), None)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        shown = grid

    im = ax.imshow(np.ma.masked_invalid(shown), origin="lower",
                   extent=[0, len(COLS), 0, NRING], aspect="auto",
                   cmap=cmap, norm=norm, interpolation="nearest")

    peak = np.unravel_index(int(np.nanargmax(grid)), grid.shape) \
        if np.isfinite(grid).any() else None
    for r in range(NRING):
        for c in range(len(COLS)):
            v = grid[r, c]
            if not np.isfinite(v):
                ax.text(c + 0.5, r + 0.5, "--", ha="center", va="center",
                        fontsize=6, color="#8a8a8a")
                continue
            ax.text(c + 0.5, r + 0.5, "%.0f" % v, ha="center", va="center",
                    fontsize=6.5,
                    color=_text_colour(cmap(norm(max(v, norm.vmin)))))
            if mark_peak and peak is not None and (r, c) == peak:
                # the cell the colour scale is pinned to, when vmax is the
                # per-event maximum
                ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, lw=1.6,
                                           ec="#00e5ff", zorder=3))

    ax.set_xticks(np.arange(len(COLS)) + 0.5)
    ax.set_xticklabels(COLS, fontsize=9)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_yticks(np.arange(NRING) + 0.5)
    ax.set_yticklabels([str(RING_LO + i) for i in range(NRING)], fontsize=7)
    ax.set_ylabel("detector ring", fontsize=9)
    ax.set_xticks(np.arange(len(COLS) + 1), minor=True)
    ax.set_yticks(np.arange(NRING + 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=2)

    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("photons", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    fig.suptitle(title, fontsize=12, y=0.975)
    fig.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=8,
             color="0.25", linespacing=1.5)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# the figure
# ----------------------------------------------------------------------------
def circ_stats(phi_deg):
    """Circular mean (deg) and resultant length R in [0, 1]."""
    r = np.radians(np.asarray(phi_deg, dtype=float))
    s, c = np.sin(r).mean(), np.cos(r).mean()
    return np.degrees(np.arctan2(s, c)) % 360.0, float(np.hypot(s, c))


def in_target(theta, phi, target_theta=TARGET_THETA, target_phi=TARGET_PHI):
    """Mask of events inside the --showTarget region.

    theta between the two limits, and phi within the given window of its
    centre -- done as a wrapped difference so a window straddling 0/360
    (which -8.5 .. +8.5 does) works without special-casing.
    """
    tlo, thi = sorted(float(v) for v in target_theta)
    plo, phi_hi = sorted(float(v) for v in target_phi)
    pmid = 0.5 * (plo + phi_hi)
    phalf = 0.5 * (phi_hi - plo)
    dphi = (np.asarray(phi, dtype=float) - pmid + 180.0) % 360.0 - 180.0
    t = np.asarray(theta, dtype=float)
    return (t >= tlo) & (t <= thi) & (np.abs(dphi) <= phalf)


def draw_target(ax, target_theta=TARGET_THETA, target_phi=TARGET_PHI):
    """The shaded target wedge on the polar axes.

    zorder 2: above the density (1) and below the dots (3), so the muons stay
    readable on top of the region.
    """
    tlo, thi = sorted(float(v) for v in target_theta)
    plo, phi_hi = sorted(float(v) for v in target_phi)
    ang = np.radians(np.linspace(plo, phi_hi, 200))
    ax.fill_between(ang, tlo, thi, color=TARGET_FILL, alpha=0.75,
                    linewidth=0, zorder=2)
    for r in (tlo, thi):
        ax.plot(ang, np.full_like(ang, r), color=TARGET_EDGE, lw=1.0,
                alpha=0.9, zorder=2.1)
    for a in (plo, phi_hi):
        ax.plot([np.radians(a)] * 2, [tlo, thi], color=TARGET_EDGE, lw=1.0,
                alpha=0.9, zorder=2.1)


def cos_power_curve(x_deg, tmax, n_in, binw_deg, power=2.0):
    """Expected muons per bin for a flux isotropic in phi and ~cos^p(theta).

    The flux per unit SOLID ANGLE goes as cos^p(theta), but a zenith-angle
    histogram counts muons per unit THETA, and the ring of directions at
    theta is 2 pi sin(theta) wide, so

        dN/dtheta  ~  cos^p(theta) sin(theta)

    Normalised so the curve's area equals the n_in events inside [0, tmax]:

        integral of cos^p sin dtheta from 0 to T  =  (1 - cos^(p+1) T)/(p+1)
    """
    t = np.radians(np.asarray(x_deg, dtype=float))
    T = np.radians(float(tmax))
    norm = (1.0 - np.cos(T) ** (power + 1.0)) / (power + 1.0)
    if norm <= 0:
        return np.zeros_like(t)
    pdf_per_deg = (np.cos(t) ** power) * np.sin(t) * (np.pi / 180.0) / norm
    return n_in * binw_deg * pdf_per_deg


def make_page(theta, phi, conf, out_paths, title, subtitle,
              nth=45, nph=36, style="auto", solid_angle=False,
              theta_max=None, radio_dphi=10.0, radio_dtheta=5.0,
              cos_power=2.0, show_cos=True, show_target=False,
              target_theta=TARGET_THETA, target_phi=TARGET_PHI,
              has_conf=True):
    """One page: theta spectrum, phi spectrum, and the circular radiograph.

    has_conf=False drops the confidence colouring from the scatter
    radiograph -- colouring every point by a column of zeros would just be a
    lie with a colourbar attached.
    """
    n = theta.size
    # fixed range, so runs can be compared directly
    tmax = float(theta_max) if theta_max else THETA_MAX_DEG
    # round up to a whole number of radiograph theta bins, so the rim lands on
    # a bin edge and the radial ticks come out on round numbers
    tmax = float(radio_dtheta * np.ceil(tmax / radio_dtheta))
    n_over = int(np.sum(theta > tmax))

    fig = plt.figure(figsize=(12.0, 14.5))

    ax_t = fig.add_axes([0.075, 0.735, 0.385, 0.165])
    ax_p = fig.add_axes([0.565, 0.735, 0.385, 0.165])
    ax_r = fig.add_axes([0.150, 0.045, 0.640, 0.565], projection="polar")
    ax_cb = fig.add_axes([0.855, 0.120, 0.022, 0.380])

    # ---- 1. theta ----
    ax_t.hist(theta, bins=np.linspace(0, tmax, nth + 1), histtype="stepfilled",
              color="#2a78d6", alpha=0.25)
    ax_t.hist(theta, bins=np.linspace(0, tmax, nth + 1), histtype="step",
              color="#1b4f8f", lw=1.8)
    if show_cos:
        binw = tmax / float(nth)
        n_in = int(np.sum((theta >= 0) & (theta <= tmax)))
        xc = np.linspace(0, tmax, 400)
        yc = cos_power_curve(xc, tmax, n_in, binw, cos_power)
        ax_t.plot(xc, yc, color="0.35", ls="--", lw=1.0, zorder=4)
        lab = (r"$\cos^{%g}\theta\,\sin\theta$" % cos_power) \
            if cos_power != 2.0 else r"$\cos^{2}\theta\,\sin\theta$"
        k = int(0.72 * xc.size)
        ax_t.annotate(lab, xy=(xc[k], yc[k]), xytext=(4, 7),
                      textcoords="offset points", ha="left", va="bottom",
                      fontsize=8.5, color="0.35")

    ax_t.set_xlabel(r"$\theta$ [degrees]", fontsize=11)
    ax_t.set_ylabel("muons / bin", fontsize=11)
    ax_t.set_title(r"zenith angle $\theta$", fontsize=12)
    ax_t.grid(alpha=0.25, lw=0.5)
    ax_t.set_xlim(0, tmax)
    if n_over:
        ax_t.annotate("%d event%s above %.0f$^\\circ$ not shown"
                      % (n_over, "" if n_over == 1 else "s", tmax),
                      xy=(0.985, 0.965), xycoords="axes fraction",
                      ha="right", va="top", fontsize=7.5, color="#cc3311",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                edgecolor="none", alpha=0.85))

    # ---- 2. phi ----
    # same theta cut as the theta panel and the radiograph: only muons with
    # theta <= tmax, so all three panels show the same events
    phi_in = phi[theta <= tmax]
    n_in_phi = phi_in.size
    ax_p.hist(phi_in, bins=np.linspace(0, 360, nph + 1),
              histtype="stepfilled", color="#cc3311", alpha=0.22)
    ax_p.hist(phi_in, bins=np.linspace(0, 360, nph + 1), histtype="step",
              color="#8f2410", lw=1.8)
    if n_over:
        ax_p.annotate("$\\theta$ $\\leq$ %.0f$^\\circ$ only  (%d event%s "
                      "excluded)" % (tmax, n_over, "" if n_over == 1 else "s"),
                      xy=(0.985, 0.965), xycoords="axes fraction",
                      ha="right", va="top", fontsize=7.5, color="#cc3311",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                edgecolor="none", alpha=0.85))
    flat = n_in_phi / float(nph)
    ax_p.axhline(flat, color="0.35", ls="--", lw=1.0)
    ax_p.annotate("flat = %.1f" % flat, xy=(0, flat),
                  xytext=(4, 3), textcoords="offset points",
                  ha="left", va="bottom", fontsize=7.5, color="0.35")
    ax_p.set_xlabel(r"$\phi$ [degrees]", fontsize=11)
    ax_p.set_ylabel("muons / bin", fontsize=11)
    ax_p.set_title(r"azimuth $\phi$", fontsize=12)
    ax_p.grid(alpha=0.25, lw=0.5)
    ax_p.set_xlim(0, 360)
    ax_p.set_xticks([0, 90, 180, 270, 360])

    # ---- 3. the radiograph ----
    use_scatter = (style == "scatter") or (style == "auto"
                                           and n < SCATTER_BELOW)
    ax_r.set_theta_zero_location("E")
    ax_r.set_theta_direction(1)          # phi as atan2 gives it: CCW from +x
    ax_r.set_rlim(0, tmax)
    ax_r.set_rlabel_position(112.5)
    rstep = 10.0 if tmax > 50 else 5.0
    ax_r.set_rticks(np.arange(rstep, tmax - 1e-9, rstep))
    ax_r.grid(alpha=0.35, lw=0.6, color="0.5")

    if show_target:
        draw_target(ax_r, target_theta, target_phi)

    if use_scatter and show_target:
        ax_r.scatter(np.radians(phi), theta, s=14, color=DOT_COLOR,
                     alpha=0.85, edgecolors="none", zorder=3)
        ax_cb.set_visible(False)
        tlo, thi = sorted(float(v) for v in target_theta)
        plo, phi_hi = sorted(float(v) for v in target_phi)
        note = ("one point per muon;  shaded: %g$^\\circ$ < $\\theta$ < "
                "%g$^\\circ$, %g$^\\circ$ < $\\phi$ < %g$^\\circ$"
                % (tlo, thi, plo, phi_hi))
    elif use_scatter and has_conf:
        sz = np.clip(6.0 + 40.0 * (conf / max(conf.max(), 1e-9)), 4, 55)
        sc = ax_r.scatter(np.radians(phi), theta, c=conf, s=sz, cmap=CMAP,
                          alpha=0.85, edgecolors="none", zorder=3)
        cb = fig.colorbar(sc, cax=ax_cb)
        cb.set_label(r"$\phi$ confidence  $|(s,c)|$", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        note = ("one point per muon, size and colour = the network's "
                "confidence in $\\phi$")
    elif use_scatter:
        ax_r.scatter(np.radians(phi), theta, s=14, color=DOT_COLOR,
                     alpha=0.85, edgecolors="none", zorder=3)
        ax_cb.set_visible(False)
        note = ("one point per muon  (single-$\\phi$-output model: no "
                "per-event confidence)")
    else:
        nph_r = int(round(360.0 / radio_dphi))
        nth_r = int(round(tmax / radio_dtheta))
        pe = np.linspace(0, 2 * np.pi, nph_r + 1)
        te = np.linspace(0, tmax, nth_r + 1)
        H, _, _ = np.histogram2d(np.radians(phi) % (2 * np.pi), theta,
                                 bins=[pe, te])
        lab = "muons / bin"
        if solid_angle:
            dom = (np.cos(np.radians(te[:-1])) - np.cos(np.radians(te[1:])))
            H = H / (np.diff(pe)[:, None] * dom[None, :])
            lab = "muons / steradian"
        P, T = np.meshgrid(pe, te, indexing="ij")
        pc = ax_r.pcolormesh(P, T, H, cmap=CMAP, shading="auto", zorder=1)
        cb = fig.colorbar(pc, cax=ax_cb)
        cb.set_label(lab, fontsize=9)
        note = ("density in %g$^\\circ$ $\\phi$ x %g$^\\circ$ "
                "$\\theta$ bins  (%d x %d)%s"
                % (radio_dphi, radio_dtheta, nph_r, nth_r,
                   ", divided by bin solid angle" if solid_angle else ""))
        if show_target:
            note += "\n" + ("shaded: %g$^\\circ$ < $\\theta$ < %g$^\\circ$, "
                            "%g$^\\circ$ < $\\phi$ < %g$^\\circ$"
                            % (min(target_theta), max(target_theta),
                               min(target_phi), max(target_phi)))
        cb.ax.tick_params(labelsize=8)

    fig.text(0.47, 0.672, r"$\theta$ vs $\phi$ radiograph", ha="center",
             va="top", fontsize=13)
    fig.text(0.47, 0.648,
             r"radius = $\theta$ (0$^\circ$ = vertical, "
             "%.0f$^\\circ$ at the rim),  angle = $\\phi$" % tmax
             + "\n" + note, ha="center", va="top", fontsize=9.5,
             color="0.25", linespacing=1.5)
    import matplotlib.patheffects as pe_fx
    rc, sc_ = ("white", "black") if not use_scatter else ("0.15", "white")
    plt.setp(ax_r.get_yticklabels(), color=rc, fontsize=9.5,
             fontweight="bold",
             path_effects=[pe_fx.withStroke(linewidth=1.8, foreground=sc_)])

    ax_r.set_xticks(np.radians(np.arange(0, 360, 45)))
    ax_r.set_xticklabels([r"$\phi$=0$^\circ$"] +
                         ["%d$^\\circ$" % d for d in range(45, 360, 45)],
                         fontsize=9)

    fig.suptitle(title, fontsize=15, y=0.975)
    fig.text(0.5, 0.950, subtitle, ha="center", va="top", fontsize=9,
             color="0.25", linespacing=1.5)
    for p in out_paths:
        fig.savefig(p, dpi=150)
    plt.close(fig)
    return tmax, use_scatter


# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datafile", help="convertDataFile.py output")
    p.add_argument("--modelType", default=DEFAULT_MODEL_TYPE,
                   choices=list(MODEL_TYPES),
                   help="which trained model to apply (default %s).  Picks "
                        "the run directory under MODEL_BASE, and is the "
                        "label put into every output filename"
                        % DEFAULT_MODEL_TYPE)
    p.add_argument("--model-dir", default=None,
                   help="explicit directory holding model_scripted.pt and "
                        "model_config.json; overrides --modelType, and the "
                        "label is then taken from its run directory name")
    p.add_argument("-o", "--out", default=None,
                   help="output text file (default: INPUT_<label>_CNN.txt)")
    p.add_argument("--min-conf", type=float, default=None,
                   help="drop events whose phi confidence |(s,c)| is below "
                        "this FROM THE PLOTS (every event is still written to "
                        "the text file, with its confidence).  Needs a "
                        "3-output sincos/unitSC model; refused on a 2-output "
                        "one")
    p.add_argument("--theta-bins", type=int, default=45,
                   help="bins in the 1D theta histogram (default 45)")
    p.add_argument("--phi-bins", type=int, default=36,
                   help="bins in the 1D phi histogram (default 36, i.e. 10 "
                        "degrees)")
    p.add_argument("--radio-dphi", type=float, default=10.0, metavar="DEG",
                   help="radiograph phi bin width in degrees (default 10)")
    p.add_argument("--radio-dtheta", type=float, default=5.0, metavar="DEG",
                   help="radiograph theta bin width in degrees (default 5).  "
                        "The outer radius is rounded up to a whole number of "
                        "these.")
    p.add_argument("--cos-power", type=float, default=2.0, metavar="P",
                   help="exponent of the cos^P(theta) reference curve drawn "
                        "on the theta panel (default 2, the sea-level "
                        "approximation)")
    p.add_argument("--no-cos-curve", action="store_true",
                   help="do not draw the cos^P(theta) sin(theta) reference "
                        "curve")
    p.add_argument("--theta-max", type=float, default=None, metavar="DEG",
                   help="outer radius of the radiograph, top of the 1D theta "
                        "axis, AND the theta cut on the 1D phi histogram "
                        "(default %g, fixed so runs can be compared; events "
                        "above it are not plotted in any panel and the number "
                        "clipped is reported)" % THETA_MAX_DEG)
    p.add_argument("--showTarget", action="store_true",
                   help="draw the target region on the radiograph as a light "
                        "blue wedge and make every muon the same red dot on "
                        "top of it.  The wedge runs %g-%g deg in theta and "
                        "%g to %g deg in phi; the fraction of muons inside it "
                        "is reported"
                        % (TARGET_THETA[0], TARGET_THETA[1],
                           TARGET_PHI[0], TARGET_PHI[1]))
    p.add_argument("--target-theta", type=float, nargs=2, metavar=("LO", "HI"),
                   default=list(TARGET_THETA),
                   help="theta limits of the --showTarget wedge (default "
                        "%g %g)" % TARGET_THETA)
    p.add_argument("--target-phi", type=float, nargs=2, metavar=("LO", "HI"),
                   default=list(TARGET_PHI),
                   help="phi limits of the --showTarget wedge, may straddle 0 "
                        "(default %g %g)" % TARGET_PHI)
    p.add_argument("--style", choices=["auto", "density", "scatter"],
                   default="auto",
                   help="radiograph style; auto draws points below %d events "
                        "and a density above (default auto)" % SCATTER_BELOW)
    p.add_argument("--solid-angle", action="store_true",
                   help="divide the radiograph density by the solid angle of "
                        "each bin, so the colour is a flux rather than a "
                        "count")
    p.add_argument("--showEventDisplays", action="store_true",
                   help="also write one detector image per event that passes "
                        "every selection -- whatever survived into the input "
                        "file AND the confidence cut here.  They go in "
                        "INPUT_<label>_CNN_displays/, in the style of "
                        "makeEventDisplays.py, with the CNN's own answer for "
                        "that event in the subtitle")
    p.add_argument("--printTextFiles", action="store_true",
                   help="also write one text file per selected event -- the "
                        "same events --showEventDisplays draws, in the same "
                        "directory.  Each holds a header line 'TrigID theta "
                        "phi phi_confidence' followed by the 4 x %d photon "
                        "array.  Works with or without --showEventDisplays"
                        % NRING)
    p.add_argument("--max-displays", type=int, default=MAX_DISPLAYS,
                   metavar="N",
                   help="stop after this many events for BOTH "
                        "--showEventDisplays and --printTextFiles, so the two "
                        "sets match one for one (default %d; 0 means every "
                        "passing event)" % MAX_DISPLAYS)
    p.add_argument("--display-dir", default=None,
                   help="where the event displays and text files go (default: "
                        "INPUT_<label>_CNN_displays next to the input file)")
    p.add_argument("--display-zmax", type=float, default=None, metavar="N",
                   help="fix the top of the event-display colour scale in "
                        "photons.  DEFAULT is per event: the largest SiPM "
                        "value in that event, outlined in cyan")
    p.add_argument("--display-log", action="store_true",
                   help="logarithmic colour scale on the event displays")
    p.add_argument("--no-pdf", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args(argv)

    src = args.datafile
    if not os.path.isfile(src):
        sys.exit("no such file: %s" % src)

    # ---- which model, its label, and which head it has ----
    if args.model_dir:
        label = model_label(args.model_dir)
    else:
        label = args.modelType
        args.model_dir = os.path.join(MODEL_BASE, MODEL_TYPES[label], "model")
    model, cfg = load_model(args.model_dir)
    head = detect_head(model, cfg, args.model_dir)
    has_conf = head["has_conf"]
    prov = cfg.get("provenance", {})
    perf = cfg.get("performance_on_test_set", {})

    stem = src[:-4] if src.lower().endswith(".txt") else src
    out_base = "%s_%s_CNN" % (stem, label)
    out_txt = args.out or (out_base + ".txt")
    out_png = out_base + ".png"
    out_pdf = out_base + ".pdf"

    print("=" * 78)
    print("model      %s   (label '%s' in the output names)"
          % (args.model_dir, label))
    print("           trained %s from %s"
          % (prov.get("trained_at", "?"),
             os.path.basename(prov.get("training_file", "?"))))
    print("           %s, %d parameters, %d epochs"
          % (cfg["model"]["class"], cfg["model"]["n_parameters"],
             cfg["training"]["epochs"]))
    if perf:
        print("           MC test resolution: sigma_theta %.2f deg, "
              "sigma_phi %.2f deg"
              % (perf.get("theta_sigma_gauss_deg") or float("nan"),
                 perf.get("phi_sigma_gauss_deg") or float("nan")))
    loss = cfg.get("training", {}).get("loss")
    if loss:
        print("           loss: %s" % loss)
    print("  head     %s  (%d outputs, measured)" % (head["name"],
                                                     head["nout"]))
    print("           %s" % head["decode"])
    print("           phi_conf: %s" % head["conf_note"])
    if head["claimed"]:
        print("           config says '%s' (from %s)"
              % (head["claimed"], head["claimed_from"]))
    else:
        print("           config names no head; decided from the output width "
              "and postprocessing keys")
    if head["mismatch"]:
        print("  WARNING  the config names a %d-output head but the network "
              "returns %d."
              % (VARIANT_WIDTH[head["claimed"]], head["nout"]))
        print("           Decoding by the measured width.  Check that "
              "model_config.json and")
        print("           model_scripted.pt came from the same training run.")
    print("-" * 78)

    trgid, images, nblocks = read_formatted(src)
    n = images.shape[0]
    print("input      %s" % src)
    print("           %d events, %d x %d%s" % (n, NROW, NRING,
          "  (--placeholder block present, skipped)" if nblocks == 2 else ""))

    fmt = cfg["format"]
    if fmt["n_rows"] != NROW or fmt["num_tiles"] != NRING:
        sys.exit("the model expects %d x %d, this file is %d x %d"
                 % (fmt["n_rows"], fmt["num_tiles"], NROW, NRING))

    # ---- domain check: is the data anything like what the model saw? ----
    pre = cfg["preprocessing"]
    d_mean, d_std = float(images.mean()), float(images.std())
    print("-" * 78)
    print("  units    photon counts on both sides (the model was trained on "
          "the MC file named above)")
    print("  %-10s %12s %12s" % ("", "MC (training)", "this data"))
    print("  %-10s %12.5f %12.5f" % ("mean", pre["img_mean"], d_mean))
    print("  %-10s %12.5f %12.5f" % ("std", pre["img_std"], d_std))
    off = abs(d_mean - pre["img_mean"]) / max(pre["img_std"], 1e-9)
    rat = d_std / max(pre["img_std"], 1e-9)
    print("  -> the data mean sits %.2f MC sigma from the MC mean, and its "
          "spread is %.2fx" % (off, rat))
    if off > 1.0 or not 0.5 < rat < 2.0:
        print("  WARNING  that is a large domain shift.  The network is being "
              "asked about")
        print("           images unlike anything it was trained on; treat "
              "the angles below")
        print("           as indicative at best.")
    prof = images.sum(axis=(0, 1)) / max(n, 1)
    print("  mean column profile (column 0 = detector ring %d):" % RING_HI)
    print("     col  " + " ".join("%7d" % c for c in range(NRING)))
    print("     ring " + " ".join("%7d" % (RING_HI - c) for c in range(NRING)))
    print("     phot " + " ".join("%7.1f" % v for v in prof))
    print("           compare this with the MC profile: if it looks mirrored, "
          "the ring")
    print("           ordering disagrees and theta will come out backwards.")

    # A model trained inside a restricted phi window cannot predict outside it.
    win = cfg.get("training", {}).get("phi_window_deg")
    if win:
        print("-" * 78)
        print("  WARNING  this model was trained ONLY on phi in [%g, %g] deg."
              % (win[0], win[1]))
        print("           It has never seen a muon outside that window and "
              "cannot predict one.")
        print("           The phi histogram and radiograph below are shaped "
              "by the training")
        print("           window, NOT by the data.  Do not use this model to "
              "study azimuthal")
        print("           structure, and do not compare its phi to the "
              "full-circle models.")
    print("-" * 78)

    if args.min_conf is not None and not has_conf:
        sys.exit("--min-conf needs a 3-output sincos/unitSC model.  This one "
                 "is %s, with a single phi output and no per-event "
                 "confidence, so there is nothing to cut on." % head["name"])
    if args.min_conf is not None and head["name"] == "unitSC":
        print("  note     --min-conf on a unitSC model cuts on a health "
              "check, not a confidence:")
        print("           |(s,c)| is pinned near 1 by the loss.")
        print("-" * 78)

    theta, phi, conf = predict(model, cfg, head, images, args.batch_size)
    tcut = training_cut(images, cfg)

    with open(out_txt, "w") as f:
        f.write("# CNN predictions for %s\n" % os.path.abspath(src))
        f.write("# model: %s  (%s)\n" % (os.path.abspath(args.model_dir),
                                         label))
        f.write("#   trained %s from %s\n"
                % (prov.get("trained_at", "?"),
                   os.path.basename(prov.get("training_file", "?"))))
        f.write("# head: %s (%d outputs): %s\n"
                % (head["name"], head["nout"], head["decode"]))
        if has_conf:
            f.write("# theta, phi in degrees.  phi_conf = %s.\n"
                    % head["conf_note"])
        else:
            f.write("# theta, phi in degrees.  phi_conf = 0 for every event: "
                    "this model has a single phi output and no per-event "
                    "confidence.  The column is kept so the format does not "
                    "change between models.\n")
        f.write("# train_cut = 1 if the event passes the selection the "
                "network was trained behind.\n")
        f.write("#%6s %9s %11s %11s %10s %14s %10s\n"
                % ("event", "trgid", "theta_deg", "phi_deg", "phi_conf",
                   "total_photons", "train_cut"))
        for i in range(n):
            f.write("%7d %9d %11.4f %11.4f %10.4f %14d %10d\n"
                    % (i, trgid[i], theta[i], phi[i], conf[i],
                       int(images[i].sum()), tcut[i]))

    # ---- plots ----
    m = np.ones(n, dtype=bool) if args.min_conf is None \
        else (conf >= args.min_conf)
    if not m.any():
        sys.exit("--min-conf %g leaves no events (max confidence is %.3f)"
                 % (args.min_conf, conf.max()))

    tag = os.path.basename(stem)
    sub = ("%d events%s   |   model %s (%s head)   |   theta = %.1f +- %.1f "
           "deg, phi resultant R = %.3f"
           % (int(m.sum()),
              "" if args.min_conf is None
              else " with |(s,c)| >= %g (of %d)" % (args.min_conf, n),
              label, head["name"],
              theta[m].mean(), theta[m].std(), circ_stats(phi[m])[1]))
    style = args.style
    if args.showTarget and style == "auto":
        style = "scatter"       # the dots -- a density has none

    paths = [out_png] + ([] if args.no_pdf else [out_pdf])
    tmax, scat = make_page(theta[m], phi[m], conf[m], paths,
                           "%s   -   CNN muon directions" % tag, sub,
                           nth=args.theta_bins, nph=args.phi_bins,
                           style=style, solid_angle=args.solid_angle,
                           theta_max=args.theta_max,
                           radio_dphi=args.radio_dphi,
                           radio_dtheta=args.radio_dtheta,
                           cos_power=args.cos_power,
                           show_cos=not args.no_cos_curve,
                           show_target=args.showTarget,
                           target_theta=args.target_theta,
                           target_phi=args.target_phi,
                           has_conf=has_conf)

    cmu, R = circ_stats(phi[m])
    print("predictions over %d events%s"
          % (int(m.sum()), "" if args.min_conf is None else " (after "
             "--min-conf %g)" % args.min_conf))
    print("  theta    mean %.2f   median %.2f   RMS %.2f   range %.2f - %.2f "
          "deg" % (theta[m].mean(), np.median(theta[m]), theta[m].std(),
                   theta[m].min(), theta[m].max()))
    print("  phi      circular mean %.2f deg   resultant R = %.4f "
          "(flat would give %.4f for N=%d)"
          % (cmu, R, 0.8862 / np.sqrt(m.sum()), int(m.sum())))
    if R > 3.0 * 0.8862 / np.sqrt(m.sum()):
        print("           -> that is well above the flat expectation: the "
              "sample is not")
        print("              azimuthally uniform.  Real if the detector "
              "sees an asymmetric")
        print("              overburden, but check it is not the network "
              "leaning on a prior.")
    if has_conf:
        print("  phi conf median %.3f   [5%%, 95%%] = [%.3f, %.3f]%s"
              % (np.median(conf[m]), np.percentile(conf[m], 5),
                 np.percentile(conf[m], 95),
                 "   (unitSC: health check)" if head["name"] == "unitSC"
                 else ""))
    print("  train cut %d/%d events pass the selection the model was trained "
          "behind" % (int(tcut.sum()), n))
    print("-" * 78)
    nover = int(np.sum(theta[m] > tmax))
    print("radiograph %s, outer radius theta = %.1f deg (fixed)%s"
          % ("scatter, one point per muon" if scat else "density", tmax,
             ";  %d event%s above it are not plotted"
             % (nover, "" if nover == 1 else "s") if nover else ""))
    if args.showTarget:
        tgt = in_target(theta[m], phi[m], args.target_theta, args.target_phi)
        tlo, thi = sorted(args.target_theta)
        plo, phi_hi = sorted(args.target_phi)
        # solid angle of the wedge / of the whole plotted cap, i.e. what an
        # isotropic sample would put there
        om_t = np.cos(np.radians(tlo)) - np.cos(np.radians(thi))
        om_cap = 1.0 - np.cos(np.radians(tmax))
        f_flat = (om_t / om_cap) * ((phi_hi - plo) / 360.0)
        nin = int(tgt.sum())
        expd = f_flat * int(m.sum())
        print("target     %g < theta < %g deg,  %g < phi < %g deg"
              % (tlo, thi, plo, phi_hi))
        print("           %d of %d muons inside (%.2f %%);  an isotropic "
              "sample would put %.1f there (%.2f %%)"
              % (nin, int(m.sum()), 100.0 * nin / max(int(m.sum()), 1),
                 expd, 100.0 * f_flat))
        if expd > 0:
            print("           ratio to isotropic = %.2f   (+-%.2f from "
                  "counting alone)"
                  % (nin / expd, np.sqrt(max(nin, 1)) / expd))
    print("wrote %s" % out_txt)
    for p_ in paths:
        print("wrote %s" % p_)

    # ---- per-event output: images, text files, or both ----
    if args.showEventDisplays or args.printTextFiles:
        idx = np.nonzero(m)[0]
        cap = len(idx) if args.max_displays <= 0 \
            else min(args.max_displays, len(idx))
        ddir = args.display_dir or (out_base + "_displays")
        os.makedirs(ddir, exist_ok=True)

        wants = [w for w, on in (("displays", args.showEventDisplays),
                                 ("text files", args.printTextFiles)) if on]
        print("-" * 78)
        print("per-event  %d event%s pass every selection; writing %s for %d "
              "of them"
              % (len(idx), "" if len(idx) == 1 else "s",
                 " and ".join(wants), cap))
        print("           content = the 4 x %d photon counts the network was "
              "given" % NRING)
        if args.showEventDisplays:
            print("           image   ring %d at the top, ring %d at the "
                  "bottom, columns %s"
                  % (RING_HI, RING_LO, " ".join(COLS)))
            print("           colour  %s"
                  % ("0 to %g photons (fixed)" % args.display_zmax
                     if args.display_zmax is not None
                     else "0 to the largest SiPM in each event (cyan box)"))
        if args.printTextFiles:
            print("           text    line 1 = TrigID theta phi "
                  "phi_confidence, then the 4 x %d array" % NRING)
            print("                   array as STORED (rows = columns %s, "
                  "array column 0 = ring %d),"
                  % (",".join(COLS), RING_HI))
            print("                   i.e. the same orientation as the input "
                  "file, not the picture")
            if not has_conf:
                print("                   phi_confidence is 0 throughout on "
                      "this %s model" % head["name"])
        print("           output  %s" % ddir)

        npng = ntxt = 0
        for i in idx[:cap]:
            base = "%s_evt%05d_trg%d" % (tag, i, trgid[i])
            if args.showEventDisplays:
                title = "%s   event %d   TrgID %d" % (tag, i, trgid[i])
                conf_bit = ",  $|(s,c)|$ = %.3f" % conf[i] if has_conf else ""
                tgt_bit = ""
                if args.showTarget:
                    tgt_bit = ("   -- IN the target region"
                               if in_target(theta[i], phi[i],
                                            args.target_theta,
                                            args.target_phi)
                               else "   -- outside the target region")
                sub = ("CNN (%s):  $\\theta$ = %.1f$^\\circ$,  $\\phi$ = "
                       "%.1f$^\\circ$%s\n"
                       "%d photons over %d cells%s"
                       % (label, theta[i], phi[i], conf_bit,
                          int(images[i].sum()), NROW * NRING, tgt_bit))
                draw_photon_event(images[i],
                                  os.path.join(ddir, base + ".png"),
                                  title, sub, vmax=args.display_zmax,
                                  logscale=args.display_log)
                npng += 1
            if args.printTextFiles:
                write_event_text(images[i], os.path.join(ddir, base + ".txt"),
                                 trgid[i], theta[i], phi[i], conf[i])
                ntxt += 1

        if npng:
            print("wrote %d png files to %s" % (npng, ddir))
        if ntxt:
            print("wrote %d txt files to %s" % (ntxt, ddir))
        if cap < len(idx):
            print("           %d passing events were not written; raise "
                  "--max-displays (0 = all)" % (len(idx) - cap))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
