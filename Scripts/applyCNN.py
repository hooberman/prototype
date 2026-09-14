#!/usr/bin/env python3
"""
applyCNN.py

Run the trained muon-direction network over a formatted DATA file and write
out the predicted (theta, phi) per event, plus a one-page figure with the
theta distribution, the phi distribution and a circular theta-vs-phi
radiograph.

    python applyCNN.py Run140_list_formatted_1000photons_innerRing3.txt

        -> Run140_list_formatted_1000photons_innerRing3_CNN.txt
           Run140_list_formatted_1000photons_innerRing3_CNN.png  (+ .pdf)

Input
-----
The output of convertDataFile.py: one event is a TrgID on its own line
followed by the 4 x 11 photon counts.  A file written with --placeholder
(TrgID, then a 4 x 11 block of -999999, then the counts) is also accepted --
the layout is detected from the file, not assumed.

Model
-----
MODEL_DIR below, or --model-dir.  Only two files are needed from it:

    model_scripted.pt   the network (TorchScript: no class definitions, no
                        training script)
    model_config.json   img_mean / img_std, theta_mean / theta_std, phi_lo

The network returns (theta_norm, s, c); theta comes back as
theta_norm * theta_std + theta_mean and phi as atan2(s, c), with |(s, c)| as
the per-event confidence in phi.

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
MODEL_DIR = ("/Users/benhoob/ScintillatorAI/plots/"
             "sipm_hits_PoC_v2p2_1Mevents_1-100_CNN_1000photons_innerRing3/"
             "ResNet2_sincos_nTiles2_nSiPM2_mV10_5epochs/model")
# ----------------------------------------------------------------------------

NROW = 4                # detector columns A, B, C, D
NRING = 11              # rings 15 (output column 0) down to 5
COLS = ["A", "B", "C", "D"]
RING_HI = 15

CMAP = "inferno"
SCATTER_BELOW = 1500    # fewer events than this: draw points, not a density


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


# ----------------------------------------------------------------------------
# the formatted data file
# ----------------------------------------------------------------------------
def read_formatted(path, nrow=NROW, nring=NRING):
    """Read convertDataFile.py output.

    Returns (trgid (N,), images (N, nrow, nring) float32).  Works whether or
    not the file carries the --placeholder block: the number of lines per
    event is measured from the file.
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


def predict(model, cfg, images, batch_size=256):
    import torch
    pre, post = cfg["preprocessing"], cfg["postprocessing"]
    x = (images - pre["img_mean"]) / pre["img_std"]
    outs = []
    with torch.no_grad():
        for k in range(0, x.shape[0], batch_size):
            t = torch.from_numpy(x[k:k + batch_size][:, None, :, :]).float()
            outs.append(model(t).cpu().numpy())
    o = np.concatenate(outs, axis=0)
    theta = o[:, 0] * post["theta_std"] + post["theta_mean"]
    s, c = o[:, 1], o[:, 2]
    phi = np.degrees(np.arctan2(s, c))
    phi = (phi - post["phi_lo"]) % 360.0 + post["phi_lo"]
    return theta, phi, np.hypot(s, c)


# ----------------------------------------------------------------------------
# the figure
# ----------------------------------------------------------------------------
def circ_stats(phi_deg):
    """Circular mean (deg) and resultant length R in [0, 1]."""
    r = np.radians(np.asarray(phi_deg, dtype=float))
    s, c = np.sin(r).mean(), np.cos(r).mean()
    return np.degrees(np.arctan2(s, c)) % 360.0, float(np.hypot(s, c))


def make_page(theta, phi, conf, out_paths, title, subtitle,
              nth=45, nph=36, style="auto", solid_angle=False,
              theta_max=None):
    """One page: theta spectrum, phi spectrum, and the circular radiograph."""
    n = theta.size
    tmax = theta_max if theta_max else float(
        min(90.0, max(10.0, np.percentile(theta, 99.5) * 1.05)))

    # explicit axes rather than a gridspec: the radiograph has to stay round
    # and large, and the two spectra have to clear the header
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
    ax_t.set_xlabel(r"$\theta$ [degrees]", fontsize=11)
    ax_t.set_ylabel("muons / bin", fontsize=11)
    ax_t.set_title(r"zenith angle $\theta$", fontsize=12)
    ax_t.grid(alpha=0.25, lw=0.5)
    ax_t.set_xlim(0, tmax)
    ax_t.legend([plt.Line2D([], [], color="none")],
                ["N = %d\nmean %.1f$^\\circ$\nmedian %.1f$^\\circ$\n"
                 "RMS %.1f$^\\circ$"
                 % (n, theta.mean(), np.median(theta), theta.std())],
                loc="upper right", fontsize=8.5, frameon=True, framealpha=0.9,
                handlelength=0, handletextpad=0)

    # ---- 2. phi ----
    ax_p.hist(phi, bins=np.linspace(0, 360, nph + 1), histtype="stepfilled",
              color="#cc3311", alpha=0.22)
    ax_p.hist(phi, bins=np.linspace(0, 360, nph + 1), histtype="step",
              color="#8f2410", lw=1.8)
    flat = n / float(nph)
    ax_p.axhline(flat, color="0.35", ls="--", lw=1.0)
    ax_p.annotate("flat = %.1f" % flat, xy=(0, flat),
                  xytext=(4, 3), textcoords="offset points",
                  ha="left", va="bottom", fontsize=7.5, color="0.35")
    cmu, R = circ_stats(phi)
    R_flat = 0.8862 / np.sqrt(n) if n else np.nan
    ax_p.set_xlabel(r"$\phi$ [degrees]", fontsize=11)
    ax_p.set_ylabel("muons / bin", fontsize=11)
    ax_p.set_title(r"azimuth $\phi$", fontsize=12)
    ax_p.grid(alpha=0.25, lw=0.5)
    ax_p.set_xlim(0, 360)
    ax_p.set_xticks([0, 90, 180, 270, 360])
    ax_p.legend([plt.Line2D([], [], color="none")],
                ["N = %d\ncirc. mean %.1f$^\\circ$\nR = %.3f\n"
                 "(flat would give %.3f)" % (n, cmu, R, R_flat)],
                loc="upper right", fontsize=8.5, frameon=True, framealpha=0.9,
                handlelength=0, handletextpad=0)

    # ---- 3. the radiograph ----
    use_scatter = (style == "scatter") or (style == "auto"
                                           and n < SCATTER_BELOW)
    ax_r.set_theta_zero_location("E")
    ax_r.set_theta_direction(1)          # phi as atan2 gives it: CCW from +x
    ax_r.set_rlim(0, tmax)
    ax_r.set_rlabel_position(112.5)
    ax_r.grid(alpha=0.35, lw=0.6, color="0.5")

    if use_scatter:
        sz = np.clip(6.0 + 40.0 * (conf / max(conf.max(), 1e-9)), 4, 55)
        sc = ax_r.scatter(np.radians(phi), theta, c=conf, s=sz, cmap=CMAP,
                          alpha=0.85, edgecolors="none", zorder=3)
        cb = fig.colorbar(sc, cax=ax_cb)
        cb.set_label(r"$\phi$ confidence  $|(s,c)|$", fontsize=9)
        note = ("one point per muon, size and colour = the network's "
                "confidence in $\\phi$")
    else:
        pe = np.linspace(0, 2 * np.pi, nph * 2 + 1)
        te = np.linspace(0, tmax, nth + 1)
        H, _, _ = np.histogram2d(np.radians(phi) % (2 * np.pi), theta,
                                 bins=[pe, te])
        lab = "muons / bin"
        if solid_angle:
            # bin solid angle = dphi * (cos t_lo - cos t_hi)
            dom = (np.cos(np.radians(te[:-1])) - np.cos(np.radians(te[1:])))
            H = H / (np.diff(pe)[:, None] * dom[None, :])
            lab = "muons / steradian"
        P, T = np.meshgrid(pe, te, indexing="ij")
        pc = ax_r.pcolormesh(P, T, H, cmap=CMAP, shading="auto", zorder=1)
        cb = fig.colorbar(pc, cax=ax_cb)
        cb.set_label(lab, fontsize=9)
        note = ("density in %d $\\phi$ x %d $\\theta$ bins%s"
                % (nph * 2, nth,
                   ", divided by bin solid angle" if solid_angle else ""))
    cb.ax.tick_params(labelsize=8)

    # the radiograph caption goes on the figure, not on the axes: a polar
    # axes title sits on top of the circle and would run into the spectra
    fig.text(0.47, 0.672, r"$\theta$ vs $\phi$ radiograph", ha="center",
             va="top", fontsize=13)
    fig.text(0.47, 0.648,
             r"radius = $\theta$ (0$^\circ$ = vertical, "
             "%.0f$^\\circ$ at the rim),  angle = $\\phi$" % tmax
             + "\n" + note, ha="center", va="top", fontsize=9.5,
             color="0.25", linespacing=1.5)
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
    p.add_argument("--model-dir", default=MODEL_DIR,
                   help="directory holding model_scripted.pt and "
                        "model_config.json (default: the hardcoded path)")
    p.add_argument("-o", "--out", default=None,
                   help="output text file (default: INPUT_CNN.txt)")
    p.add_argument("--min-conf", type=float, default=None,
                   help="drop events whose phi confidence |(s,c)| is below "
                        "this FROM THE PLOTS (every event is still written to "
                        "the text file, with its confidence)")
    p.add_argument("--theta-bins", type=int, default=45)
    p.add_argument("--phi-bins", type=int, default=36)
    p.add_argument("--theta-max", type=float, default=None,
                   help="outer radius of the radiograph and the top of the "
                        "theta axis (default: 99.5th percentile)")
    p.add_argument("--style", choices=["auto", "density", "scatter"],
                   default="auto",
                   help="radiograph style; auto draws points below %d events "
                        "and a density above (default auto)" % SCATTER_BELOW)
    p.add_argument("--solid-angle", action="store_true",
                   help="divide the radiograph density by the solid angle of "
                        "each bin, so the colour is a flux rather than a "
                        "count")
    p.add_argument("--no-pdf", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args(argv)

    src = args.datafile
    if not os.path.isfile(src):
        sys.exit("no such file: %s" % src)

    stem = src[:-4] if src.lower().endswith(".txt") else src
    out_txt = args.out or (stem + "_CNN.txt")
    out_png = stem + "_CNN.png"
    out_pdf = stem + "_CNN.pdf"

    model, cfg = load_model(args.model_dir)
    prov = cfg.get("provenance", {})
    perf = cfg.get("performance_on_test_set", {})

    print("=" * 78)
    print("model      %s" % args.model_dir)
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
    print("-" * 78)

    trgid, images, nblocks = read_formatted(src)
    n = images.shape[0]
    print("input      %s" % src)
    print("           %d events, %d x %d%s" % (n, NROW, NRING,
          "  (--placeholder block present, skipped)" if nblocks == 2 else ""))

    exp = cfg["format"]
    if exp["n_rows"] != NROW or exp["num_tiles"] != NRING:
        sys.exit("the model expects %d x %d, this file is %d x %d"
                 % (exp["n_rows"], exp["num_tiles"], NROW, NRING))

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
    print("-" * 78)

    theta, phi, conf = predict(model, cfg, images, args.batch_size)
    tcut = training_cut(images, cfg)

    with open(out_txt, "w") as f:
        f.write("# CNN predictions for %s\n" % os.path.abspath(src))
        f.write("# model: %s\n" % os.path.abspath(args.model_dir))
        f.write("#   trained %s from %s\n"
                % (prov.get("trained_at", "?"),
                   os.path.basename(prov.get("training_file", "?"))))
        f.write("# theta, phi in degrees.  phi_conf = |(s,c)|, the network's "
                "confidence in phi (1 = sure, 0 = azimuthally ambiguous).\n")
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
    sub = ("%d events%s   |   model %s   |   theta = %.1f +- %.1f deg, "
           "phi resultant R = %.3f"
           % (int(m.sum()),
              "" if args.min_conf is None
              else " with |(s,c)| >= %g (of %d)" % (args.min_conf, n),
              os.path.basename(os.path.dirname(args.model_dir)),
              theta[m].mean(), theta[m].std(), circ_stats(phi[m])[1]))
    paths = [out_png] + ([] if args.no_pdf else [out_pdf])
    tmax, scat = make_page(theta[m], phi[m], conf[m], paths,
                           "%s   -   CNN muon directions" % tag, sub,
                           nth=args.theta_bins, nph=args.phi_bins,
                           style=args.style, solid_angle=args.solid_angle,
                           theta_max=args.theta_max)

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
    print("  phi conf median %.3f   [5%%, 95%%] = [%.3f, %.3f]"
          % (np.median(conf[m]), np.percentile(conf[m], 5),
             np.percentile(conf[m], 95)))
    print("  train cut %d/%d events pass the selection the model was trained "
          "behind" % (int(tcut.sum()), n))
    print("-" * 78)
    print("radiograph %s, outer radius theta = %.1f deg"
          % ("scatter, one point per muon" if scat else "density", tmax))
    print("wrote %s" % out_txt)
    for p in paths:
        print("wrote %s" % p)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
