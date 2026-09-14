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
    0.  --requireTrigger (optional): at least 2 of the six CosmicWatch
        channels ch0-ch5 fired, a channel counting as fired when its RAW
        HG > 1000 OR its RAW LG > 500.  Same criterion as
        makeEventDisplays.py --requireTrigger.  Applied first, so the photon
        fractions below are then quoted among the events that fired, and
        '_requireTrigger' is added to the output file name.
    1.  total photons over the 4 x 11 array >= 1000
    2.  the column with the most photons (summed over the 4 SiPMs in it) is
        not one of the first two (0, 1) or the last two (9, 10)

Every fraction is reported, along with how many of the six paddles fired per
event whether or not the cut is on.  The first few events are printed in full so the
ring ordering and the conversion can be checked by eye.
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

# --requireTrigger: the six CosmicWatch paddles.  A channel counts as fired
# when raw HG > TRIG_HG_MIN or raw LG > TRIG_LG_MIN, and the event is kept
# when at least TRIG_NMIN of the six fired.  Same criterion as
# makeEventDisplays.py --requireTrigger.
TRIG_HG_MIN = 1000.0
TRIG_LG_MIN = 500.0
TRIG_NMIN = 2


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

TRIG_CH = tuple(range(6))        # CW top, CW bottom, CWA, CWB, CWC, CWD
TRIG_NAME = {0: "CW top", 1: "CW bottom", 2: "CWA", 3: "CWB", 4: "CWC",
             5: "CWD"}


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
               fired=None, nmin=TRIG_NMIN):
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
        print("              trigger %d/6 fired%s   %s"
              % (len(fired),
                 " [%s]" % ", ".join("ch%d %s" % (c, TRIG_NAME[c])
                                     for c in fired) if fired else "",
                 "PASS" if len(fired) >= nmin else "fail (need %d)" % nmin))
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
                   help="also require at least %d of the six CosmicWatch "
                        "channels ch0-ch5 to have fired, where fired means "
                        "raw HG > %g OR raw LG > %g -- the same criterion as "
                        "makeEventDisplays.py.  '_requireTrigger' is added to "
                        "the default output file name"
                        % (TRIG_NMIN, TRIG_HG_MIN, TRIG_LG_MIN))
    p.add_argument("--trig-hg", type=float, default=TRIG_HG_MIN,
                   help="raw HG above which a trigger channel counts as "
                        "fired (default %g)" % TRIG_HG_MIN)
    p.add_argument("--trig-lg", type=float, default=TRIG_LG_MIN,
                   help="raw LG above which a trigger channel counts as "
                        "fired (default %g)" % TRIG_LG_MIN)
    p.add_argument("--trig-nmin", type=int, default=TRIG_NMIN,
                   help="how many of the 6 trigger channels must fire "
                        "(default %d)" % TRIG_NMIN)
    p.add_argument("--placeholder", action="store_true",
                   help="also write a 4 x %d block of -999999 before the "
                        "counts, in the slot the MC files use for the "
                        "arrival-time array.  Makes the output readable by "
                        "the CNN loader as it stands (skip_leading_blocks=1)"
                        % NRING)
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
    if args.requireTrigger:
        print("trigger    --requireTrigger ON: >= %d of ch0-ch5 fired, where "
              "fired = raw HG > %g OR raw LG > %g"
              % (args.trig_nmin, args.trig_hg, args.trig_lg))
    if args.no_select:
        print("selection  NONE (--no-select)%s"
              % ("   -- the trigger cut still applies"
                 if args.requireTrigger else ""))
    else:
        print("selection  1: total photons >= %d" % args.min_photons)
        print("           2: max column not in 0..%d or %d..%d"
              % (args.edge - 1, NRING - args.edge, NRING - 1))
    print("=" * 78)

    n = nt = n1 = n12 = nsat_tot = 0
    nfired_hist = [0] * (len(TRIG_CH) + 1)
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
            nfired_hist[len(fired)] += 1
            trig_ok = (not args.requireTrigger) \
                or len(fired) >= args.trig_nmin

            img, nsat = make_image(hg, lg, args.adc_per_photon,
                                   args.lg_scale, args.hg_sat)
            nsat_tot += nsat
            res = evaluate(img, args.min_photons, args.edge)

            if n <= args.nprint:
                show_event(n - 1, trgid, img, res, nsat, args.min_photons,
                           args.edge,
                           fired if args.requireTrigger else None,
                           args.trig_nmin)

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
        print("  pass --requireTrigger (>= %d of 6)   %8d    fraction of all "
              "= %s" % (args.trig_nmin, nt, frac(nt, n)))
    print("  pass 1 (>= %d photons)            %8d    fraction of %-9s "
          "= %s" % (args.min_photons, n1,
                    "triggered" if args.requireTrigger else "all",
                    frac(n1, base)))
    print("  pass 1 and 2 (max column not edge)  %8d    fraction of pass 1 "
          "= %s" % (n12, frac(n12, n1)))
    print("  %-34s  %8d    fraction of all   = %s"
          % ("written to the output", nwritten, frac(nwritten, n)))
    print("-" * 78)
    print("  trigger channels fired per event: %s"
          % "  ".join("%d->%d" % (i, v)
                      for i, v in enumerate(nfired_hist) if v))
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
