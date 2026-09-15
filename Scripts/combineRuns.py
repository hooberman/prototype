#!/usr/bin/env python3
"""
combineRuns.py

Concatenate any number of CAEN Janus / DT5202 list files into one.

Usage
-----
    python combineRuns.py Run134_list.txt Run137_list.txt

writes

    Runs134_137_list.txt

No output name is needed: it is built from the run numbers found in the input
names, in ascending order, with the suffix of the first input ("_list.txt").
-o overrides it.  Inputs may be given in any order -- they are sorted by run
number, and "the first run" everywhere below means the lowest-numbered one.
A file that is itself a combination ("Runs134_137_list.txt") is understood and
contributes all of its run numbers.

The header
----------
The "//" header block and the "Brd Ch LG HG ..." column line are taken VERBATIM
from the first run, as required, and a short provenance block is appended to it
recording which runs went in, how many events each contributed, and the
offsets applied.  Those extra lines are ordinary "//" comments, so every reader
that already skips comments is unaffected.

Timestamps and trigger IDs
--------------------------
Each Janus run restarts TStamp_us, Tref_TStamp_us and TrgID at zero.  Simply
concatenating the files therefore produces a stream whose clock jumps backwards
at every run boundary, which quietly ruins anything computed from it -- the run
duration, the event rate, the trigger rate, and any de-duplication by TrgID.
So by default both are made monotonic across the combined file:

  --time-mode header  (default)  Shift each run by the difference between its
        "// Run start time" and the first run's.  Timestamps then read as
        microseconds since the start of the first run, and the real wall-clock
        gaps BETWEEN runs are preserved, so a rate computed over the whole file
        is a true average rate.  Falls back to "pack" if the headers are
        missing, unparsable or not in ascending order.

  --time-mode pack    Lay the runs end to end with no gap: each run starts
        where the previous one stopped (plus --gap seconds).  Use this when the
        runs are not contiguous in time and you only want an ordered stream.

  --time-mode none    Leave TStamp_us and Tref_TStamp_us exactly as they are.

TrgID is offset so it keeps counting up across runs (--no-trgid-offset leaves
it alone).  Rewritten numbers keep the column widths and the decimal places of
the originals, and the file's CRLF line endings are preserved.

The first event of a run
------------------------
Event 0 of a Janus run can come out with every channel at ADC full scale -- a
start-of-run artifact, not data.  Downstream tools drop the first event of the
file they are given, which in a combined file would leave the artifact from
every run after the first.  --skip-events therefore drops the first N events of
EACH run, and defaults to 1 for that reason.  Pass --skip-events 0 to
concatenate losslessly.

Consistency
-----------
The column header line must match across all inputs or the files cannot be
concatenated at all; that is an error unless --force.  A difference in Board,
Acquisition Mode, File Format Version or ToA/ToT LSB is reported as a warning,
since it usually means the runs should not be pooled.
"""

import argparse
import calendar
import os
import re
import sys
import time

RUN_RE = re.compile(r"runs?[_\-\s]*0*(\d+(?:[_\-]0*\d+)*)", re.IGNORECASE)

# header keys that must agree for the runs to belong in the same file
CHECK_KEYS = ["Board", "File Format Version", "Acquisition Mode",
              "ToA/ToT LSB", "Energy Histogram NBins"]

MAX_HEADER_LINES = 200          # how far in to look for the column header


# ----------------------------------------------------------------------------
def run_numbers(path):
    """'Runs134_137_list.txt' -> ['134', '137'];  'Run99_list.txt' -> ['99']."""
    m = RUN_RE.search(os.path.basename(path))
    if not m:
        return []
    out = []
    for piece in re.split(r"[_\-]", m.group(1)):
        piece = piece.lstrip("0") or "0"
        if piece not in out:
            out.append(piece)
    return out


def name_suffix(path):
    """Whatever follows the run numbers in the name: usually '_list.txt'."""
    base = os.path.basename(path)
    m = RUN_RE.search(base)
    return base[m.end():] if m else os.path.splitext(base)[1]


def sort_key(path):
    nums = run_numbers(path)
    return (0, int(nums[0]), os.path.basename(path)) if nums \
        else (1, 0, os.path.basename(path))


def parse_start(text):
    """'Mon Sep 14 19:24:09 2026 UTC' -> seconds, or None."""
    text = (text or "").strip()
    for fmt in ("%a %b %d %H:%M:%S %Y UTC", "%a %b %d %H:%M:%S %Y"):
        try:
            return calendar.timegm(time.strptime(text, fmt))
        except ValueError:
            continue
    return None


def read_header(path):
    """The leading '//' block and the column line.

    Returns (comment lines with their terminators, column header line,
    {key: value}, the column names).
    """
    comments = []
    colline = None
    meta = {}
    with open(path, "r", errors="replace", newline="") as f:
        for i, line in enumerate(f):
            core = line.rstrip("\r\n")
            s = core.strip()
            if s.startswith("//"):
                comments.append(line)
                body = s.lstrip("/").strip()
                if ":" in body:
                    k, v = body.split(":", 1)
                    if k.strip() and not k.strip().startswith("*"):
                        meta[k.strip()] = v.strip()
                continue
            if s.split()[:1] == ["Brd"]:
                colline = line
                break
            if i > MAX_HEADER_LINES:
                break
    if colline is None:
        sys.exit("%s: no 'Brd ...' column header line found" % path)
    return comments, colline, meta, colline.split()


def col_index(names):
    """Positions of TStamp_us, Tref_TStamp_us and TrgID in the event line."""
    idx = {}
    for i, n in enumerate(names):
        low = n.lower()
        if low.startswith("tstamp"):
            idx.setdefault("ts", i)
        elif low.startswith("tref"):
            idx.setdefault("tref", i)
        elif low.startswith("trgid"):
            idx.setdefault("trg", i)
    return idx


def _decimals(tok):
    return len(tok.split(".", 1)[1]) if "." in tok else 0


def rewrite_event(core, spans, idx, dt_us, dtrg):
    """Add the offsets, keeping every column width and decimal place.

    core is the line without its terminator; spans are the (start, end) of its
    whitespace-separated fields.  Only the three numeric columns move, and each
    is re-rendered right-aligned in the width it already occupied, so a file
    that did line up still lines up.
    """
    edits = []
    for key, delta, as_int in (("ts", dt_us, False), ("tref", dt_us, False),
                               ("trg", dtrg, True)):
        i = idx.get(key)
        if i is None or delta == 0 or i >= len(spans):
            continue
        s, e = spans[i]
        tok = core[s:e]
        try:
            new = ("%d" % (int(tok) + int(delta))) if as_int \
                else ("%.*f" % (_decimals(tok), float(tok) + delta))
        except ValueError:
            continue
        edits.append((s, e, new.rjust(e - s)))
    if not edits:
        return core
    out = core
    for s, e, new in sorted(edits, reverse=True):     # right to left
        out = out[:s] + new + out[e:]
    return out


# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("listfiles", nargs="+",
                   help="two or more Janus list files, in any order")
    p.add_argument("-o", "--out", default=None,
                   help="output file (default: Runs<a>_<b>_..._list.txt built "
                        "from the input names, next to the first input)")
    p.add_argument("--skip-events", type=int, default=1,
                   help="drop this many events from the start of EACH run "
                        "(default 1: event 0 is a start-of-run artifact; "
                        "pass 0 to concatenate losslessly)")
    p.add_argument("--time-mode", choices=["header", "pack", "none"],
                   default="header",
                   help="how to make TStamp_us monotonic: 'header' shifts each "
                        "run by its Run start time so the real gaps between "
                        "runs survive (default), 'pack' lays the runs end to "
                        "end, 'none' leaves the timestamps alone")
    p.add_argument("--gap", type=float, default=0.0,
                   help="seconds of dead time to insert between runs in "
                        "--time-mode pack (default 0)")
    p.add_argument("--no-trgid-offset", action="store_true",
                   help="do not make TrgID continue across runs")
    p.add_argument("--force", action="store_true",
                   help="combine even if the column headers differ, and "
                        "overwrite an existing output file")
    p.add_argument("--dry-run", action="store_true",
                   help="say what would be written, and write nothing")
    args = p.parse_args(argv)

    for path in args.listfiles:
        if not os.path.exists(path):
            sys.exit("no such list file: %s" % path)
    if len(args.listfiles) < 2:
        print("note: only one input file; the output is a copy with the "
              "offsets and --skip-events applied")

    files = sorted(args.listfiles, key=sort_key)

    # ---- headers first, so a mismatch is caught before anything is written --
    heads = [read_header(f) for f in files]
    ref_cols = heads[0][3]
    bad = [files[i] for i in range(1, len(files)) if heads[i][3] != ref_cols]
    if bad:
        msg = ("the column header differs from %s in: %s"
               % (os.path.basename(files[0]),
                  ", ".join(os.path.basename(b) for b in bad)))
        if not args.force:
            sys.exit("error: %s\n       these files cannot be concatenated; "
                     "--force to try anyway" % msg)
        print("WARNING  %s" % msg)

    for key in CHECK_KEYS:
        vals = {}
        for f, (_c, _l, meta, _n) in zip(files, heads):
            if key in meta:
                vals.setdefault(meta[key], []).append(os.path.basename(f))
        if len(vals) > 1:
            print("WARNING  '%s' differs between runs: %s"
                  % (key, "; ".join("%s (%s)" % (v, ", ".join(fs))
                                    for v, fs in vals.items())))

    idx = col_index(ref_cols)
    nfull = len(ref_cols)
    for need in ("ts", "tref", "trg"):
        if need not in idx and args.time_mode != "none":
            print("WARNING  no '%s' column found; those offsets are skipped"
                  % need)

    # ---- output name -------------------------------------------------------
    nums = []
    for f in files:
        for n in run_numbers(f):
            if n not in nums:
                nums.append(n)
    if args.out:
        out_path = args.out
    elif nums:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(files[0])),
            "Runs%s%s" % ("_".join(nums), name_suffix(files[0])))
    else:
        sys.exit("no run numbers in the input names; pass -o")
    if os.path.abspath(out_path) in {os.path.abspath(f) for f in files}:
        sys.exit("the output would overwrite an input: %s" % out_path)
    if os.path.exists(out_path) and not args.force and not args.dry_run:
        sys.exit("%s exists; --force to overwrite" % out_path)

    # ---- how the clock is handled -----------------------------------------
    starts = [parse_start(meta.get("Run start time"))
              for _c, _l, meta, _n in heads]
    mode = args.time_mode
    if mode == "header":
        if any(s is None for s in starts):
            print("note: 'Run start time' missing or unparsable in at least "
                  "one run -- falling back to --time-mode pack")
            mode = "pack"
        elif any(starts[i] < starts[i - 1] for i in range(1, len(starts))):
            print("note: run start times are not in ascending order -- "
                  "falling back to --time-mode pack")
            mode = "pack"

    print("=" * 78)
    print("runs        %s" % ", ".join(nums) if nums else "(none parsed)")
    for f, s in zip(files, starts):
        print("  %-34s %s" % (os.path.basename(f),
                              time.asctime(time.gmtime(s)) + " UTC"
                              if s else "(no start time)"))
    print("output      %s" % out_path)
    print("time mode   %s%s" % (mode,
                                "   gap %g s" % args.gap
                                if mode == "pack" else ""))
    print("trgid       %s" % ("left as-is" if args.no_trgid_offset
                              else "offset to keep counting across runs"))
    print("skip        first %d event(s) of each run" % args.skip_events)
    if args.dry_run:
        print("dry run: nothing written")
        return
    print("-" * 78)

    term = heads[0][0][0][len(heads[0][0][0].rstrip("\r\n")):] or "\n"

    # ---- write -------------------------------------------------------------
    rows = []
    dt_us = 0.0
    dtrg = 0
    cum_end_us = 0.0          # end of everything written so far, for 'pack'
    kept_total = 0

    with open(out_path, "w", newline="") as out:
        for k, (path, (comments, colline, meta, names)) in \
                enumerate(zip(files, heads)):
            if k == 0:
                for line in comments[:-1] if len(comments) > 1 else comments:
                    out.write(line)
                prov = ["// combineRuns.py: %d runs combined -> %s"
                        % (len(files), os.path.basename(out_path)),
                        "// combined runs: %s" % ", ".join(nums),
                        "// time mode: %s; trgid offset: %s; skipped first "
                        "%d event(s) of each run"
                        % (mode, "no" if args.no_trgid_offset else "yes",
                           args.skip_events)]
                for line in prov:
                    out.write(line + term)
                if len(comments) > 1:
                    out.write(comments[-1])          # the closing //***** rule
                out.write(colline)

            # offset for this run
            if k == 0:
                dt_us = 0.0
                dtrg = 0
            else:
                if mode == "header":
                    dt_us = (starts[k] - starts[0]) * 1e6
                elif mode == "pack":
                    dt_us = cum_end_us + args.gap * 1e6
                else:
                    dt_us = 0.0
            if args.no_trgid_offset:
                dtrg_use = 0
            else:
                dtrg_use = dtrg

            nev = 0
            nkept = 0
            first_ts = last_ts = None
            max_trg = -1
            keep = False
            seen_cols = False

            with open(path, "r", errors="replace", newline="") as f:
                for line in f:
                    core = line.rstrip("\r\n")
                    s = core.strip()
                    if not s:
                        continue
                    if s.startswith("//"):
                        continue                      # header handled above
                    if not seen_cols and s.split()[:1] == ["Brd"]:
                        seen_cols = True
                        continue
                    if s.split()[:1] == ["Brd"]:
                        continue
                    spans = [m.span() for m in re.finditer(r"\S+", core)]
                    if len(spans) >= nfull:           # starts a new event
                        nev += 1
                        keep = nev > args.skip_events
                        i_ts, i_trg = idx.get("ts"), idx.get("trg")
                        if i_ts is not None and i_ts < len(spans):
                            try:
                                v = float(core[spans[i_ts][0]:spans[i_ts][1]])
                                first_ts = v if first_ts is None else first_ts
                                last_ts = v
                            except ValueError:
                                pass
                        if i_trg is not None and i_trg < len(spans):
                            try:
                                max_trg = max(
                                    max_trg,
                                    int(core[spans[i_trg][0]:spans[i_trg][1]]))
                            except ValueError:
                                pass
                        if keep:
                            nkept += 1
                            core = rewrite_event(core, spans, idx,
                                                 dt_us, dtrg_use)
                    if keep:
                        out.write(core + line[len(line.rstrip("\r\n")):])

            dur = (last_ts - first_ts) / 1e6 \
                if (first_ts is not None and last_ts is not None) else 0.0
            rows.append((os.path.basename(path), nev, nkept, dur,
                         max_trg, dt_us / 1e6, dtrg_use))
            kept_total += nkept
            cum_end_us = max(cum_end_us, dt_us + (last_ts or 0.0))
            if not args.no_trgid_offset:
                dtrg += max_trg + 1

    print("  %-28s %8s %8s %10s %9s %12s %9s"
          % ("run file", "events", "kept", "span [s]", "trgid max",
             "t shift [s]", "trg shift"))
    for name, nev, nkept, dur, mtrg, sh, dtr in rows:
        print("  %-28s %8d %8d %10.1f %9d %12.1f %9d"
              % (name, nev, nkept, dur, mtrg, sh, dtr))
    print("-" * 78)
    print("wrote %s" % out_path)
    print("      %d events from %d runs (%d dropped as start-of-run)"
          % (kept_total, len(files),
             sum(r[1] - r[2] for r in rows)))
    if kept_total:
        print("note: downstream, --skip-events 0 is now enough; the "
              "start-of-run events are already gone"
              if args.skip_events else
              "note: nothing was skipped; each run still carries its event 0")


if __name__ == "__main__":
    main()
