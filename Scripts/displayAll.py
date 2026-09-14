#!/usr/bin/env python3
"""
Loop over all events in a Janus DT5202 Spect_Timing file, select events
where HG(channel 0) > 1000 and HG(channel 1) > 1000, and save one detector
image per selected event.

Usage:
    python displayAll.py FILE [-hg]

Examples:
    python displayAll.py Run111_list.txt
    python displayAll.py Run111_list.txt -hg

The event-selection threshold is always applied to HG for channels 0 and 1.
The detector image displays LG by default; use -hg to display HG instead.

Output:
    RunX/eventY.png

where X is parsed from the input filename (e.g. Run111_list.txt -> Run111)
and Y is the TrgID.
"""

import sys
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


TRIGGER_THRESHOLD = 1000.0

CHANNEL_MAP = {
    6: ("A", 3),  7: ("C", 3),
    8: ("A", 4),  9: ("C", 4),
    10: ("A", 5), 11: ("C", 5),
    12: ("A", 6), 13: ("C", 6),
    14: ("A", 7), 15: ("C", 7),
    16: ("A", 8), 17: ("C", 8),
    18: ("A", 9), 19: ("C", 9),
    20: ("A", 10), 21: ("C", 10),
    22: ("A", 11), 23: ("C", 11),
    24: ("A", 12), 25: ("C", 12),
    26: ("A", 13), 27: ("C", 13),
    28: ("A", 14), 29: ("C", 14),
    30: ("A", 15), 31: ("C", 15),
    32: ("B", 0), 33: ("D", 0),
    34: ("B", 1), 35: ("D", 1),
    36: ("B", 2), 37: ("D", 2),
    38: ("B", 3), 39: ("D", 3),
    40: ("B", 4), 41: ("D", 4),
    42: ("B", 5), 43: ("D", 5),
    44: ("B", 6), 45: ("D", 6),
    46: ("B", 7), 47: ("D", 7),
    48: ("B", 8), 49: ("D", 8),
    50: ("B", 9), 51: ("D", 9),
    52: ("B", 10), 53: ("D", 10),
    54: ("B", 11), 55: ("D", 11),
    56: ("B", 12), 57: ("D", 12),
    58: ("B", 13), 59: ("D", 13),
    60: ("B", 14), 61: ("D", 14),
    62: ("B", 15), 63: ("D", 15),
}

BOARD_TO_COL = {"A": 0, "B": 1, "C": 2, "D": 3}


def channel_assignment(ch):
    if ch == 0:
        return "CW top"
    if ch == 1:
        return "CW bottom"
    if 2 <= ch <= 5:
        return "unconnected"
    if ch in CHANNEL_MAP:
        board, pos = CHANNEL_MAP[ch]
        return f"{board}{pos}"
    return "unassigned"


def read_all_events(filename):
    events = []
    current_event = None
    current_trgid = None

    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("//") or line.startswith("Brd"):
                continue

            fields = line.split()
            if len(fields) < 6:
                continue

            try:
                ch = int(fields[1])
                lg = float(fields[2])
                hg = float(fields[3])
            except (ValueError, IndexError):
                continue

            if len(fields) >= 10:
                try:
                    trgid = int(fields[8])
                except ValueError:
                    continue

                if current_event is not None:
                    events.append((current_trgid, current_event))

                current_trgid = trgid
                current_event = {}

            if current_event is not None:
                current_event[ch] = {
                    "LG": lg,
                    "HG": hg,
                    "ToA_ns": fields[4],
                    "ToT_ns": fields[5],
                }

    if current_event is not None:
        events.append((current_trgid, current_event))

    return events


def passes_trigger(event):
    return (
        0 in event and 1 in event
        and event[0]["HG"] > TRIGGER_THRESHOLD
        and event[1]["HG"] > TRIGGER_THRESHOLD
    )


def make_image(event, gain="LG"):
    image = np.full((16, 4), np.nan, dtype=float)
    for ch, (board, idx) in CHANNEL_MAP.items():
        if ch in event:
            image[idx, BOARD_TO_COL[board]] = event[ch][gain]
    return image


def print_event(event, trgid):
    print()
    print("=" * 62)
    print(
        f"TrgID {trgid} selected: "
        f"ch0 HG={event[0]['HG']:g}, ch1 HG={event[1]['HG']:g}"
    )
    print("=" * 62)
    print(f"{'Channel':>7}  {'LG':>8}  {'HG':>8}  {'Assignment':>12}")
    print(f"{'-'*7}  {'-'*8}  {'-'*8}  {'-'*12}")

    for ch in range(64):
        if ch in event:
            lg = f"{event[ch]['LG']:g}"
            hg = f"{event[ch]['HG']:g}"
        else:
            lg = "-"
            hg = "-"
        print(f"{ch:7d}  {lg:>8}  {hg:>8}  {channel_assignment(ch):>12}")


def save_event_plot(event, trgid, gain, run_label, output_dir):
    image = make_image(event, gain)

    finite = image[np.isfinite(image)]
    vmin = 0.0
    vmax = 1000.0 if gain == "LG" else 4000.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("0.85")

    fig, ax = plt.subplots(figsize=(6.5, 9.5))

    ax.imshow(
        np.ma.masked_invalid(image),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=(-0.5, 3.5, -0.5, 15.5),
    )

    for x in np.arange(-0.5, 4.0, 1.0):
        ax.axvline(x, linewidth=0.8, color="black", alpha=0.45)
    for y in np.arange(-0.5, 16.0, 1.0):
        ax.axhline(y, linewidth=0.6, color="black", alpha=0.35)

    ax.set_xticks(range(4))
    ax.set_xticklabels(["A", "B", "C", "D"], fontsize=13, fontweight="bold")
    ax.set_yticks(range(16))
    ax.set_yticklabels(range(16))
    ax.set_xlabel("Detector board", fontsize=12)
    ax.set_ylabel("SiPM position", fontsize=12)
    ax.set_title(f"{run_label} — TrgID {trgid} — {gain}", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 15.5)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.03)
    cbar.set_label(f"{gain} counts", fontsize=11)

    fig.tight_layout()
    outfile = output_dir / f"event{trgid}.png"
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outfile


def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python displayAll.py FILE [-hg]")
        sys.exit(1)

    filename = sys.argv[1]
    gain = "LG"
    if len(sys.argv) == 3:
        if sys.argv[2].lower() != "-hg":
            print("ERROR: optional argument must be -hg")
            sys.exit(1)
        gain = "HG"

    if not Path(filename).is_file():
        print(f"ERROR: file not found: {filename}")
        sys.exit(1)

    match = re.search(r"Run(\d+)", Path(filename).name, re.IGNORECASE)
    if match:
        run_dir_name = f"Run{match.group(1)}"
        run_label = f"Run {match.group(1)}"
    else:
        run_dir_name = Path(filename).stem
        run_label = Path(filename).stem

    output_dir = Path(run_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = read_all_events(filename)
    selected = [(trgid, event) for trgid, event in events if passes_trigger(event)]

    print(f"Input file: {filename}")
    print(f"Total events: {len(events)}")
    print(
        f"Events with ch0 HG > {TRIGGER_THRESHOLD:g} "
        f"and ch1 HG > {TRIGGER_THRESHOLD:g}: {len(selected)}"
    )
    print(f"Output folder: {output_dir}/")

    for trgid, event in selected:
        print_event(event, trgid)
        outfile = save_event_plot(event, trgid, gain, run_label, output_dir)
        print(f"Saved {outfile}")

    print()
    print(f"Number of events above threshold: {len(selected)}")


if __name__ == "__main__":
    main()
