#!/usr/bin/env python3
"""
Display one Janus DT5202 Spect_Timing event as a 4-column x 16-row detector image.

Usage:
    python display.py FILE TrgID [-hg]

Example:
    python display.py Run111_list.txt 545
    python display.py Run111_list.txt 545 -hg

Columns correspond to the four longitudinal SiPM boards around the cylinder:
    A  B  C  D
Rows correspond to SiPM positions 0..15, with 0 at the bottom and 15 at the top.

Color encodes LG counts by default. Use -hg to display HG counts instead.
The Sept. 9 Janus-channel mapping is hard-coded below.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# Sept. 9 mapping: Janus channel -> (detector board, SiPM position)
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


def read_event(filename, wanted_trgid):
    """Return {channel: {LG, HG, ToA_ns, ToT_ns}} for one TrgID."""
    event = {}
    in_event = False

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

            # The first line of each event has the extra timestamp / TrgID fields.
            if len(fields) >= 10:
                try:
                    trgid = int(fields[8])
                except ValueError:
                    continue

                if in_event and trgid != wanted_trgid:
                    break
                in_event = (trgid == wanted_trgid)

            if in_event:
                event[ch] = {
                    "LG": lg,
                    "HG": hg,
                    "ToA_ns": fields[4],
                    "ToT_ns": fields[5],
                }

    if not event:
        raise ValueError(f"TrgID {wanted_trgid} was not found in {filename}")

    return event


def make_image(event, gain="LG"):
    """Build a 16-row x 4-column image for LG or HG; row 0 is bottom."""
    image = np.full((16, 4), np.nan, dtype=float)

    for ch, (board, idx) in CHANNEL_MAP.items():
        if ch in event:
            image[idx, BOARD_TO_COL[board]] = event[ch][gain]

    return image


def print_mapping(event, gain="LG"):
    print(f"\nJanus channel    {gain}    mapped channel")
    print("-------------  ------  --------------")
    for ch in sorted(CHANNEL_MAP):
        if ch not in event:
            continue
        board, idx = CHANNEL_MAP[ch]
        value = event[ch][gain]
        print(f"{ch:13d}  {value:6g}  {board}{idx}")
    print()


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python display.py FILE TrgID [-hg]")
        sys.exit(1)

    filename = sys.argv[1]
    gain = "LG"
    if len(sys.argv) == 4:
        if sys.argv[3].lower() != "-hg":
            print("ERROR: optional argument must be -hg")
            sys.exit(1)
        gain = "HG"

    try:
        wanted_trgid = int(sys.argv[2])
    except ValueError:
        print("ERROR: TrgID must be an integer")
        sys.exit(1)

    if not Path(filename).is_file():
        print(f"ERROR: file not found: {filename}")
        sys.exit(1)

    try:
        event = read_event(filename, wanted_trgid)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print_mapping(event, gain)
    image = make_image(event, gain)

    finite = image[np.isfinite(image)]
    vmin = 0.0
    vmax = max(1.0, float(np.max(finite))) if finite.size else 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    #cmap = plt.get_cmap("viridis").copy()
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("0.85")  # gray = not connected

    fig, ax = plt.subplots(figsize=(6.5, 9.5))

    # origin='lower' means SiPM 0 is at the bottom and 15 at the top.
    ax.imshow(
        np.ma.masked_invalid(image),
        origin="lower",
        aspect="auto",
        #cmap=cmap,
        cmap="Blues",
        norm=norm,
        interpolation="nearest",
        extent=(-0.5, 3.5, -0.5, 15.5),
    )

    # Cell boundaries only; no channel/count text inside the image.
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
    ax.set_title(f"TrgID {wanted_trgid} — {gain}", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 15.5)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.03)
    cbar.set_label(f"{gain} counts", fontsize=11)

    fig.tight_layout()
    out = f"event_{wanted_trgid}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
