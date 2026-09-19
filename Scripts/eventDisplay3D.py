#!/usr/bin/env python3
"""
eventDisplay3D.py

Interactive 3-D event display for the SiPM cylinder prototype, built on
PyVista / VTK.

    python eventDisplay3D.py EVENTS.txt                # interactive window
    python eventDisplay3D.py EVENTDIR/*.txt            # several files at once
    python eventDisplay3D.py EVENTS.txt --html out.html   # standalone web page
    python eventDisplay3D.py EVENTS.txt --html site/      # a page per event

The window opens on the bare detector, with no event in it.  Right loads the
first event, Right again steps forward, Left steps back, and Left from the
first event returns to the empty detector.

Layout
------
Three viewports.  The middle one holds the detector, and the HUD, the x/y/z
triad and the photon colour bar all sit inside it, next to the detector rather
than out at the window edges.  Left is the 4 x 11 photon map of the event on
screen; right is the theta-phi radiograph, with the trigger acceptance shaded
and one dot for this event.  Both side panels are drawn as their viewport's
background, so they stay flat and still while the detector is rotated, zoomed
or panned.  --no-panels drops them and gives the detector the whole window.

Mouse
-----
    left drag    rotate
    scroll       zoom in / out
    middle drag  pan          (or shift + left drag)
    right drag   dolly

Keys
----
    Right / n    next event  (from the setup view, load the first event)
    Left  / p    previous event, back to the setup view
    Home         the setup view again
    Return       save the window as <event>.png and <event>.pdf
    a            replay the flight, with --animate
    r            reset the camera
    s / w        surface / wireframe
    q            quit

Input
-----
One file holds any number of events, five lines each, as applyCNN.py
--printTextFiles writes them:

    TrigID  theta  phi  phi_confidence  [t0 t1 t2 t3]
    <4 rows of 11 photon counts>

so a 5-event file is 25 lines.  Blank lines and #-comments are skipped, and
several files can be given at once; they are concatenated in order.

Rows are detector columns A, B, C, D; array column 0 is ring 15 and column 10
is ring 5, exactly as convertDataFile.py stores them.

t0..t3 are optional CosmicWatch trigger ADC values, 0 to 1024.  They are
printed in the HUD, starred above 500; the channels themselves are static
geometry and carry no per-event readout.  Channels are numbered along the
muon's path: 0 and 1 are the upper pair (outer unit first), 2 and 3 the lower
pair (inner unit first).

Trigger telescope
-----------------
Four 5 x 5 x 1 cm channels, the 1 cm axis radial about the track's axis
crossing at z = TRACK_Z0, in two facing pairs on the accepted-muon direction
theta = 50, phi = 0, 17 cm out from the pivot (--trig-distance).  Each is a
scintillator slab on a board of the same 5 x 5 footprint; the slab is drawn in
the cylinder's translucent blue, though the real one is wrapped in tape.  A
SiPM sits at the centre of each board, drawn like the ones on the cylinder
boards.  It carries no read-out value, so it is simply dark on the setup view
and lit at the top of the colour ramp -- the shade the brightest SiPM on the
cylinder is wearing -- whenever an event is loaded.

At 17 cm the face spans theta 41.6 to 58.4 and phi -10.9 to +10.9.  A square
face always subtends MORE in phi than in theta -- the phi lever arm is
d*sin(theta), the theta lever arm is d -- so no distance gives both 40-60 in
theta and +/-8.5 in phi at once; +/-8.5 would need the width cut to 3.9 cm.

Geometry
--------
Taken from the CAD and the bv4 gerbers:

    scintillator    cylinder, 7.5 cm diameter x 30 cm tall
    PCB             360 x 12 x 1.6 mm, four of them at 90 degree spacing,
                    tangential to the cylinder, component side facing inward
    SiPM            16 positions at 19.6 mm pitch (the gerber's U1..U16),
                    6 x 6 mm; rings 5..15 are the eleven that are read out

Ring 15 is at the top.  Rings 0-4 exist on the board but are not in the data,
so only the upper eleven positions are drawn.

Which way is phi
----------------
Board A sits at phi = 0 and B, C, D follow counter-clockwise seen from above,
at 90, 180 and 270.  That handedness is checked against the training MC: the
azimuth of the light centroid, computed from the four board totals, reproduces
the generated phi exactly (mean cos(dphi) = 1.000 over 4000 events), while the
mirrored assignment gives 0.000.  The brass bearing ring under the detector
shows the four boards and carries a gold needle along phi, and the HUD prints
both the network's phi and the azimuth of the light, so the two can be
compared at a glance.

Which end of the track is at phi is NOT a free convention, and the MC cannot
answer it: there the ring profile is identical on all four boards, so the
training data has no z-phi correlation at all.  The real data does have one.
Fitting the ring-centroid difference between the board nearest the light and
the board opposite it, over the 834 triggered events of Run140:

    cen_near - cen_far = +0.62 * tan(theta) - 0.44      (slope 9.4 sigma)

with the ring index increasing downwards.  A positive slope means the phi-side
board is lit LOWER than the board opposite, and the separation grows with the
track's tilt, exactly as a straight track should.  So the phi end of the track
is its BOTTOM end: the muon TRAVELS TOWARDS phi, entering high on the far side
and leaving low on the phi side.  That is the default.  --phi-sense flips it:

    to     (default)  muon travels towards phi -- the sense the data supports
    from              muon arrives from phi: the same line rotated 180 deg in
                      azimuth, kept for comparison

(The fitted slope is ~6x smaller than the 2R/pitch = 3.8 a track through the
detector centre would give, which is what you would expect once the light
spreads and the tracks are spread over impact parameter.  The -0.44 offset is
a common-mode board asymmetry and does not affect the sign.)

Scintillation photons
---------------------
--showPhotons draws the light.  Photons are emitted at random points along the
chord the muon cuts through the cylinder, at --nPhotonsPerCM per cm (default
500), isotropically, and each line stops where it meets the wall or an end cap.
They are drawn at 420 nm, the emission peak of the plastic, faint enough that
the detector still shows through several thousand of them.

This is a picture of an isotropic emitter, not a light-collection simulation:
the lines are straight, and there is no refraction at the wall, no reflection,
no attenuation and no wrapping.  Do not read the density at the wall as the
number of photons a SiPM would see.

--animate flies the muon along its track instead of drawing it finished: the
line grows from the entry point, and each photon appears as the muon reaches
its emission point and spreads outward behind it.  --speed sets the pace in
cm/s (default 12, about 3.8 s for the 46 cm of drawn track) and the photons
travel at that same pace, which is the other place the picture parts company
with the physics -- real scintillation light would cross the 7.5 cm cylinder
in about 0.4 ns, some 10^10 times faster than the muon covers the same ground.
Equal speeds are what make the light legible as it goes.  A window is needed
for the clock, so --animate is ignored for --screenshot and --html; 'a'
replays the event on screen.

Web
---
--html writes a self-contained page (vtk.js under the hood) that rotates,
zooms and pans in any browser with no server and no Python.  Copy it onto a
web server, or just open it.  Give a directory instead of a .html file and
every event gets a page plus an index.
"""

import argparse
import glob
import os
import sys
import tempfile

import time

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------------
# geometry, in centimetres
# ----------------------------------------------------------------------------
R_CYL = 3.75                 # scintillator radius (7.5 cm diameter)
H_CYL = 30.0                 # scintillator height

N_RING = 16                  # SiPM positions on a board (gerber U1..U16)
SIPM_PITCH = 1.96            # 19.6 mm, measured from the gerber paste layer
SIPM_SIZE = 0.60             # 6 mm square
SIPM_THICK = 0.15            # the die itself
# The die sits on the INNER face of the board, where it cannot be seen from
# outside.  It is drawn thick enough to pass through the board so the channel
# reads as a lit patch from either side -- the one deliberate departure from
# the real geometry, and the difference between a display you can read and one
# you cannot.
SIPM_DRAW_THICK = 0.34

PCB_LEN = 36.0               # 360 mm, from the gerber outline
PCB_WIDTH = 1.2              # 12 mm
PCB_THICK = 0.16

GAP = 0.25                   # air gap between scintillator and board
R_PCB_IN = R_CYL + GAP                        # inner face of the board
R_PCB = R_PCB_IN + PCB_THICK / 2.0            # board centre
R_SIPM = R_PCB                                # drawn centred on the board

# ---- scintillation photons (--showPhotons) ---------------------------------
PHOTON_NM = 420.0            # emission peak of the plastic, in nanometres
PHOTONS_PER_CM = 500.0       # lines per cm of track inside the scintillator
# Per-line alpha.  It goes down as the default density goes up: five thousand
# lines through one cylinder stack into an opaque violet wall unless each one
# is faint, and the point of drawing them is to see the detector behind them.
PHOTON_OPACITY = 0.08
# VTK quantises line width to quarter-pixel steps -- 1.0, 1.25, 1.5 ... render
# distinguishably, anything between them does not -- so this is two steps up
# from a hairline.
PHOTON_WIDTH = 1.5
# A hard ceiling on the lines drawn.  A near-vertical track cuts the full 30 cm
# of the cylinder, which at the default density is 15000 photons -- so that
# case just fits, and a higher --nPhotonsPerCM is shown with fewer photons
# rather than at a frame rate that makes the animation crawl.
PHOTON_MAX = 15000

# ---- the drawn muon, and the animation of it (--animate) -------------------
MUON_HALF_LEN = 23.0         # the track is drawn this far either side of z0
MUON_MIN_ARC = 0.02          # a tube needs a length; frame zero gets this one
# Centimetres per second along the track, and the photons travel at the same
# speed -- a light-speed photon would be at the wall before the eye caught it,
# so this is a deliberate slow-motion, not a time-of-flight statement.  The
# track is 2 x MUON_HALF_LEN long, so the default crossing takes about 3.8 s.
ANIM_SPEED = 12.0
ANIM_FRAME_MS = 33           # ~30 frames a second
ANIM_MAX_STEPS = 10 ** 7     # the timer runs for the life of the window

COLS = ["A", "B", "C", "D"]
COL_PHI = {"A": 0.0, "B": 90.0, "C": 180.0, "D": 270.0}

TRACK_Z0 = 3.0               # height at which the drawn track crosses the axis

# ---- CosmicWatch trigger telescope -----------------------------------------
# Four channels in two facing pairs, on the accepted-muon axis through
# (0, 0, TRACK_Z0).  Channel order is along the muon's path: 0 and 1 are the
# upper pair (outer first), 2 and 3 the lower pair (inner first).
N_TRIG = 4
TRIG_SIZE = 5.0              # scintillator face, 5 x 5 cm
TRIG_THICK = 1.0             # 1 cm, along the radial direction
TRIG_PAIR_GAP = 0.60         # between the two units of a pair
TRIG_THETA = 50.0            # accepted direction: theta at the centre
TRIG_PHI = 0.0               # ... and phi
TRIG_DIST = 17.0             # pivot to the centre of each pair
TRIG_ADC_MAX = 1024.0        # Arduino analogRead full scale
TRIG_HOT = 500.0             # HUD stars a channel above this

CW_PCB = "#1d6b46"           # the CosmicWatch board
CW_PCB_X = 0.5 * TRIG_THICK + 0.11   # board centre, along the channel's own +x
# The trigger scintillator is drawn in the same pale blue as the cylinder
# (SCINT, below).  The cylinder gets its look from two stacked shells, 0.17
# over 0.10, so a single slab needs a touch more than 0.17 to sit at the same
# apparent density.
TRIG_SCINT_OPACITY = 0.26
# The trigger SiPM lights at full scale, so its bloom is the size build_halo
# gives a tile at the event maximum: 1.0 + 2.1.
TRIG_HALO_SCALE = 3.1

RING_HI, RING_LO = 15, 5     # the eleven rings that are read out
NRING_DATA = RING_HI - RING_LO + 1

# z of ring i, with the 16-position array centred on the cylinder
Z0 = -0.5 * (N_RING - 1) * SIPM_PITCH         # ring 0
def ring_z(i):
    return Z0 + i * SIPM_PITCH


# ----------------------------------------------------------------------------
# palette -- dark, brass and cyan, after the helmet
# ----------------------------------------------------------------------------
BG_LOW = "#06090d"
BG_HIGH = "#16222e"
SCINT = "#9ec9ee"
PCB_GREEN = "#12503a"
BRASS = "#c39a5e"
BRASS_DARK = "#8a6a38"
BRASS_LIT = "#ffd27f"          # the phi needle on the bearing ring
GUNMETAL = "#2b3138"
MUON = "#31e8ff"
HUD = "#8fe8f2"
# Point sizes for the text in the 3-D viewport, at a 950-pixel-tall window.
# Kept small so the block in the top corner stays clear of the detector; both
# are scaled by the window height at draw time.
HUD_BASE = 5.9
FOOT_BASE = 4.6
DEAD = "#243039"              # a SiPM with no light

# VTK's built-in courier/arial have no Greek glyphs -- theta and phi come out
# as blanks -- so the HUD is drawn with matplotlib's DejaVu Sans Mono, which
# has them and keeps the columns lined up.
def _hud_font():
    try:
        import matplotlib
        path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf",
                            "DejaVuSansMono.ttf")
        if os.path.exists(path):
            return {"font_file": path}
    except Exception:
        pass
    return {"font": "courier"}


HUD_FONT = None                               # filled in on first use
THETA, PHI = "\u03b8", "\u03c6"


def hud_font():
    global HUD_FONT
    if HUD_FONT is None:
        HUD_FONT = _hud_font()
    return dict(HUD_FONT)


# cyan through brass to pale gold: the two accent colours of the helmet, in a
# ramp that still reads monotonically from dark to bright
GLOW = LinearSegmentedColormap.from_list("glow", [
    "#0b1a26", "#0f4a63", "#1793b4", "#43dbe6",
    "#a8c96a", "#d8a63c", "#f2cf6d", "#fff3c4",
])


# ----------------------------------------------------------------------------
# input
# ----------------------------------------------------------------------------
def read_events(path, nrow=4, nring=NRING_DATA):
    """Every event in one file -> list of dicts.

    A file is a stack of (1 + nrow)-line blocks, so it holds one event or a
    thousand with no change of format.  Blank lines and #-comments are
    dropped first, which is what lets a hand-edited file still parse.
    """
    block = 1 + nrow
    with open(path) as f:
        lines = [ln.strip() for ln in f
                 if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        sys.exit("%s: no events found" % path)
    if len(lines) % block:
        sys.exit("%s: %d content lines is not a whole number of %d-line "
                 "events (%d left over)"
                 % (path, len(lines), block, len(lines) % block))

    base = os.path.splitext(os.path.basename(path))[0]
    nev = len(lines) // block
    events = []
    for k in range(nev):
        off = k * block
        h = lines[off].split()
        if len(h) < 4:
            sys.exit("%s line %d: header should be 'TrigID theta phi "
                     "phi_confidence [t0 t1 t2 t3]'" % (path, off + 1))
        trig = None
        if len(h) >= 4 + N_TRIG:
            trig = np.array([float(x) for x in h[4:4 + N_TRIG]], dtype=float)
        elif len(h) > 4:
            sys.exit("%s line %d: %d extra values after phi_confidence, "
                     "expected %d trigger ADC values"
                     % (path, off + 1, len(h) - 4, N_TRIG))
        counts = []
        for r in range(nrow):
            v = lines[off + 1 + r].split()
            if len(v) != nring:
                sys.exit("%s line %d: %d values, expected %d"
                         % (path, off + r + 2, len(v), nring))
            counts.append([float(x) for x in v])
        trgid = int(float(h[0]))
        events.append({"path": path,
                       "name": base if nev == 1
                               else "%s_evt%03d_trg%d" % (base, k, trgid),
                       "trgid": trgid,
                       "theta": float(h[1]),
                       "phi": float(h[2]),
                       "conf": float(h[3]),
                       "trig": trig,
                       "counts": np.asarray(counts, dtype=float)})
    return events


def read_event(path, **kw):
    """The first event in a file, for callers that want just one."""
    return read_events(path, **kw)[0]


# ----------------------------------------------------------------------------
# static geometry: everything that does not change from event to event
# ----------------------------------------------------------------------------
def cyl_shell(radius, height, z=0.0, res=96, capping=False):
    return pv.Cylinder(center=(0, 0, z), direction=(0, 0, 1), radius=radius,
                       height=height, resolution=res, capping=capping)


def placed_box(radius, phi_deg, z, dx, dy, dz):
    """A box on the cylinder at (radius, phi, z).

    dx is radial (thickness), dy tangential (width), dz along the axis.  Built
    on the +x axis and rotated round to phi, so 'tangential' comes out right
    without any quaternion juggling.
    """
    b = pv.Cube(center=(radius, 0.0, z), x_length=dx, y_length=dy,
                z_length=dz)
    return b.rotate_z(phi_deg, point=(0, 0, 0), inplace=False)


def build_detector():
    """Every static actor, as (mesh, style dict) pairs."""
    parts = []

    # ---- scintillator ----
    parts.append((cyl_shell(R_CYL, H_CYL, capping=True),
                  dict(color=SCINT, opacity=0.17, smooth_shading=True,
                       specular=1.0, specular_power=35, ambient=0.28,
                       name="scintillator")))
    # a brighter skin just inside it, so the cylinder reads as a solid object
    # rather than a pane of glass
    parts.append((cyl_shell(R_CYL * 0.995, H_CYL * 0.999),
                  dict(color=SCINT, opacity=0.10, smooth_shading=True,
                       specular=1.0, specular_power=80, name="scint_skin")))

    # ---- brass collars top and bottom ----
    for z in (-H_CYL / 2 + 0.45, H_CYL / 2 - 0.45):
        parts.append((cyl_shell(R_CYL + 0.06, 0.9, z=z),
                      dict(color=BRASS, smooth_shading=True, specular=1.0,
                           specular_power=60, ambient=0.18, diffuse=0.85)))
        parts.append((cyl_shell(R_CYL + 0.13, 0.28, z=z),
                      dict(color=BRASS_DARK, smooth_shading=True,
                           specular=0.9, specular_power=50)))

    for col in COLS:
        phi = COL_PHI[col]

        # ---- the board ----
        z_top = ring_z(N_RING - 1) + 5.1          # connector end
        z_bot = z_top - PCB_LEN
        z_mid = 0.5 * (z_top + z_bot)
        parts.append((placed_box(R_PCB, phi, z_mid, PCB_THICK, PCB_WIDTH,
                                 PCB_LEN),
                      dict(color=PCB_GREEN, smooth_shading=True,
                           specular=0.45, specular_power=25, ambient=0.20)))

        # brass rails down both long edges -- the one steampunk flourish that
        # also happens to be where a real board would be clamped
        for sgn in (-1, 1):
            rail = placed_box(R_PCB, phi, z_mid, PCB_THICK * 1.05, 0.10,
                              PCB_LEN * 0.995)
            rail = rail.translate(
                (-sgn * np.sin(np.radians(phi)) * (PCB_WIDTH / 2 - 0.05),
                 sgn * np.cos(np.radians(phi)) * (PCB_WIDTH / 2 - 0.05), 0.0),
                inplace=False)
            parts.append((rail, dict(color=BRASS, smooth_shading=True,
                                     specular=1.0, specular_power=70)))

        # ---- connector at the top of the board ----
        parts.append((placed_box(R_PCB + PCB_THICK, phi, z_top - 2.2,
                                 0.35, PCB_WIDTH * 0.9, 2.6),
                      dict(color=GUNMETAL, smooth_shading=True, specular=0.7,
                           specular_power=30)))

        # ---- standoffs to the cylinder ----
        for z in (-11.0, 0.0, 11.0):
            rod = pv.Cylinder(center=(R_CYL + GAP / 2, 0, z),
                              direction=(1, 0, 0), radius=0.11, height=GAP,
                              resolution=16)
            parts.append((rod.rotate_z(phi, point=(0, 0, 0), inplace=False),
                          dict(color=BRASS_DARK, smooth_shading=True,
                               specular=0.9, specular_power=40)))

        # ---- the five unread SiPM positions, drawn dark ----
        for i in range(0, RING_LO):
            parts.append((placed_box(R_SIPM, phi, ring_z(i),
                                     SIPM_DRAW_THICK, SIPM_SIZE, SIPM_SIZE),
                          dict(color=DEAD, smooth_shading=True,
                               specular=0.3, ambient=0.25)))
    return parts


def build_sipms(counts):
    """The 44 read-out SiPMs as ONE mesh carrying the photon count per cell.

    One merged mesh rather than 44 actors: it is what keeps the scene light
    enough to spin smoothly, and it gives the scalar bar for free.
    """
    blocks = []
    for ci, col in enumerate(COLS):
        phi = COL_PHI[col]
        for k in range(NRING_DATA):
            ring = RING_HI - k                    # column 0 is ring 15
            tile = placed_box(R_SIPM, phi, ring_z(ring),
                              SIPM_DRAW_THICK, SIPM_SIZE, SIPM_SIZE)
            tile.cell_data["photons"] = np.full(tile.n_cells,
                                                counts[ci, k], dtype=float)
            blocks.append(tile)
    return pv.MultiBlock(blocks).combine()


def build_halo(counts, frac=0.12):
    """A soft oversized shell around the brightest tiles: cheap bloom.

    VTK has no bloom pass here, so the glow is faked with a translucent box
    a little larger than the tile.  Only the tiles above `frac` of the event
    maximum get one, or the whole detector would look foggy.
    """
    vmax = float(np.max(counts)) if counts.size else 0.0
    if vmax <= 0:
        return None
    blocks = []
    for ci, col in enumerate(COLS):
        phi = COL_PHI[col]
        for k in range(NRING_DATA):
            v = counts[ci, k]
            if v < frac * vmax:
                continue
            ring = RING_HI - k
            s = 1.0 + 2.1 * (v / vmax)
            halo = placed_box(R_SIPM, phi, ring_z(ring),
                              SIPM_DRAW_THICK * 1.15,
                              SIPM_SIZE * s, SIPM_SIZE * s)
            halo.cell_data["photons"] = np.full(halo.n_cells, v, dtype=float)
            blocks.append(halo)
    return pv.MultiBlock(blocks).combine() if blocks else None


def muon_entry(theta_deg, phi_deg, phi_sense="to", z0=TRACK_Z0,
               length=MUON_HALF_LEN):
    """(entry point, direction) of the drawn track: where the muon comes in."""
    u = muon_axis(theta_deg, phi_deg, phi_sense)
    return -u * length + np.array([0.0, 0.0, float(z0)]), u


def muon_arc_meshes(entry_pt, u, arc):
    """Track tube, arrow head and bloom tube for a muon `arc` cm past entry.

    Each piece has the same topology whatever the arc, which is what lets the
    animation copy a new frame straight into the meshes already on screen.
    """
    arc = max(float(arc), MUON_MIN_ARC)
    head_pt = np.asarray(entry_pt, dtype=float) + arc * u
    track = pv.Line(entry_pt, head_pt).tube(radius=0.085, n_sides=20)
    head = pv.Cone(center=head_pt - u * 0.9, direction=u, height=1.8,
                   radius=0.42, resolution=24)
    # a faint fat tube around the track, same bloom trick as the SiPMs
    bloom = pv.Line(entry_pt, head_pt).tube(radius=0.30, n_sides=16)
    return track, head, bloom


def muon_meshes(theta_deg, phi_deg, length=MUON_HALF_LEN, phi_sense="to",
                z0=TRACK_Z0):
    """The muon as a line through the detector axis at z0, plus an arrow head.

    theta is the zenith angle (0 = straight down the axis) and phi the
    azimuth, the same convention the CNN predicts.  The impact point is not
    reconstructed, so the track is drawn crossing the axis at a fixed height
    z0 -- it shows the DIRECTION, not where the muon actually crossed.

    z0 defaults to +3 cm rather than 0 because the eleven instrumented rings
    run from z = -4.9 to +14.7 cm, so the read-out part of the detector is not
    centred on the origin and a track through z = 0 sits low in it.

    The muon always goes downwards.  `phi_sense` says what phi means:

      "to"    (default) phi is the azimuth the muon TRAVELS TOWARDS: in from
              the far side high up, out through the phi side low down, so the
              track passes through the brightest SiPMs.  This is what the real
              data says -- see the fit quoted in the module docstring.

      "from"  phi is the azimuth the muon ARRIVES FROM.  The same line rotated
              180 deg in azimuth.
    """
    entry_pt, u = muon_entry(theta_deg, phi_deg, phi_sense, z0, length)
    track, head, _ = muon_arc_meshes(entry_pt, u, 2.0 * length)
    entry = pv.Sphere(radius=0.28, center=entry_pt)
    return track, head, entry


def wavelength_rgb(nm, gamma=0.8):
    """Approximate sRGB hex for a monochromatic wavelength (Bruton's map).

    Only ever called on the 420 nm scintillation peak, but written out so the
    colour follows from the wavelength instead of being a hex string that has
    to be taken on trust.  420 nm lands in the violet-blue corner of sRGB; a
    monitor cannot really show a spectral line, so this is the usual polite
    fiction, not a colorimetric statement.
    """
    nm = float(nm)
    if 380.0 <= nm < 440.0:
        rgb = [-(nm - 440.0) / 60.0, 0.0, 1.0]
    elif 440.0 <= nm < 490.0:
        rgb = [0.0, (nm - 440.0) / 50.0, 1.0]
    elif 490.0 <= nm < 510.0:
        rgb = [0.0, 1.0, -(nm - 510.0) / 20.0]
    elif 510.0 <= nm < 580.0:
        rgb = [(nm - 510.0) / 70.0, 1.0, 0.0]
    elif 580.0 <= nm < 645.0:
        rgb = [1.0, -(nm - 645.0) / 65.0, 0.0]
    elif 645.0 <= nm <= 780.0:
        rgb = [1.0, 0.0, 0.0]
    else:
        rgb = [0.0, 0.0, 0.0]
    # the eye rolls off at both ends of the visible band
    if 380.0 <= nm < 420.0:
        f = 0.3 + 0.7 * (nm - 380.0) / 40.0
    elif 700.0 < nm <= 780.0:
        f = 0.3 + 0.7 * (780.0 - nm) / 80.0
    elif 420.0 <= nm <= 700.0:
        f = 1.0
    else:
        f = 0.0
    from matplotlib.colors import to_hex
    return to_hex([(c * f) ** gamma if c > 0 else 0.0 for c in rgb])


PHOTON_COLOUR = None                          # filled in on first use


def photon_colour():
    global PHOTON_COLOUR
    if PHOTON_COLOUR is None:
        PHOTON_COLOUR = wavelength_rgb(PHOTON_NM)
    return PHOTON_COLOUR


def cylinder_span(p, d, radius=R_CYL, half_h=H_CYL / 2.0):
    """(t_in, t_out) for which p + t*d lies inside the closed cylinder.

    None when the line misses it altogether.  The side wall gives a quadratic
    in t and the two end caps give a slab in z; the inside is where the two
    intervals overlap, which is what keeps a nearly-horizontal line from being
    cut at the wall of the infinite cylinder far above the real one.
    """
    p = np.asarray(p, dtype=float)
    d = np.asarray(d, dtype=float)
    lo, hi = -np.inf, np.inf

    a = d[0] * d[0] + d[1] * d[1]
    c = p[0] * p[0] + p[1] * p[1] - radius * radius
    if a < 1e-12:                              # parallel to the axis
        if c > 0.0:
            return None
    else:
        b = 2.0 * (p[0] * d[0] + p[1] * d[1])
        disc = b * b - 4.0 * a * c
        if disc <= 0.0:
            return None
        s = np.sqrt(disc)
        lo, hi = (-b - s) / (2.0 * a), (-b + s) / (2.0 * a)

    if abs(d[2]) < 1e-12:                      # perpendicular to the axis
        if abs(p[2]) > half_h:
            return None
    else:
        t1 = (-half_h - p[2]) / d[2]
        t2 = (half_h - p[2]) / d[2]
        lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))

    return None if hi <= lo else (lo, hi)


def scintillation_photons(theta_deg, phi_deg, phi_sense="to", z0=TRACK_Z0,
                          per_cm=PHOTONS_PER_CM, seed=0, max_n=PHOTON_MAX,
                          length=MUON_HALF_LEN):
    """Photons from the muon's path through the scintillator, as four arrays.

    Returns (starts, dirs, ranges, emitted_at), or None when the track misses
    the scintillator: where each photon began, the unit vector it left on, how
    far it can go before it meets the wall or an end cap, and how far the muon
    had travelled from the top of its drawn track when it was emitted.  Kept
    as arrays rather than a mesh because the animation has to redraw them at a
    new length thirty times a second; photon_lines() turns them into geometry.

    Emission points are scattered at random along the chord the track cuts
    through the cylinder -- and only along that chord, so nothing is emitted
    in the air above it or in the boards -- and each photon leaves in a random
    isotropic direction and stops where it meets the wall or an end cap.

    The lines are what an isotropic emitter looks like, not a light-collection
    simulation: they are drawn straight, with no refraction at the wall, no
    reflection off it, no attenuation along the way and no wrapping, so the
    picture says where the light goes, not how much of it reaches a SiPM.

    The generator is seeded per event, so stepping away from an event and back
    redraws the same photons rather than reshuffling them.
    """
    u = muon_axis(theta_deg, phi_deg, phi_sense)
    p0 = np.array([0.0, 0.0, float(z0)])
    span = cylinder_span(p0, u)
    if span is None:                           # track misses the scintillator
        return None
    t_in, t_out = span
    chord = t_out - t_in                       # u is a unit vector, so cm
    n = int(round(float(per_cm) * chord))
    if n <= 0:
        return None
    n = min(n, int(max_n))

    rng = np.random.default_rng(seed)
    t = t_in + chord * rng.random(n)
    starts = p0 + np.outer(t, u)
    # isotropic: cos(polar) flat in [-1, 1], azimuth flat in [0, 2pi).  Taking
    # the polar angle itself flat would crowd the photons at the poles.
    cz = rng.uniform(-1.0, 1.0, n)
    sz = np.sqrt(np.maximum(0.0, 1.0 - cz * cz))
    az = rng.uniform(0.0, 2.0 * np.pi, n)
    dirs = np.c_[sz * np.cos(az), sz * np.sin(az), cz]

    ranges = np.empty(n)
    for i in range(n):
        sp = cylinder_span(starts[i], dirs[i])
        ranges[i] = max(sp[1], 0.0) if sp else 0.0

    # t runs from the axis crossing; the drawn track starts `length` before it
    return starts, dirs, ranges, t + float(length)


def photon_lines(starts, dirs, lengths):
    """One segment per photon, at the given lengths, as a single mesh.

    The topology never changes with `lengths`, so the animation can push a new
    set of points into the mesh it is already showing instead of building an
    actor a frame.
    """
    n = len(starts)
    pts = np.empty((2 * n, 3))
    pts[0::2] = starts
    pts[1::2] = starts + np.asarray(lengths)[:, None] * dirs
    lines = np.column_stack([np.full(n, 2), np.arange(0, 2 * n, 2),
                             np.arange(1, 2 * n, 2)]).ravel()
    return pv.PolyData(pts, lines=lines)


def muon_axis(theta_deg, phi_deg, phi_sense="to"):
    """Unit vector along a muon's direction of travel.  Always down-going.

    theta is the zenith angle and phi the azimuth; `phi_sense` says whether
    phi is the azimuth travelled towards ("to") or arrived from ("from"), as
    muon_meshes explains at length.
    """
    t, p = np.radians(theta_deg), np.radians(phi_deg)
    # unit vector pointing up-and-out along azimuth phi
    a = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    if phi_sense == "to":
        # down-going, azimuth phi: mirror the horizontal part, keep it falling
        return np.array([a[0], a[1], -a[2]])
    return -a                                      # down-going, away from phi


def trig_axis(phi_sense="to", theta=TRIG_THETA, phi=TRIG_PHI):
    """Unit vector along the accepted muon's direction of travel."""
    return muon_axis(theta, phi, phi_sense)


def trig_placements(phi_sense="to", z0=TRACK_Z0, dist=TRIG_DIST,
                    theta=TRIG_THETA, phi=TRIG_PHI):
    """(centre, outward radial unit vector, is_outer) for the four channels.

    Ordered along the muon's path: the upper pair first, outer unit first in
    each pair, so channel 0 is the one the muon meets first.
    """
    u = trig_axis(phi_sense, theta, phi)
    pivot = np.array([0.0, 0.0, float(z0)])
    half = 0.5 * (TRIG_THICK + TRIG_PAIR_GAP)
    out = []
    for side in (-1.0, +1.0):                  # -1 = entry (upper) pair
        w = side * u                           # outward from the pivot
        for outer in (True, False) if side < 0 else (False, True):
            r = dist + (half if outer else -half)
            out.append((pivot + w * r, w, outer))
    return out


def _cw_frame(centre, w, outer):
    """The 4x4 that maps the local frame -- x radial (the 1 cm axis), y
    azimuthal, z polar -- onto the channel's place in the world."""
    w = np.asarray(w, dtype=float)
    w = w / np.linalg.norm(w)
    ex = w if outer else -w
    zhat = np.array([0.0, 0.0, 1.0])
    ey = np.cross(zhat, ex)
    if np.linalg.norm(ey) < 1e-9:
        ey = np.array([0.0, 1.0, 0.0])
    ey /= np.linalg.norm(ey)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = ex, ey, np.cross(ex, ey)
    M[:3, 3] = centre
    return M


def cosmicwatch_body(centre, w, outer=True):
    """The parts of a channel that never change: the scintillator and the
    board behind it.  Static, so stepping through events does not rebuild
    the telescope.

    The scintillator carries the same pale translucent blue as the cylinder,
    so the two read as the same material; smooth shading stays off here so the
    cube keeps its edges instead of rounding them off.
    """
    M = _cw_frame(centre, w, outer)
    s = TRIG_SIZE
    parts = []

    def add(mesh, **style):
        parts.append((mesh.transform(M, inplace=False), style))

    add(pv.Cube(center=(0, 0, 0), x_length=TRIG_THICK, y_length=s, z_length=s),
        color=SCINT, opacity=TRIG_SCINT_OPACITY, smooth_shading=False,
        specular=1.0, specular_power=35, ambient=0.28, diffuse=0.75)
    add(pv.Cube(center=(CW_PCB_X, 0.0, 0.0),
                x_length=0.22, y_length=s, z_length=s),
        color=CW_PCB, smooth_shading=True, specular=0.4, specular_power=22,
        ambient=0.22)
    return parts


def trig_sipms(placements, scale=1.0, thick=SIPM_DRAW_THICK):
    """The four trigger SiPMs as ONE mesh, one per CosmicWatch board.

    Same 6 mm square as the tiles on the cylinder boards and centred on the
    board the same way, so it stands proud of both faces and reads from either
    side.  Merged into a single mesh for the same reason build_sipms is: one
    actor to add and drop per event instead of four.

    `scale` and `thick` are what build_halo does to a tile, so the same call
    also makes the bloom box.
    """
    blocks = []
    for centre, w, outer in placements:
        M = _cw_frame(centre, w, outer)
        tile = pv.Cube(center=(CW_PCB_X, 0.0, 0.0), x_length=thick,
                       y_length=SIPM_SIZE * scale, z_length=SIPM_SIZE * scale)
        blocks.append(tile.transform(M, inplace=False))
    return pv.MultiBlock(blocks).combine() if blocks else None


def cmap_top_colour(cmap):
    """The colour the brightest SiPM on the cylinder comes out, as hex.

    cmap is the GLOW ramp by default but a matplotlib name when --cmap is
    given, and the lookup for a name moved between matplotlib versions, so
    both spellings are tried.
    """
    from matplotlib.colors import to_hex
    m = cmap
    if not callable(m):
        try:
            import matplotlib
            m = matplotlib.colormaps[m]
        except Exception:
            from matplotlib import cm
            m = cm.get_cmap(m)
    return to_hex(m(1.0))


def compass_meshes():
    """A brass bearing ring under the detector, with the four boards marked.

    This exists so that 'which way is phi' is answerable from the picture
    instead of from the source code.
    """
    z = -0.5 * H_CYL - 1.9
    rr = R_PCB * 1.55
    ang = np.radians(np.arange(0.0, 361.0, 3.0))
    pts = np.c_[rr * np.cos(ang), rr * np.sin(ang), np.full(ang.size, z)]
    parts = [(pv.lines_from_points(pts).tube(radius=0.055, n_sides=12),
              dict(color=BRASS, smooth_shading=True, specular=1.0,
                   specular_power=70, ambient=0.20))]
    for deg in range(0, 360, 15):
        major = (deg % 90) == 0
        ln = 0.55 if major else 0.25
        parts.append((placed_box(rr + ln / 2, float(deg), z,
                                 ln, 0.11 if major else 0.06, 0.11),
                      dict(color=BRASS if major else BRASS_DARK,
                           smooth_shading=True, specular=0.9,
                           specular_power=60)))
    return parts, z, rr


def compass_labels():
    """(positions, strings) for the A/B/C/D board tags on the bearing ring."""
    _, z, rr = compass_meshes()
    pos, txt = [], []
    for col in COLS:
        p = np.radians(COL_PHI[col])
        pos.append([(rr + 1.2) * np.cos(p), (rr + 1.2) * np.sin(p), z])
        txt.append("%s  %d°" % (col, int(COL_PHI[col])))
    return np.asarray(pos), txt


def axes_triad_meshes(length=2.3):
    """A small x/y/z triad parked outside the bearing ring.

    VTK's orientation-marker widget is drawn by a renderer of its own, and in
    a multi-viewport window it intermittently comes out zoomed to a solid
    block, so the triad is ordinary scene geometry instead.  It sits at the
    ring, right under the detector, which is where it is wanted anyway.
    """
    _, z, _rr = compass_meshes()
    o = np.array([0.0, 0.0, z - 3.4])            # directly below the detector
    parts, pos, txt = [], [], []
    for d, col, lab in (((1, 0, 0), "#ff6b5e", "x"),
                        ((0, 1, 0), "#7ee06a", "y"),
                        ((0, 0, 1), "#6fa8ff", "z")):
        parts.append((pv.Arrow(start=o, direction=d, scale=length,
                               tip_length=0.26, tip_radius=0.085,
                               shaft_radius=0.028),
                      dict(color=col, smooth_shading=True, ambient=0.55,
                           diffuse=0.55, specular=0.7, specular_power=40)))
        pos.append(o + np.array(d, dtype=float) * (length * 1.22))
        txt.append(lab)
    return parts, np.asarray(pos), txt


def phi_pointer(phi_deg):
    """The gold needle on the bearing ring, laid along azimuth phi."""
    _, z, rr = compass_meshes()
    p = np.radians(phi_deg)
    d = (np.cos(p), np.sin(p), 0.0)
    shaft = pv.Line((0, 0, z), (rr * 0.93 * d[0], rr * 0.93 * d[1], z)) \
              .tube(radius=0.07, n_sides=12)
    tip = pv.Cone(center=(rr * 0.99 * d[0], rr * 0.99 * d[1], z),
                  direction=d, height=0.9, radius=0.26, resolution=20)
    return shaft, tip, z, rr


# ----------------------------------------------------------------------------
# the scene
# ----------------------------------------------------------------------------
def light_rig(pl):
    pl.remove_all_lights()
    for pos, inten, col in (((1.0, -0.7, 0.8), 1.25, "#fff4e4"),   # key
                            ((-1.0, -0.4, 0.3), 0.70, "#8fc4f0"),  # cool fill
                            ((-0.5, 1.0, -0.6), 0.65, "#ffd39a"),  # warm rim
                            ((0.2, 0.3, -1.0), 0.40, "#9fd8e8")):  # from below
        v = np.array(pos, dtype=float)
        v = v / np.linalg.norm(v) * 60.0
        pl.add_light(pv.Light(position=tuple(v), focal_point=(0, 0, 0),
                              color=col, intensity=inten,
                              light_type="scene light"))


def light_azimuth(counts):
    """Azimuth of the light centroid, from the four board totals.

    In the training MC this is exactly the generated phi, so it is the one
    number that says whether the lit SiPMs and the drawn track agree.
    """
    tot = counts.sum(axis=1)
    b = np.radians([COL_PHI[c] for c in COLS])
    x, y = float((tot * np.cos(b)).sum()), float((tot * np.sin(b)).sum())
    if x == 0.0 and y == 0.0:
        return float("nan")
    return np.degrees(np.arctan2(y, x)) % 360.0


# ----------------------------------------------------------------------------
# the two flat side panels
#
# These are matplotlib figures drawn as the background image of their own
# viewport, rather than as geometry in the scene.  That is what keeps them
# square to the screen and the same size no matter how the detector is
# rotated, zoomed or panned.
# ----------------------------------------------------------------------------
# The window is split into three viewports: the flat 4 x 11 map on the left,
# the 3-D scene in the middle, the radiograph on the right.  The panels are
# drawn as that viewport's background image, which is what keeps them square
# to the screen and fixed while the detector is rotated, zoomed or panned.
PANEL_WEIGHTS = (1.0, 2.1, 1.5)
PANEL_DPI = 100.0

# the trigger acceptance drawn as the shaded wedge on the radiograph
BAND_THETA = (40.0, 60.0)
BAND_PHI = (-8.5, 8.5)
BAND_FILL = "#6ba6dd"
EVENT_DOT = "#ff5b39"



def _panel_figure(figsize, dpi=100):
    from matplotlib.figure import Figure
    fig = Figure(figsize=figsize, dpi=dpi)
    # Opaque, deliberately.  A transparent panel composites over whatever the
    # viewport held last frame instead of covering it, which is what doubled
    # up the titles and the colour-bar ticks.
    fig.patch.set_facecolor(BG_LOW)
    fig.patch.set_alpha(1.0)
    return fig


PANEL_FS_REF = 3.6           # panel width, in inches, the base sizes suit


def _fs(figsize, base):
    """Font size for a panel of this width.

    The panels are built at their viewport's pixel size, so point sizes are
    pixel sizes: text has to shrink with a dragged-in window or it clips, and
    grow with a big one or it is unreadable.  Clamped at both ends."""
    return float(np.clip(base * figsize[0] / PANEL_FS_REF, 5.0,
                         base * 2.6))


def _style_axes(ax, colour=HUD, fs=8.0):
    ax.set_facecolor("none")
    for sp in ax.spines.values():
        sp.set_color(colour)
        sp.set_linewidth(0.8)
    ax.tick_params(colors=colour, labelsize=fs, length=3, width=0.8)
    ax.yaxis.label.set_color(colour)
    ax.xaxis.label.set_color(colour)
    ax.title.set_color(colour)


def panel_grid_figure(counts, cmap, figsize, title=None, vmax=None):
    """The 4 x 11 photon map, ring 15 at the top, exactly as the data is
    stored.  Cells carry their value; the hottest one gets a cyan box."""
    from matplotlib.patches import Rectangle
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    img = np.asarray(counts).T                  # (ring, column)
    top = float(vmax) if vmax else max(1.0, float(img.max()))
    fig = _panel_figure(figsize)
    ax = fig.add_axes([0.17, 0.045, 0.60, 0.845])
    _style_axes(ax, fs=_fs(figsize, 8))
    norm = Normalize(0.0, top)
    ax.imshow(img, cmap=cmap, norm=norm, aspect="auto", origin="upper",
              interpolation="nearest")

    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels(COLS, fontsize=_fs(figsize, 11))
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(NRING_DATA))
    ax.set_yticklabels([str(RING_HI - k) for k in range(NRING_DATA)],
                       fontsize=_fs(figsize, 8))
    ax.set_ylabel("detector ring", fontsize=_fs(figsize, 9))
    ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, NRING_DATA, 1), minor=True)
    ax.grid(which="minor", color=BG_HIGH, linewidth=0.8)
    ax.tick_params(which="minor", length=0)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    for r in range(NRING_DATA):
        for c in range(len(COLS)):
            v = img[r, c]
            rgb = sm.to_rgba(v)[:3]
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            ax.text(c, r, "%d" % round(v), ha="center", va="center",
                    fontsize=_fs(figsize, 7.5),
                    color="#101010" if lum > 0.6 else "#e8f4f8")
    if img.max() > 0:                            # nothing to point at when empty
        hot = np.unravel_index(int(np.argmax(img)), img.shape)
        ax.add_patch(Rectangle((hot[1] - 0.5, hot[0] - 0.5), 1, 1, fill=False,
                               edgecolor=MUON, linewidth=2.0))

    cax = fig.add_axes([0.81, 0.045, 0.045, 0.845])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("photons", color=HUD, fontsize=_fs(figsize, 9))
    cb.ax.tick_params(colors=HUD, labelsize=_fs(figsize, 7.5))
    cb.outline.set_edgecolor(HUD)
    if title:
        fig.text(0.5, 0.965, title, ha="center", va="top", color=HUD,
                 fontsize=_fs(figsize, 9), family="monospace")
    return fig


def panel_radiograph_figure(theta=None, phi=None, figsize=(4.5, 4.5)):
    """theta vs phi on a polar plot, with the trigger acceptance shaded.

    One red dot, for the event on screen -- this is a per-event readout of
    where the muon sits in the acceptance, not a population plot.
    """
    fig = _panel_figure(figsize)
    w, h = fig.get_size_inches()
    side = min(0.78, 0.78 * w / h)               # keep the dial round
    ax = fig.add_axes([0.5 - side * h / w / 2.0, 0.5 - side / 2.0,
                       side * h / w, side], projection="polar")
    ax.set_facecolor("none")
    ax.set_ylim(0, 80)
    ax.set_rgrids(range(10, 80, 10),
                  labels=["%d" % v for v in range(10, 80, 10)],
                  color="#e8f4f8", fontsize=_fs(figsize, 7.5),
                  angle=112.5)
    ax.set_thetagrids(range(0, 360, 45),
                      labels=["%d°" % v for v in range(0, 360, 45)],
                      color=HUD, fontsize=_fs(figsize, 8))
    ax.grid(color="#2c4356", linewidth=0.7)
    ax.spines["polar"].set_color(HUD)

    a = np.radians(np.linspace(BAND_PHI[0], BAND_PHI[1], 60))
    ax.fill_between(a, BAND_THETA[0], BAND_THETA[1], color=BAND_FILL,
                    alpha=0.45, edgecolor=BAND_FILL, linewidth=1.0)

    if theta is not None and phi is not None:
        ax.plot([np.radians(phi)], [min(float(theta), 80.0)], "o",
                color=EVENT_DOT, markersize=7,
                markeredgecolor="#ffd0c4", markeredgewidth=0.8, zorder=5)

    fig.text(0.5, 0.5 + side / 2.0 + 0.082,
             u"%s vs %s radiograph" % (THETA, PHI),
             ha="center", va="bottom", color=HUD,
             fontsize=_fs(figsize, 9), family="monospace")
    fig.text(0.5, 0.5 + side / 2.0 + 0.050,
             u"trigger: %g-%g\u00b0 %s,  %+g to %+g\u00b0 %s"
             % (BAND_THETA[0], BAND_THETA[1], THETA,
                BAND_PHI[0], BAND_PHI[1], PHI),
             ha="center", va="bottom", color="#5d7d8a",
             fontsize=_fs(figsize, 7.5), family="monospace")
    return fig


def setup_text(nev, trig_dist, track_z0, triggers=True):
    """The panel shown before any event is loaded."""
    lines = ["DETECTOR SETUP",
             "",
             "scint    cylinder %.1f cm dia x %.0f cm" % (2 * R_CYL, H_CYL),
             u"boards   %s at %s\u00b0"
             % ("/".join(COLS), "/".join("%.0f" % COL_PHI[c] for c in COLS)),
             "SiPMs    %d per board, %.2f cm pitch" % (N_RING, SIPM_PITCH),
             "         rings %d-%d read out (%d)"
             % (RING_LO, RING_HI, NRING_DATA)]
    if triggers:
        rho = trig_dist * np.sin(np.radians(TRIG_THETA))
        dt = np.degrees(np.arctan(TRIG_SIZE / 2.0 / trig_dist))
        dp = np.degrees(np.arctan(TRIG_SIZE / 2.0 / rho))
        lines += ["trigger  %d CosmicWatch, %g x %g x %g cm"
                  % (N_TRIG, TRIG_SIZE, TRIG_SIZE, TRIG_THICK),
                  u"         %.1f cm from (0,0,%g) on %s %g\u00b0 %s %g\u00b0"
                  % (trig_dist, track_z0, THETA, TRIG_THETA, PHI, TRIG_PHI),
                  u"         %s %.1f-%.1f\u00b0, %s %+.1f to %+.1f\u00b0"
                  % (THETA, TRIG_THETA - dt, TRIG_THETA + dt, PHI, -dp, dp)]
    lines += ["%d event%s loaded -- press Right for the first"
              % (nev, "" if nev == 1 else "s")]
    return "\n".join(lines)


def hud_text(ev, counts, phi_sense="to"):
    tot = int(counts.sum())
    hot = np.unravel_index(int(np.argmax(counts)), counts.shape)
    bt = counts.sum(axis=1)
    return (u"TrgID %d\n"
            u"%s           %7.2f\u00b0\n"
            u"%s           %7.2f\u00b0   (muon %s %s)\n"
            u"confidence  %7.3f\n"
            u"photons     %7d\n"
            u"hottest     %s ring %d  (%d)\n"
            u"boards      %s\n"
            u"light at    %7.2f\u00b0%s"
            % (ev["trgid"], THETA, ev["theta"], PHI, ev["phi"],
               "arrives from" if phi_sense == "from" else "travels to", PHI,
               ev["conf"], tot,
               COLS[hot[0]], RING_HI - hot[1], int(counts[hot]),
               "  ".join("%s %d" % (c, bt[i]) for i, c in enumerate(COLS)),
               light_azimuth(counts),
               "" if ev.get("trig") is None else
               "\ntrigger  " + "  ".join(
                   "%d%s" % (v, "*" if v > TRIG_HOT else "")
                   for v in ev["trig"].astype(int))))


class Display:
    """Holds the plotter and swaps the per-event actors in and out."""

    def __init__(self, events, cmap=GLOW, zmax=None, logscale=False,
                 off_screen=False, window_size=(1280, 960), halo=True,
                 phi_sense="to", compass=True, track_z0=TRACK_Z0,
                 triggers=True, trig_dist=TRIG_DIST, panels=True,
                 savedir=".", show_photons=False,
                 photons_per_cm=PHOTONS_PER_CM, animate=False,
                 speed=ANIM_SPEED):
        self.events = events
        self.i = -1                              # -1 = the bare detector
        self.cmap = cmap
        self.zmax = zmax
        self.log = logscale
        self.halo = halo
        self.phi_sense = phi_sense
        self.track_z0 = float(track_z0)
        self.triggers = triggers
        self.trig_dist = float(trig_dist)
        self.compass = compass
        self.panels = panels
        self.show_photons = bool(show_photons)
        self.photons_per_cm = float(photons_per_cm)
        # An animation needs a window with a clock in it.  A screenshot or a
        # web page gets the finished track instead, not frame zero of one.
        self.animate = bool(animate) and not off_screen
        self.speed = float(speed)
        self._anim = None                        # the flight in progress
        self.window_size = tuple(window_size)
        self._tmp = tempfile.mkdtemp(prefix="evd3d_")
        self._panel_actor = {}                   # textured plane per side
        self._panel_size = {}
        self._saved = 0
        self._last_ws = tuple(window_size)
        self.savedir = savedir
        self.dynamic = []                        # actors to clear each event

        shape = (1, 3) if panels else (1, 1)
        kw = dict(col_weights=list(PANEL_WEIGHTS), border=False) if panels \
            else {}
        self.pl = pv.Plotter(off_screen=off_screen, window_size=window_size,
                             lighting="none", shape=shape, **kw)
        for c in range(shape[1]):
            self.pl.subplot(0, c)
            self.pl.set_background(BG_LOW, top=BG_HIGH)
        self.mid()
        for mesh, style in build_detector():
            self.pl.add_mesh(mesh, **style)
        if self.compass:
            for mesh, style in compass_meshes()[0]:
                self.pl.add_mesh(mesh, reset_camera=False, **style)
            pos, txt = compass_labels()
            self.pl.add_point_labels(pos, txt,
                                     font_size=self.hud_size(11),
                                     text_color=BRASS, shape=None,
                                     show_points=False, always_visible=True)
        light_rig(self.pl)

        # transparency has to be resolved properly or the SiPMs vanish behind
        # the scintillator from some angles
        try:
            self.pl.enable_depth_peeling(number_of_peels=8)
        except Exception:
            pass
        # SSAA renders at double size, which makes VTK drop the side
        # viewports' background images entirely.  FXAA costs a little sharpness
        # but composites correctly, so it is what the panelled layout uses.
        for mode in (("fxaa",) if panels else ("ssaa", "fxaa")):
            try:
                self.pl.enable_anti_aliasing(mode)
                break
            except Exception:
                continue

        # the orientation marker, the HUD and the colour bar all live in the
        # middle band of the window, between the two flat panels, so they read
        # as labels on the detector rather than as window furniture
        if self.triggers:
            for c, w, outer in trig_placements(phi_sense, self.track_z0,
                                               self.trig_dist):
                for mesh, style in cosmicwatch_body(c, w, outer):
                    self.pl.add_mesh(mesh, reset_camera=False, **style)

        tri, tpos, ttxt = axes_triad_meshes()
        for mesh, style in tri:
            self.pl.add_mesh(mesh, reset_camera=False, **style)
        self.pl.add_point_labels(tpos, ttxt, font_size=self.hud_size(11),
                                 text_color=HUD, shape=None,
                                 show_points=False, always_visible=True)
        # The timer goes in before the first draw, so that a VTK build without
        # one falls back to finished tracks rather than to frozen stubs.
        if self.animate:
            try:
                self.pl.add_timer_event(max_steps=ANIM_MAX_STEPS,
                                        duration=ANIM_FRAME_MS,
                                        callback=self._anim_frame)
            except Exception:
                self.animate = False

        for key in ("Right", "n"):
            self.pl.add_key_event(key, self.next)
        for key in ("Left", "p"):
            self.pl.add_key_event(key, self.prev)
        self.pl.add_key_event("Home", self.setup)
        self.pl.add_key_event("a", self.replay)
        for key in ("Return", "KP_Enter"):
            self.pl.add_key_event(key, self.save_view)
        self.draw()
        self.home_view()
        self.pl.add_key_event("r", self.home_view)

        # the panel figures are built at the viewport's pixel size, so a
        # resized window needs them rebuilt or they letterbox and leave stale
        # pixels down the sides
        if not off_screen and self.panels:
            try:
                self.pl.iren.add_observer("ModifiedEvent", self._on_resize)
            except Exception:
                pass

    def _on_resize(self, *args):
        try:
            ws = tuple(self.pl.window_size)
        except Exception:
            return
        if max(abs(ws[0] - self._last_ws[0]), abs(ws[1] - self._last_ws[1])) < 8:
            return
        self._last_ws = ws
        self.draw()

    # -- viewports ---------------------------------------------------------
    def add_hud(self, text, position, base, colour, shadow=False):
        """A HUD text actor, with a little leading -- DejaVu Sans Mono sets
        its lines tighter than VTK's courier did and they touch otherwise.

        render=False matters: add_text() would otherwise draw the block at the
        default line spacing, and the next render would redraw it at 1.28 and
        the whole block would visibly jump.  Nothing is painted until draw()
        renders once at the end.
        """
        actor = self.pl.add_text(text, position=position,
                                 font_size=self.hud_size(base), color=colour,
                                 shadow=shadow, render=False, **hud_font())
        try:
            actor.GetTextProperty().SetLineSpacing(1.28)
        except Exception:
            pass
        return actor

    def hud_size(self, base):
        """VTK text is sized in points, so it has to be scaled by hand or it
        shrinks away on a big window."""
        try:
            h = self.pl.window_size[1]
        except Exception:
            h = self.window_size[1]
        return int(round(float(np.clip(base * h / 950.0,
                                       max(7.0, base * 0.7),
                                       max(8.0, base * 2.6)))))

    def mid(self):
        """Make the 3-D viewport active.  Every camera, light, actor and text
        call below acts on whichever renderer is current, so this has to be
        the first thing any of them does."""
        if self.panels:
            self.pl.subplot(0, 1)

    def panel_px(self, which):
        """Pixel size of a side viewport, so the matplotlib figure can be made
        at exactly that aspect and fill it without letterboxing."""
        try:
            w, h = self.pl.window_size
        except Exception:
            w, h = self.window_size
        tot = float(sum(PANEL_WEIGHTS))
        return (w * PANEL_WEIGHTS[0 if which == "left" else 2] / tot, float(h))

    def panel_figsize(self, which):
        px = self.panel_px(which)
        return (px[0] / PANEL_DPI, px[1] / PANEL_DPI)

    def show_panel(self, col, fig):
        """Draw a panel figure onto a textured plane filling its viewport.

        Not pyvista's background-image machinery: that puts the panel on a
        renderer of its own which does not clear its viewport, so anything the
        previous frame left outside the new image stayed on screen -- the
        ghosting after a resize.  An ordinary actor in the viewport's own
        renderer is erased and redrawn every frame like everything else.
        """
        path = os.path.join(self._tmp, "panel%d.png" % col)
        fig.savefig(path, dpi=PANEL_DPI, facecolor=BG_LOW, transparent=False)
        tex = pv.read_texture(path)
        w, h = tex.dimensions[:2] if hasattr(tex, "dimensions") else (1, 1)
        try:
            w, h = tex.to_image().dimensions[:2]
        except Exception:
            pass

        self.pl.subplot(0, col)
        actor = self._panel_actor.get(col)
        if actor is None:
            plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
                             i_size=float(w), j_size=float(h),
                             i_resolution=1, j_resolution=1)
            actor = self.pl.add_mesh(plane, texture=tex, lighting=False,
                                     ambient=1.0, diffuse=0.0, specular=0.0,
                                     show_scalar_bar=False, reset_camera=False)
            self._panel_actor[col] = actor
            self._panel_size[col] = (float(w), float(h))
            # the panel must not respond to the mouse at all
            try:
                self.pl.renderer.InteractiveOff()
            except Exception:
                pass
        else:
            try:
                actor.SetTexture(tex)
            except Exception:
                actor.texture = tex
            if self._panel_size.get(col) != (float(w), float(h)):
                self.pl.remove_actor(actor, render=False)
                del self._panel_actor[col]
                self.mid()
                return self.show_panel(col, fig)
        self.fit_panel(col)
        self.mid()

    def fit_panel(self, col):
        """Parallel camera square to the plane, scaled so it fills the
        viewport exactly -- the figure is already made at the viewport's
        aspect, so this is a straight 1:1 fit."""
        size = self._panel_size.get(col)
        if size is None:
            return
        w, h = size
        self.pl.subplot(0, col)
        cam = self.pl.camera
        cam.enable_parallel_projection()
        cam.SetFocalPoint(0.0, 0.0, 0.0)
        cam.SetPosition(0.0, 0.0, max(w, h) * 2.0)
        cam.SetViewUp(0.0, 1.0, 0.0)
        try:
            vx0, vy0, vx1, vy1 = self.pl.renderer.viewport
            ww, hh = self.pl.window_size
            aspect = max(1e-6, (vx1 - vx0) * ww / max(1e-6, (vy1 - vy0) * hh))
        except Exception:
            aspect = w / h
        cam.parallel_scale = max(h / 2.0, (w / 2.0) / aspect)
        self.pl.reset_camera_clipping_range()

    def trig_placements(self):
        return trig_placements(self.phi_sense, self.track_z0, self.trig_dist)

    def home_view(self):
        """Slightly raised view framed on the detector, not on the track.

        The camera sits on the -y axis looking along +y with z up, so +x runs
        exactly left-to-right across the screen and +z exactly up.  Screen
        right is (direction x view-up), so x lands on the horizontal only when
        the camera has no x offset; any azimuthal swing would tilt it.  The
        small +z offset keeps a little of the top of the cylinder in view.

        reset_camera() would fit the muon line as well, which is twice as long
        as the cylinder and would leave the detector tiny in the middle.
        """
        self.mid()
        self.pl.camera_position = [(0.0, -1.77, 0.58), (0, 0, 0), (0, 0, 1)]
        z_top = ring_z(N_RING - 1) + 5.1          # the connector end
        z_bot = z_top - PCB_LEN
        rad = R_PCB * 2.0
        if self.compass:
            _, z_c, r_c = compass_meshes()
            rad = max(rad, r_c + 1.6)
            z_bot = min(z_bot, z_c - 6.2)
        if self.triggers:
            for c, _w, _o in self.trig_placements():
                rad = max(rad, abs(c[0]) + 4.0, abs(c[1]) + 4.0)
                z_bot = min(z_bot, c[2] - 6.0)
                z_top = max(z_top, c[2] + 4.0)
        self.pl.reset_camera(bounds=(-rad, rad, -rad, rad, z_bot, z_top))
        self.pl.camera.zoom(1.02)
        self.pl.render()

    # -- event switching ---------------------------------------------------
    # The index runs -1, 0, 1, ... len-1, where -1 is the empty detector.  It
    # clamps rather than wraps: stepping past the last event should not drop
    # you back at the first without noticing.
    def next(self):
        if self.i < len(self.events) - 1:
            self.i += 1
            self.draw()

    def prev(self):
        if self.i > -1:
            self.i -= 1
            self.draw()

    def setup(self):
        self.i = -1
        self.draw()

    def draw_panels(self, ev=None, counts=None, vmax=None):
        """Left: the 4 x 11 map.  Right: the radiograph with one dot for the
        event on screen.  Both are viewport backgrounds, so the camera never
        touches them."""
        if not self.panels:
            return
        if counts is None:
            counts = np.zeros((len(COLS), NRING_DATA))
        title = (u"TrgID %d   %s %.1f   %s %.1f   confidence %.3f"
                 % (ev["trgid"], THETA, ev["theta"], PHI, ev["phi"],
                    ev["conf"])
                 if ev else "no event loaded")
        self.show_panel(0, panel_grid_figure(counts, self.cmap,
                                             self.panel_figsize("left"),
                                             title=title, vmax=vmax))
        self.show_panel(2, panel_radiograph_figure(
            ev["theta"] if ev else None, ev["phi"] if ev else None,
            self.panel_figsize("right")))

    def draw_panels(self, ev=None, counts=None, vmax=None):
        """Left: the 4 x 11 map.  Right: the radiograph with one dot for the
        event on screen.  Both are viewport backgrounds, so the camera never
        touches them."""
        if not self.panels:
            return
        if counts is None:
            counts = np.zeros((len(COLS), NRING_DATA))
        title = (u"TrgID %d   %s %.1f   %s %.1f   confidence %.3f"
                 % (ev["trgid"], THETA, ev["theta"], PHI, ev["phi"],
                    ev["conf"])
                 if ev else "no event loaded")
        self.show_panel(0, panel_grid_figure(counts, self.cmap,
                                             self.panel_figsize("left"),
                                             title=title, vmax=vmax))
        self.show_panel(2, panel_radiograph_figure(
            ev["theta"] if ev else None, ev["phi"] if ev else None,
            self.panel_figsize("right")))

    def draw(self):
        self.mid()
        for a in self.dynamic:
            try:
                self.pl.remove_actor(a, render=False)
            except Exception:
                pass
        self.dynamic = []

        if self.i < 0:
            self.draw_setup()
            return

        ev = self.events[self.i]
        counts = ev["counts"]
        vmax = self.zmax if self.zmax else max(float(counts.max()), 1.0)
        clim = (max(1.0, 0.0) if self.log else 0.0, vmax)

        self.draw_panels(ev, counts, vmax if self.zmax else None)

        # pyvista reuses a scalar bar that already carries this title and
        # keeps its old range, so the bar has to go before the tiles are
        # re-added or it shows the previous event's scale
        try:
            self.pl.remove_scalar_bar("photons", render=False)
        except Exception:
            pass

        tiles = build_sipms(counts)
        self.dynamic.append(self.pl.add_mesh(
            tiles, scalars="photons", cmap=self.cmap, clim=clim,
            reset_camera=False,
            log_scale=self.log, smooth_shading=False, specular=0.55,
            specular_power=25, ambient=0.30, diffuse=0.85,
            show_scalar_bar=True,
            scalar_bar_args=dict(title="photons", color=HUD, n_labels=6,
                                 vertical=True, width=0.035, height=0.42,
                                 position_x=0.875, position_y=0.30,
                                 title_font_size=self.hud_size(12),
                                 label_font_size=self.hud_size(10),
                                 fmt="%.0f")))

        if self.halo:
            h = build_halo(counts)
            if h is not None:
                self.dynamic.append(self.pl.add_mesh(
                    h, scalars="photons", cmap=self.cmap, clim=clim,
                    reset_camera=False,
                    opacity=0.22, show_scalar_bar=False, ambient=1.0,
                    diffuse=0.0, specular=0.0))

        # The trigger SiPMs are not read out, so there is no count to map:
        # they simply come up at the top of the ramp, the colour the brightest
        # tile on the cylinder is wearing, with the tiles' own lighting so the
        # two match under every light in the rig.
        if self.triggers:
            lit = cmap_top_colour(self.cmap)
            places = self.trig_placements()
            m = trig_sipms(places)
            if m is not None:
                self.dynamic.append(self.pl.add_mesh(
                    m, color=lit, reset_camera=False, smooth_shading=False,
                    specular=0.55, specular_power=25, ambient=0.30,
                    diffuse=0.85, show_scalar_bar=False))
                if self.halo:
                    self.dynamic.append(self.pl.add_mesh(
                        trig_sipms(places, scale=TRIG_HALO_SCALE,
                                   thick=SIPM_DRAW_THICK * 1.15),
                        color=lit, opacity=0.22, reset_camera=False,
                        ambient=1.0, diffuse=0.0, specular=0.0,
                        show_scalar_bar=False))

        if self.compass:
            shaft, tip, _, _ = phi_pointer(ev["phi"])
            for m in (shaft, tip):
                self.dynamic.append(self.pl.add_mesh(
                    m, color=BRASS_LIT, smooth_shading=True,
                    reset_camera=False, ambient=0.55, diffuse=0.7,
                    specular=1.0, specular_power=70, show_scalar_bar=False))

        # The muon and its light are the two things that move.  Animated, they
        # are put on screen at the start of the flight and grown in place by
        # the timer; otherwise the same meshes are simply built finished.
        full_arc = 2.0 * MUON_HALF_LEN
        arc0 = MUON_MIN_ARC if self.animate else full_arc
        entry_pt, u = muon_entry(ev["theta"], ev["phi"], self.phi_sense,
                                 self.track_z0)

        photons = None
        if self.show_photons:
            photons = scintillation_photons(ev["theta"], ev["phi"],
                                            phi_sense=self.phi_sense,
                                            z0=self.track_z0,
                                            per_cm=self.photons_per_cm,
                                            seed=int(ev["trgid"]))
        # Animated, the photons have no actor yet: it is made in the first
        # frame that has light in it.  Drawing them at zero length instead
        # would lay every emission point down at once, and the chord would
        # show as a violet line through the cylinder before the muon arrived.
        ph_mesh = None
        if photons is not None and not self.animate:
            starts, dirs, ranges, _ = photons
            ph_mesh = photon_lines(starts, dirs, ranges)
            self._add_photons(ph_mesh)

        track, head, bloom = muon_arc_meshes(entry_pt, u, arc0)
        entry = pv.Sphere(radius=0.28, center=entry_pt)
        for m, op in ((track, 1.0), (head, 1.0), (entry, 0.9)):
            self.dynamic.append(self.pl.add_mesh(
                m, color=MUON, smooth_shading=True, opacity=op,
                reset_camera=False,
                ambient=0.75, diffuse=0.5, specular=1.0, specular_power=60,
                show_scalar_bar=False))
        self.dynamic.append(self.pl.add_mesh(
            bloom, color=MUON, opacity=0.10, ambient=1.0, diffuse=0.0,
            reset_camera=False, show_scalar_bar=False))

        if self.animate:
            self._anim = dict(t0=time.time(), entry=entry_pt, u=u,
                              full=full_arc, track=track, head=head,
                              bloom=bloom, photons=photons, ph_mesh=ph_mesh)

        self.dynamic.append(self.add_hud(
            hud_text(ev, counts, self.phi_sense), "upper_left", HUD_BASE,
            HUD, shadow=True))
        foot = "[%d/%d]  %s   (Right = next, Left = previous)" \
            % (self.i + 1, len(self.events), ev["name"])
        self.dynamic.append(self.add_hud(foot, "lower_left", FOOT_BASE,
                                         "#5d7d8a"))
        self.pl.render()

    # -- the flight ---------------------------------------------------------
    def _add_photons(self, mesh):
        """Put a photon mesh on screen, in the 420 nm line style."""
        self.dynamic.append(self.pl.add_mesh(
            mesh, color=photon_colour(), opacity=PHOTON_OPACITY,
            line_width=PHOTON_WIDTH, lighting=False,
            reset_camera=False, show_scalar_bar=False))

    def _anim_frame(self, step=0):
        """One frame of --animate: the track grows, the light follows it out.

        Driven off the wall clock rather than off the frame count, so --speed
        stays centimetres per second even when the frame rate sags.
        """
        a = self._anim
        if a is None:
            return
        arc = self.speed * (time.time() - a["t0"])
        done = arc >= a["full"]
        arc = min(arc, a["full"])

        track, head, bloom = muon_arc_meshes(a["entry"], a["u"], arc)
        a["track"].copy_from(track)
        a["head"].copy_from(head)
        a["bloom"].copy_from(bloom)

        if a["photons"] is not None:
            starts, dirs, ranges, emitted_at = a["photons"]
            # The photons travel at the muon's own speed, so how far one has
            # got is simply how far the muon has come since emitting it --
            # negative for light not emitted yet, and clipped at the wall.
            grown = np.clip(arc - emitted_at, 0.0, ranges)
            lit = grown > 0.0                    # the rest do not exist yet
            if lit.any():
                frame = photon_lines(starts[lit], dirs[lit], grown[lit])
                if a["ph_mesh"] is None:
                    a["ph_mesh"] = frame
                    self._add_photons(frame)
                else:
                    a["ph_mesh"].copy_from(frame)

        if done:
            self._anim = None                    # leaves the finished track up
        self.pl.render()

    def replay(self):
        """Fly the muon through again, for the event already on screen."""
        if self.animate and self.i >= 0:
            self.draw()

    def draw_setup(self):
        """The bare detector: no muon, no light, no phi needle."""
        self.mid()
        self.draw_panels(None, None)
        dark = build_sipms(np.zeros((len(COLS), NRING_DATA)))
        self.dynamic.append(self.pl.add_mesh(
            dark, color=DEAD, reset_camera=False, smooth_shading=True,
            specular=0.3, ambient=0.25, show_scalar_bar=False))
        if self.triggers:
            m = trig_sipms(self.trig_placements())
            if m is not None:
                self.dynamic.append(self.pl.add_mesh(
                    m, color=DEAD, reset_camera=False, smooth_shading=True,
                    specular=0.3, ambient=0.25, show_scalar_bar=False))


        self.dynamic.append(self.add_hud(
            setup_text(len(self.events), self.trig_dist, self.track_z0,
                       self.triggers), "upper_left", HUD_BASE - 1, HUD,
            shadow=True))
        self.dynamic.append(self.add_hud(
            "[setup]   no event loaded   (Right = first event)",
            "lower_left", FOOT_BASE, "#5d7d8a"))
        self.pl.render()

    # -- outputs -----------------------------------------------------------
    def save_view(self):
        """Write what is on screen to PNG and PDF, side by side.

        The PDF wraps the rendered pixels rather than going through GL2PS:
        vector export drops the viewport background images, which would lose
        both side panels, and the point of the button is to capture exactly
        what the window shows.
        """
        stem = self.events[self.i]["name"] if self.i >= 0 else "setup"
        stem = os.path.join(self.savedir, stem)
        path = stem
        n = 0
        while os.path.exists(path + ".png") or os.path.exists(path + ".pdf"):
            n += 1
            path = "%s_%d" % (stem, n)
        self.pl.screenshot(path + ".png")
        try:
            from matplotlib.figure import Figure
            from matplotlib.image import imread
            arr = imread(path + ".png")
            h, w = arr.shape[:2]
            fig = Figure(figsize=(w / 150.0, h / 150.0), dpi=150)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(arr, interpolation="nearest")
            ax.set_axis_off()
            fig.savefig(path + ".pdf", dpi=150, facecolor=BG_LOW)
            print("saved %s.png and %s.pdf" % (path, path))
        except Exception as exc:                 # PNG is still on disk
            print("saved %s.png (PDF failed: %s)" % (path, exc))
        self._saved += 1
        self.draw()          # the capture leaves a stale frame in the panels

    def show(self):
        self.pl.show(title="SiPM cylinder -- event display")

    def screenshot(self, path):
        self.pl.screenshot(path)

    def html(self, path):
        self.pl.trame.export_html(path)


# ----------------------------------------------------------------------------
def write_index(outdir, events, pages):
    """A small landing page listing every exported event."""
    rows = "\n".join(
        '<tr><td><a href="%s">%s</a></td><td>%d</td><td>%.2f</td>'
        '<td>%.2f</td><td>%.3f</td><td>%d</td></tr>'
        % (os.path.basename(pg), ev["name"], ev["trgid"], ev["theta"],
           ev["phi"], ev["conf"], int(ev["counts"].sum()))
        for ev, pg in zip(events, pages))
    html = """<!doctype html>
<title>SiPM cylinder event display</title>
<style>
 body{background:#06090d;color:#8fe8f2;font:14px/1.6 ui-monospace,Menlo,
      Consolas,monospace;margin:0;padding:32px}
 h1{font-size:19px;font-weight:600;letter-spacing:.04em;margin:0 0 4px}
 p{color:#5d7d8a;margin:0 0 24px}
 table{border-collapse:collapse;width:100%%;max-width:820px}
 th,td{text-align:left;padding:7px 14px;border-bottom:1px solid #16222e}
 th{color:#c39a5e;font-weight:600}
 a{color:#31e8ff;text-decoration:none}
 a:hover{text-decoration:underline}
</style>
<h1>SiPM cylinder &mdash; event display</h1>
<p>%d events. Drag to rotate, scroll to zoom, middle-drag or shift-drag to
pan.</p>
<table>
<tr><th>event</th><th>TrgID</th><th>theta</th><th>phi</th><th>|(s,c)|</th>
<th>photons</th></tr>
%s
</table>
""" % (len(events), rows)
    path = os.path.join(outdir, "index.html")
    with open(path, "w") as f:
        f.write(html)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("eventfiles", nargs="+",
                   help="event text files (applyCNN.py --printTextFiles), "
                        "each holding any number of 5-line events, or a "
                        "directory of them")
    p.add_argument("--html", default=None, metavar="PATH",
                   help="write a standalone web page instead of opening a "
                        "window.  A path ending in .html writes the first "
                        "event; anything else is treated as a directory and "
                        "gets one page per event plus an index")
    p.add_argument("--screenshot", default=None, metavar="PNG",
                   help="write a still image instead of opening a window")
    p.add_argument("--zmax", type=float, default=None,
                   help="fix the top of the colour scale (default: the "
                        "brightest SiPM in the event)")
    p.add_argument("--log", action="store_true",
                   help="logarithmic colour scale")
    p.add_argument("--cmap", default=None,
                   help="any matplotlib colormap name (default: the built-in "
                        "cyan-to-gold ramp)")
    p.add_argument("--no-halo", action="store_true",
                   help="turn off the soft glow around the bright SiPMs")
    p.add_argument("--phi-sense", choices=("to", "from"), default="to",
                   help="what phi means for the drawn track.  'to' "
                        "(default) = the muon travels towards azimuth phi, "
                        "leaving low on the phi side, which is the sense the "
                        "data supports.  'from' = it arrives from phi, i.e. "
                        "the same line rotated 180 deg in azimuth")
    p.add_argument("--track-z", type=float, default=TRACK_Z0,
                   metavar="CM", help="height at which the drawn track crosses"
                        " the detector axis, in cm (default %g).  The impact "
                        "point is not reconstructed, so this only sets where "
                        "the line is placed, not its direction" % TRACK_Z0)
    p.add_argument("--no-triggers", action="store_true",
                   help="hide the four CosmicWatch trigger channels")
    p.add_argument("--trig-distance", type=float, default=TRIG_DIST,
                   metavar="CM", help="distance from the track's axis crossing"
                        " to the centre of each trigger pair, in cm "
                        "(default %g)" % TRIG_DIST)
    p.add_argument("--savedir", default=".", metavar="DIR",
                   help="where Return writes its PNG and PDF (default: the "
                        "current directory)")
    p.add_argument("--no-panels", action="store_true",
                   help="hide the two flat side panels (the 4 x 11 photon map "
                        "on the left, the theta-phi radiograph on the right)")
    p.add_argument("--no-compass", action="store_true",
                   help="hide the brass bearing ring, its A/B/C/D board tags "
                        "and the gold phi needle")
    p.add_argument("--showPhotons", "--show-photons", dest="show_photons",
                   action="store_true",
                   help="draw scintillation photons: straight lines from the "
                        "muon's path inside the cylinder, isotropic, each "
                        "stopping where it meets the wall or an end cap "
                        "(default off)")
    p.add_argument("--nPhotonsPerCM", "--photons-per-cm", dest="photons_per_cm",
                   type=float, default=PHOTONS_PER_CM, metavar="N",
                   help="photons drawn per cm of track inside the "
                        "scintillator (default %g); at most %d lines in all"
                        % (PHOTONS_PER_CM, PHOTON_MAX))
    p.add_argument("--animate", action="store_true",
                   help="fly the muon along its track when an event is "
                        "loaded, the photons spreading out behind it as it "
                        "crosses the scintillator; 'a' replays it.  Needs a "
                        "window, so it is ignored for --screenshot and --html")
    p.add_argument("--speed", type=float, default=ANIM_SPEED, metavar="CM_S",
                   help="how fast the muon and its photons travel during "
                        "--animate, in cm/s (default %g, which crosses the "
                        "%g cm track in %.1f s)"
                        % (ANIM_SPEED, 2 * MUON_HALF_LEN,
                           2 * MUON_HALF_LEN / ANIM_SPEED))
    p.add_argument("--size", type=int, nargs=2, default=[2850, 1500],
                   metavar=("W", "H"), help="window / image size")
    args = p.parse_args(argv)

    paths = []
    for a in args.eventfiles:
        if os.path.isdir(a):
            paths += sorted(glob.glob(os.path.join(a, "*.txt")))
        else:
            paths += sorted(glob.glob(a)) if any(c in a for c in "*?[") \
                else [a]
    paths = [q for q in paths if os.path.isfile(q)]
    if not paths:
        sys.exit("no event files found")

    events = []
    for q in paths:
        events += read_events(q)
    if not events:
        sys.exit("no events found")
    cmap = args.cmap or GLOW

    print("=" * 70)
    print("events     %d in %d file%s"
          % (len(events), len(paths), "" if len(paths) == 1 else "s"))
    print("detector   cylinder %.1f cm diameter x %.0f cm, 4 boards at "
          "%s deg" % (2 * R_CYL, H_CYL,
                      "/".join("%.0f" % COL_PHI[c] for c in COLS)))
    print("SiPMs      %d per board at %.2f cm pitch; rings %d-%d read out "
          "(%d lit)" % (N_RING, SIPM_PITCH, RING_LO, RING_HI, NRING_DATA))
    print("colour     %s%s" % ("custom cyan-to-gold" if args.cmap is None
                               else args.cmap, ", log" if args.log else ""))
    if not args.no_triggers:
        rho = args.trig_distance * np.sin(np.radians(TRIG_THETA))
        dt = np.degrees(np.arctan(TRIG_SIZE / 2.0 / args.trig_distance))
        dp = np.degrees(np.arctan(TRIG_SIZE / 2.0 / rho))
        print("trigger    4 CosmicWatch channels, %g x %g x %g cm, %.2f cm "
              "from (0,0,%g)" % (TRIG_SIZE, TRIG_SIZE, TRIG_THICK,
                                 args.trig_distance, args.track_z))
        print("           axis theta %g phi %g; spans theta %.1f-%.1f, "
              "phi %+.2f to %+.2f" % (TRIG_THETA, TRIG_PHI, TRIG_THETA - dt,
                                      TRIG_THETA + dt, -dp, dp))
    if args.show_photons:
        print("photons    %g per cm of track inside the scintillator, "
              "isotropic, %g nm (%s)"
              % (args.photons_per_cm, PHOTON_NM, photon_colour()))
    if args.animate:
        if args.html or args.screenshot:
            print("animate    ignored: --screenshot and --html have no clock")
        else:
            print("animate    %g cm/s, so %g cm of track in %.1f s; "
                  "'a' replays" % (args.speed, 2 * MUON_HALF_LEN,
                                   2 * MUON_HALF_LEN / max(args.speed, 1e-9)))
    print("phi sense  muon %s phi  (--phi-sense %s)"
          % ("ARRIVES FROM" if args.phi_sense == "from" else "TRAVELS TO",
             args.phi_sense))
    print("-" * 70)
    w = max(18, max(len(e["name"]) for e in events))
    print("  %-*s %8s %9s   %s" % (w, "event", "phi_CNN", "phi_light",
                                   "board totals  " +
                                   " ".join("%5s" % c for c in COLS)))
    for e in events:
        bt = e["counts"].sum(axis=1)
        print("  %-*s %8.2f %9.2f                 %s"
              % (w, e["name"], e["phi"], light_azimuth(e["counts"]),
                 " ".join("%5d" % v for v in bt)))
    print("-" * 70)

    off = bool(args.html or args.screenshot)
    d = Display(events, cmap=cmap, zmax=args.zmax, logscale=args.log,
                off_screen=off, window_size=tuple(args.size),
                halo=not args.no_halo, phi_sense=args.phi_sense,
                compass=not args.no_compass, track_z0=args.track_z,
                triggers=not args.no_triggers, trig_dist=args.trig_distance,
                panels=not args.no_panels, savedir=args.savedir,
                show_photons=args.show_photons,
                photons_per_cm=args.photons_per_cm,
                animate=args.animate, speed=args.speed)

    if off:
        d.i = 0                      # a still or a web page wants an event
        d.draw()

    if args.screenshot:
        d.screenshot(args.screenshot)
        print("wrote %s" % args.screenshot)

    if args.html:
        if args.html.lower().endswith(".html"):
            d.html(args.html)
            print("wrote %s" % args.html)
        else:
            os.makedirs(args.html, exist_ok=True)
            pages = []
            for k in range(len(events)):
                d.i = k
                d.draw()
                pg = os.path.join(args.html, events[k]["name"] + ".html")
                d.html(pg)
                pages.append(pg)
                print("  [%d/%d] %s" % (k + 1, len(events), pg))
            idx = write_index(args.html, events, pages)
            print("wrote %d pages and %s" % (len(pages), idx))

    if not off:
        print("mouse      left = rotate, scroll = zoom, middle/shift+left = "
              "pan")
        print("keys       Right = next event, Left = previous, "
              "Home = setup view,")
        print("           Return = save PNG + PDF to %s, r = reset camera, "
              "q = quit" % os.path.abspath(args.savedir))
        print("view       opens on the bare detector; Right loads event 1 "
              "of %d" % len(events))
        print("=" * 70)
        d.show()
    else:
        print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
