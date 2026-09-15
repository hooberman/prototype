#!/usr/bin/env python3
"""
eventDisplay3D.py

Interactive 3-D event display for the SiPM cylinder prototype, built on
PyVista / VTK.

    python eventDisplay3D.py EVENT.txt                 # interactive window
    python eventDisplay3D.py EVENTDIR/*.txt            # n / p step events
    python eventDisplay3D.py EVENT.txt --html out.html # standalone web page
    python eventDisplay3D.py EVENT.txt --html site/    # a page per event + index

Mouse
-----
    left drag    rotate
    scroll       zoom in / out
    middle drag  pan          (or shift + left drag)
    right drag   dolly

Keys
----
    n / p        next / previous event (when several files are given)
    r            reset the camera
    s / w        surface / wireframe
    q            quit

Input
-----
The per-event text files applyCNN.py --printTextFiles writes:

    TrigID  theta  phi  phi_confidence
    <4 rows of 11 photon counts>

Rows are detector columns A, B, C, D; array column 0 is ring 15 and column 10
is ring 5, exactly as convertDataFile.py stores them.

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

COLS = ["A", "B", "C", "D"]
COL_PHI = {"A": 0.0, "B": 90.0, "C": 180.0, "D": 270.0}

TRACK_Z0 = 3.0               # height at which the drawn track crosses the axis

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
DEAD = "#243039"              # a SiPM with no light

# cyan through brass to pale gold: the two accent colours of the helmet, in a
# ramp that still reads monotonically from dark to bright
GLOW = LinearSegmentedColormap.from_list("glow", [
    "#0b1a26", "#0f4a63", "#1793b4", "#43dbe6",
    "#a8c96a", "#d8a63c", "#f2cf6d", "#fff3c4",
])


# ----------------------------------------------------------------------------
# input
# ----------------------------------------------------------------------------
def read_event(path, nrow=4, nring=NRING_DATA):
    """applyCNN.py --printTextFiles output -> dict."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 1 + nrow:
        sys.exit("%s: expected %d lines, found %d" % (path, 1 + nrow,
                                                      len(lines)))
    h = lines[0].split()
    if len(h) < 4:
        sys.exit("%s: header should be 'TrigID theta phi phi_confidence'"
                 % path)
    counts = []
    for r in range(nrow):
        v = lines[1 + r].split()
        if len(v) != nring:
            sys.exit("%s line %d: %d values, expected %d"
                     % (path, r + 2, len(v), nring))
        counts.append([float(x) for x in v])
    return {"path": path,
            "name": os.path.splitext(os.path.basename(path))[0],
            "trgid": int(float(h[0])),
            "theta": float(h[1]),
            "phi": float(h[2]),
            "conf": float(h[3]),
            "counts": np.asarray(counts, dtype=float)}


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


def muon_meshes(theta_deg, phi_deg, length=19.0, phi_sense="to", z0=TRACK_Z0):
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
    t, p = np.radians(theta_deg), np.radians(phi_deg)
    # unit vector pointing up-and-out along azimuth phi
    a = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    if phi_sense == "to":
        # down-going, azimuth phi: mirror the horizontal part, keep it falling
        u = np.array([a[0], a[1], -a[2]])
    else:
        u = -a                                     # down-going, away from phi
    shift = np.array([0.0, 0.0, float(z0)])
    entry_pt = -u * length + shift                 # always top -> bottom
    exit_pt = u * length + shift
    track = pv.Line(entry_pt, exit_pt).tube(radius=0.085, n_sides=20)
    head = pv.Cone(center=exit_pt - u * 0.9, direction=u, height=1.8,
                   radius=0.42, resolution=24)
    entry = pv.Sphere(radius=0.28, center=entry_pt)
    return track, head, entry


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


def hud_text(ev, counts, phi_sense="to"):
    tot = int(counts.sum())
    hot = np.unravel_index(int(np.argmax(counts)), counts.shape)
    bt = counts.sum(axis=1)
    return ("TrgID %d\n"
            "theta    %7.2f deg\n"
            "phi      %7.2f deg   (muon %s)\n"
            "conf     %7.3f\n"
            "photons  %7d\n"
            "hottest  %s ring %d  (%d)\n"
            "boards   %s\n"
            "light at %7.2f deg"
            % (ev["trgid"], ev["theta"], ev["phi"],
               "arrives from phi" if phi_sense == "from" else "travels to phi",
               ev["conf"], tot,
               COLS[hot[0]], RING_HI - hot[1], int(counts[hot]),
               "  ".join("%s %d" % (c, bt[i]) for i, c in enumerate(COLS)),
               light_azimuth(counts)))


class Display:
    """Holds the plotter and swaps the per-event actors in and out."""

    def __init__(self, events, cmap=GLOW, zmax=None, logscale=False,
                 off_screen=False, window_size=(1280, 960), halo=True,
                 phi_sense="to", compass=True, track_z0=TRACK_Z0):
        self.events = events
        self.i = 0
        self.cmap = cmap
        self.zmax = zmax
        self.log = logscale
        self.halo = halo
        self.phi_sense = phi_sense
        self.track_z0 = float(track_z0)
        self.compass = compass
        self.dynamic = []                        # actors to clear each event

        self.pl = pv.Plotter(off_screen=off_screen, window_size=window_size,
                             lighting="none")
        self.pl.set_background(BG_LOW, top=BG_HIGH)
        for mesh, style in build_detector():
            self.pl.add_mesh(mesh, **style)
        if self.compass:
            for mesh, style in compass_meshes()[0]:
                self.pl.add_mesh(mesh, reset_camera=False, **style)
            pos, txt = compass_labels()
            self.pl.add_point_labels(pos, txt, font_size=13,
                                     text_color=BRASS, shape=None,
                                     show_points=False, always_visible=True)
        light_rig(self.pl)

        # transparency has to be resolved properly or the SiPMs vanish behind
        # the scintillator from some angles
        try:
            self.pl.enable_depth_peeling(number_of_peels=8)
        except Exception:
            pass
        try:
            self.pl.enable_anti_aliasing("ssaa")
        except Exception:
            pass

        self.pl.add_axes(color=HUD, line_width=3, labels_off=False,
                         xlabel="x", ylabel="y", zlabel="z")
        if len(events) > 1:
            self.pl.add_key_event("n", self.next)
            self.pl.add_key_event("p", self.prev)
        self.draw()
        self.home_view()
        self.pl.add_key_event("r", self.home_view)

    def home_view(self):
        """Three-quarter view framed on the detector, not on the track.

        reset_camera() would fit the muon line as well, which is twice as long
        as the cylinder and would leave the detector tiny in the middle.
        """
        self.pl.camera_position = [(1.35, -1.15, 0.58), (0, 0, 0), (0, 0, 1)]
        z_top = ring_z(N_RING - 1) + 5.1          # the connector end
        z_bot = z_top - PCB_LEN
        rad = R_PCB * 2.0
        if self.compass:
            _, z_c, r_c = compass_meshes()
            rad = max(rad, r_c + 1.6)
            z_bot = min(z_bot, z_c - 1.0)
        self.pl.reset_camera(bounds=(-rad, rad, -rad, rad, z_bot, z_top))
        self.pl.camera.zoom(1.02)
        self.pl.render()

    # -- event switching ---------------------------------------------------
    def next(self):
        self.i = (self.i + 1) % len(self.events)
        self.draw()

    def prev(self):
        self.i = (self.i - 1) % len(self.events)
        self.draw()

    def draw(self):
        for a in self.dynamic:
            try:
                self.pl.remove_actor(a, render=False)
            except Exception:
                pass
        self.dynamic = []

        ev = self.events[self.i]
        counts = ev["counts"]
        vmax = self.zmax if self.zmax else max(float(counts.max()), 1.0)
        clim = (max(1.0, 0.0) if self.log else 0.0, vmax)

        tiles = build_sipms(counts)
        self.dynamic.append(self.pl.add_mesh(
            tiles, scalars="photons", cmap=self.cmap, clim=clim,
            reset_camera=False,
            log_scale=self.log, smooth_shading=False, specular=0.55,
            specular_power=25, ambient=0.30, diffuse=0.85,
            show_scalar_bar=True,
            scalar_bar_args=dict(title="photons", color=HUD, n_labels=6,
                                 vertical=True, width=0.035, height=0.42,
                                 position_x=0.905, position_y=0.30,
                                 title_font_size=15, label_font_size=12,
                                 fmt="%.0f")))

        if self.halo:
            h = build_halo(counts)
            if h is not None:
                self.dynamic.append(self.pl.add_mesh(
                    h, scalars="photons", cmap=self.cmap, clim=clim,
                    reset_camera=False,
                    opacity=0.22, show_scalar_bar=False, ambient=1.0,
                    diffuse=0.0, specular=0.0))

        if self.compass:
            shaft, tip, _, _ = phi_pointer(ev["phi"])
            for m in (shaft, tip):
                self.dynamic.append(self.pl.add_mesh(
                    m, color=BRASS_LIT, smooth_shading=True,
                    reset_camera=False, ambient=0.55, diffuse=0.7,
                    specular=1.0, specular_power=70, show_scalar_bar=False))

        track, head, entry = muon_meshes(ev["theta"], ev["phi"],
                                         phi_sense=self.phi_sense,
                                         z0=self.track_z0)
        for m, op in ((track, 1.0), (head, 1.0), (entry, 0.9)):
            self.dynamic.append(self.pl.add_mesh(
                m, color=MUON, smooth_shading=True, opacity=op,
                reset_camera=False,
                ambient=0.75, diffuse=0.5, specular=1.0, specular_power=60,
                show_scalar_bar=False))
        # a faint fat tube around the track, same bloom trick as the SiPMs
        self.dynamic.append(self.pl.add_mesh(
            pv.Line(*[p for p in (track.points[0], track.points[-1])])
              .tube(radius=0.30, n_sides=16),
            color=MUON, opacity=0.10, ambient=1.0, diffuse=0.0,
            reset_camera=False,
            show_scalar_bar=False))

        self.dynamic.append(self.pl.add_text(
            hud_text(ev, counts, self.phi_sense),
            position="upper_left", font_size=11,
            color=HUD, font="courier", shadow=True))
        foot = ev["name"]
        if len(self.events) > 1:
            foot = "[%d/%d]  %s   (n = next, p = previous)" \
                % (self.i + 1, len(self.events), foot)
        self.dynamic.append(self.pl.add_text(
            foot, position="lower_left", font_size=8, color="#5d7d8a",
            font="courier"))
        self.pl.render()

    # -- outputs -----------------------------------------------------------
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
                   help="event text files (applyCNN.py --printTextFiles), or "
                        "a directory of them")
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
                   help="what phi means for the drawn track.  'from' (default)"
                        " = the muon arrives from azimuth phi, entering high "
                        "on the phi side, which is the sense that matches the "
                        "training MC.  'to' = it travels towards phi, i.e. the"
                        " same line rotated 180 deg in azimuth")
    p.add_argument("--track-z", type=float, default=TRACK_Z0,
                   metavar="CM", help="height at which the drawn track crosses"
                        " the detector axis, in cm (default %g).  The impact "
                        "point is not reconstructed, so this only sets where "
                        "the line is placed, not its direction" % TRACK_Z0)
    p.add_argument("--no-compass", action="store_true",
                   help="hide the brass bearing ring, its A/B/C/D board tags "
                        "and the gold phi needle")
    p.add_argument("--size", type=int, nargs=2, default=[1280, 960],
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

    events = [read_event(q) for q in paths]
    cmap = args.cmap or GLOW

    print("=" * 70)
    print("events     %d" % len(events))
    print("detector   cylinder %.1f cm diameter x %.0f cm, 4 boards at "
          "%s deg" % (2 * R_CYL, H_CYL,
                      "/".join("%.0f" % COL_PHI[c] for c in COLS)))
    print("SiPMs      %d per board at %.2f cm pitch; rings %d-%d read out "
          "(%d lit)" % (N_RING, SIPM_PITCH, RING_LO, RING_HI, NRING_DATA))
    print("colour     %s%s" % ("custom cyan-to-gold" if args.cmap is None
                               else args.cmap, ", log" if args.log else ""))
    print("phi sense  muon %s phi  (--phi-sense %s)"
          % ("ARRIVES FROM" if args.phi_sense == "from" else "TRAVELS TO",
             args.phi_sense))
    print("-" * 70)
    print("  %-18s %8s %8s   %s" % ("event", "phi_CNN", "phi_light",
                                    "board totals  " +
                                    " ".join("%5s" % c for c in COLS)))
    for e in events:
        bt = e["counts"].sum(axis=1)
        print("  %-18s %8.2f %8.2f                 %s"
              % (e["name"], e["phi"], light_azimuth(e["counts"]),
                 " ".join("%5d" % v for v in bt)))
    print("-" * 70)

    off = bool(args.html or args.screenshot)
    d = Display(events, cmap=cmap, zmax=args.zmax, logscale=args.log,
                off_screen=off, window_size=tuple(args.size),
                halo=not args.no_halo, phi_sense=args.phi_sense,
                compass=not args.no_compass, track_z0=args.track_z)

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
        if len(events) > 1:
            print("keys       n = next event, p = previous, r = reset camera, "
                  "q = quit")
        else:
            print("keys       r = reset camera, q = quit")
        print("=" * 70)
        d.show()
    else:
        print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
