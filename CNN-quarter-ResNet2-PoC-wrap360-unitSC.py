#!/usr/bin/env python3
"""
train_cnn.py FILENAME

Train a CNN to regress muon (theta, phi) from 4xN SiPM "images"
stored in a text file of the form:

theta phi z
row0 (NUM_TILES floats)
row1 (NUM_TILES floats)
row2 (NUM_TILES floats)
row3 (NUM_TILES floats)

(blank line between events is allowed)

The network:
- Input: 1 x 4 x NUM_TILES
- Parallel Conv2d branches with kernel sizes (4,1), (4,2), (4,3), (4,4), (4,5)
- No pooling layers
- Fully-connected layers -> 3 outputs (theta_norm, sin_phi, cos_phi)

Why 3 outputs: the seam
-----------------------
phi is periodic, and a single unconstrained output cannot cover the circle --
there is no continuous map from the input onto it, so the network has to put
a seam somewhere and every event near the seam is predicted badly.  Wrapping
the loss ((dphi + 180) % 360 - 180) removes the discontinuity in the PENALTY
but not the one in the REPRESENTATION.  So the head keeps predicting a
2-vector (s, c) with phi = atan2(s, c): the circle is a smooth 1-manifold in
R^2 and there is no seam.

Why (s, c) is NORMALISED before the loss here
---------------------------------------------
This variant differs from the plain sin/cos version in one place, the loss.
Written out with n = |(s, c)|, the unnormalised squared error is

    (s - sin phi)^2 + (c - cos phi)^2  =  n^2 + 1 - 2 n cos(dphi)

so the gradient pulling phi toward the truth is proportional to n, and n's own
optimum is the resultant R of p(phi | event).  Azimuthally ambiguous events
therefore shrink n, stop receiving angular gradient, and end up with a phi set
by the SHAPE of the (s, c) cloud rather than by the data.  A cloud that is a
rounded square rather than a ring puts those events on the DIAGONALS, i.e. a
4-fold (m = 4) modulation peaking near 45/135/225/315 deg.

That is not hypothetical: the plain sin/cos model produced exactly that --
four peaks near 38, 135, 215, 305 deg in reconstructed phi on real data -- and
a single-phi-output model trained on the same events did not.  Normalising
(s, c) before the squared error removes the radial degree of freedom without
giving up the seamless representation, so every event gets full-strength
angular gradient however uncertain it is.

The loss is then

    n        = |(s, c)|
    loss_phi = (s/n - sin phi)^2 + (c/n - cos phi)^2  =  2(1 - cos dphi)
    dtheta   = (theta_norm_pred - theta_norm_true) * theta_std * pi/180
    loss     = dtheta^2 + loss_phi + W_NORM * (n - 1)^2

No w_phi: 2(1 - cos dphi) ~ dphi_rad^2 and dtheta is converted back to
radians, so one degree of theta error and one degree of phi error cost exactly
the same.  (Without the theta_std * pi/180 factor the standardised theta
residual is over-weighted by (180/pi/theta_std)^2 -- a factor 10 at
theta_std ~ 18 deg.)  loss_phi still saturates at 4 rather than growing like
dphi^2, so it stays outlier-tolerant.

The (n - 1)^2 anchor is not optional.  Once normalised the phi term is
scale-invariant in (s, c): nothing controls n, the gradient through the
normalisation goes as 1/n, and weight_decay drags the head toward zero.  Left
alone that ends in NaNs a few epochs in.

|(s, c)| IS NO LONGER A CONFIDENCE.  The anchor pins it near 1 by
construction.  It is still computed and exported, but as a health check:
phi_confidence.pdf should show a narrow spike at 1 and a FLAT resolution-vs-
|(s,c)| curve.  Anything else means the anchor is losing to weight_decay.

theta is still standardised for the network, since it is not periodic.  phi is
NOT standardised -- a linear mean of a circular variable is meaningless (the
mean of 5 deg and 355 deg is 180 deg, exactly wrong).
"""

from scipy.optimize import curve_fit

import copy
import datetime
import json
import platform
import shutil
import warnings
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")       # headless: compute nodes have no display
import matplotlib.pyplot as plt
from typing import Tuple
import torch.nn.functional as F

# -----------------------
# Event selection (config)
# -----------------------
# Every knob below can be overridden from the environment, so the same file
# serves a short test job and the full run without being edited:
#   CNN_EPOCHS=20 CNN_BATCH=256 CNN_MAX_EVENTS=50000 python this_script.py FILE
def _envint(name, default):
    v = os.environ.get(name)
    return default if v is None or v == "" else int(v)


NUM_EPOCHS       = _envint("CNN_EPOCHS", 20)         # number of epochs
VOLT_THRESHOLD_V = 0.01        # Require "hit" SiPMs to be >= this voltage (in VOLTS)
MIN_SIPMS_PER_TILE = 2         # Per tile: need at least this many SiPMs above threshold
MIN_TILES_PER_EVENT = 2        # Per event: need at least this many tiles passing the rule
NUM_TILES = 11
MAX_EVENTS = _envint("CNN_MAX_EVENTS", -1)   # -1 means use all events
BATCH_SIZE  = _envint("CNN_BATCH", 32)
NUM_WORKERS = _envint("CNN_WORKERS", 0)      # >0 needs the DataLoader fork

THETA_FIT_RANGE = (-5.0, 5.0)
PHI_FIT_RANGE   = (-10.0, 10.0)

# There is no w_phi any more: both loss terms are squared errors in radians,
# so they are already 1:1 per degree.  See the module docstring.
#
# W_NORM is the weight of the (|(s,c)| - 1)^2 anchor that keeps the output
# vector off zero once the phi term has been made scale-invariant.  It only
# has to beat weight_decay, not shape the fit -- 0.1 is plenty, and the run
# is insensitive to it over at least 0.01 to 1.  Watch phi_confidence.pdf:
# if |(s,c)| is not a narrow spike at 1, raise it.
W_NORM = 0.1

OUTPUTDIR = (
    "ResNet2_unitSC_"
    f"nTiles{MIN_TILES_PER_EVENT}"
    f"_nSiPM{MIN_SIPMS_PER_TILE}"
    f"_mV{int(round(1000*VOLT_THRESHOLD_V))}"
    f"_{NUM_EPOCHS}epochs"
)

if MAX_EVENTS > 0:
    OUTPUTDIR += f"_maxEv{MAX_EVENTS}"
    
def event_passes_voltage_cut(image_4xN: np.ndarray) -> bool:
    """
    image_4xN: shape (4, NUM_TILES) in VOLTS (raw, before normalization)

    Pass if at least MIN_TILES_PER_EVENT tiles have at least
    MIN_SIPMS_PER_TILE SiPMs with V >= VOLT_THRESHOLD_V.
    """

    if image_4xN.shape != (4, NUM_TILES):
        raise ValueError(f"Expected (4,{NUM_TILES}), got {image_4xN.shape}")

    # Boolean mask of SiPMs above threshold: (4, N)
    above = (image_4xN >= VOLT_THRESHOLD_V)

    # Count "hit" SiPMs per tile: (NUM_TILES,)
    hit_sipms_per_tile = np.sum(above, axis=0)

    # Tiles that pass: need at least MIN_SIPMS_PER_TILE hit SiPMs
    tiles_passing = np.sum(hit_sipms_per_tile >= MIN_SIPMS_PER_TILE)


    # ---- TEMP DEBUG (remove after first few events) ----
    #print("image_4xN shape:", image_4xN.shape)
    #print("above shape:", above.shape)
    #print("hit_sipms_per_tile shape:", hit_sipms_per_tile.shape)
    #print("hit_sipms_per_tile (first 10 tiles):", hit_sipms_per_tile[:10])
    # ----------------------------------------------------
    
    return tiles_passing >= MIN_TILES_PER_EVENT



class AngularLoss(nn.Module):
    """Squared angular error on theta and phi, both in RADIANS, 1:1 per degree,
    with (s, c) normalised before the phi term.

    y_pred : (B, 3)  = theta_norm, s, c          <- network output
    y_true : (B, 2)  = theta_norm, phi_degrees   <- phi is NOT standardised

        n        = |(s, c)|
        loss_phi = (s/n - sin phi)^2 + (c/n - cos phi)^2  =  2(1 - cos dphi)
        dtheta   = (theta_norm_pred - theta_norm_true) * theta_std * pi/180
        loss     = dtheta^2 + loss_phi + w_norm * (n - 1)^2

    The module docstring has the full argument.  In one line: normalising
    first makes the phi gradient independent of n, which is what stops
    azimuthally ambiguous events from having their phi decided by the shape of
    the (s, c) cloud (the m = 4 artifact at 45/135/225/315 deg).  The anchor
    keeps n off zero, without which the 1/n gradient through the
    normalisation plus weight_decay gives NaNs.

    |(s, c)| is NOT a confidence in this variant -- the anchor pins it near 1.
    """

    def __init__(self, theta_mean, theta_std, w_norm=W_NORM):
        super().__init__()
        self.theta_mean = float(theta_mean)
        self.theta_std  = float(theta_std)
        self.w_norm     = float(w_norm)
        # standardised theta residual -> radians, so theta and phi are 1:1
        self.theta_scale = float(theta_std) * np.pi / 180.0

    def forward(self, y_pred, y_true):
        dtheta = (y_pred[:, 0] - y_true[:, 0]) * self.theta_scale

        phi_rad = y_true[:, 1] * (np.pi / 180.0)
        s, c = y_pred[:, 1], y_pred[:, 2]

        n = torch.sqrt(s * s + c * c + 1e-6)
        s, c = s / n, c / n
        loss_phi = (s - torch.sin(phi_rad)) ** 2 + (c - torch.cos(phi_rad)) ** 2

        return torch.mean(dtheta ** 2 + loss_phi
                          + self.w_norm * (n - 1.0) ** 2)


def decode_phi(s, c, phi_lo=0.0):
    """(s, c) -> phi in degrees, wrapped into [phi_lo, phi_lo + 360)."""
    phi = np.degrees(np.arctan2(s, c))
    return wrap_to(phi, phi_lo)


def wrap_to(phi_deg, lo=0.0):
    """Wrap degrees into [lo, lo + 360)."""
    return (np.asarray(phi_deg, dtype=float) - lo) % 360.0 + lo


def delta_phi_deg(phi_pred, phi_true):
    """
    Smallest signed difference between two angles in degrees,
    in the range [-180, 180).
    """
    diff = (phi_pred - phi_true + 180.0) % 360.0 - 180.0
    return diff


# -----------------------
# Reproducibility helpers
# -----------------------
def set_seed(seed: int = 1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------
# Data loading
# -----------------------
def load_events(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse CNN-style file:
      line 1: theta phi z
      next 4 lines: NUM_TILES floats each (SiPM voltages)
      blank line optional between events

    Returns:
        images: shape (N, 4, NUM_TILES)
        labels: shape (N, 2) -> [theta, phi]
    """
    images = []
    labels = []

    with open(filename, "r") as f:
        # strip trailing newline, skip empty lines
        lines = [ln.strip() for ln in f.readlines() if ln.strip() != ""]

    i = 0
    n = len(lines)
    while i < n:
        header = lines[i].split()
        if len(header) < 2:
            raise ValueError(f"Malformed header at line {i+1}: {lines[i]}")
        theta = float(header[0])
        phi = float(header[1])
        # third value (z) is ignored for now

        # Full 2*pi in phi.  This is what the sin/cos head is for -- with a
        # single phi output the seam would live somewhere in here.
        keep_event = True
        #keep_event = (phi >= 50.0) and (phi <= 310.0)
        #keep_event = (phi >= 135.0) and (phi <= 225.0)

        
        # Next 4 lines: 4xN image

        # --- Skip first 4xN block (junk / placeholder) ---
        for _ in range(4):
            i += 1
            if i >= n:
                raise ValueError("Unexpected end of file while skipping first image block.")

        
        rows = []
        for r in range(4):
            i += 1
            if i >= n:
                raise ValueError("Unexpected end of file while reading image rows.")
            row_vals = [float(x) for x in lines[i].split()]
            if len(row_vals) != NUM_TILES:
                raise ValueError(
                    f"Expected {NUM_TILES} values in image row at line {i+1}, got {len(row_vals)}"
                )
            rows.append(row_vals)

        image = np.array(rows, dtype=np.float32)  # (4,NUM_TILES)


        if keep_event:
            images.append(image)
            labels.append([theta, phi])

        i += 1  # move to next header (or blank)

    images = np.stack(images, axis=0)  # (N,4,NUM_TILES)
    labels = np.array(labels, dtype=np.float32)  # (N,2)
    return images, labels


class MuonDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        """
        images: (N,4,NUM_TILES)
        labels: (N,2)
        """
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # Add channel dimension: (1,4,NUM_TILES)
        x = torch.from_numpy(self.images[idx][None, :, :])  # (1,4,NUM_TILES)
        y = torch.from_numpy(self.labels[idx])              # (2,)
        return x, y


# -----------------------
# Model definition
# -----------------------


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = self.relu(out)

        return out


class MuonDirectionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_channels = 64

        # CIFAR-style stem
        self.stem = nn.Sequential(
            nn.Conv2d(
                1,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ResNet-18 layers
        self.layer1 = self._make_layer(64, 2, stride=(1, 1))
        self.layer2 = self._make_layer(128, 2, stride=(1, 2))
        self.layer3 = self._make_layer(256, 2, stride=(1, 2))
        self.layer4 = self._make_layer(512, 2, stride=(1, 2))

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Regression head: theta_norm, sin_phi, cos_phi
        self.out = nn.Linear(512, 3)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = [
            BasicBlock(self.in_channels, out_channels, stride)
        ]
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Preserve your circular padding
        x = F.pad(x, (0, 0, 0, 3), mode="circular")

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        return self.out(x)



# -----------------------
# Training / evaluation
# -----------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    n_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    return running_loss / n_samples


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    n_samples = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)

            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

            all_true.append(y.cpu().numpy())
            all_pred.append(y_pred.cpu().numpy())

    avg_loss = running_loss / n_samples
    if all_true:
        all_true = np.concatenate(all_true, axis=0)
        all_pred = np.concatenate(all_pred, axis=0)
    else:
        all_true, all_pred = None, None

    return avg_loss, all_true, all_pred


# -----------------------
# Main script
# -----------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python train_cnn.py FILE_CNN.txt")
        sys.exit(1)

    filename = sys.argv[1]
    set_seed(42)

    print(f"Loading events from {filename} ...")
    images, labels = load_events(filename)

    # Apply MAX_EVENTS limit (before cuts, so it's truly "first N events")
    if MAX_EVENTS > 0:
        images = images[:MAX_EVENTS]
        labels = labels[:MAX_EVENTS]
        print(f"Using only first {MAX_EVENTS} events.")

    print(f"Loaded {images.shape[0]} events. Image shape: {images.shape[1:]}")

    #outdir = os.path.join("plots", OUTPUTDIR)

    # FILE = input filename without .txt
    file_stem = os.path.splitext(os.path.basename(filename))[0]

    base_outdir = os.path.join("plots", file_stem, OUTPUTDIR)
    #base_outdir = os.path.join("plots", OUTPUTDIR)
    outdir = base_outdir

    suffix = 2
    while os.path.exists(outdir):
        outdir = f"{base_outdir}_Take{suffix}"
        suffix += 1

    print(f"Writing plots to: {outdir} ...")

    #os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir)

    # Copy this script (CNN.py / train_cnn.py) into outdir for provenance
    script_path = os.path.abspath(__file__)
    shutil.copy2(script_path, os.path.join(outdir, os.path.basename(script_path)))

    
    # Symlink the input file (avoid copying huge data)
    input_path = os.path.abspath(filename)
    link_name  = os.path.join(outdir, os.path.basename(input_path))

    try:
        os.symlink(input_path, link_name)
    except FileExistsError:
        pass  # shouldn't happen with your unique outdir, but harmless

    # ---------------------------------------------------------
    # Event selection on RAW voltages (before normalization)
    # ---------------------------------------------------------
    keep_mask = np.array([event_passes_voltage_cut(img) for img in images], dtype=bool)
    
    n_before = images.shape[0]
    images = images[keep_mask]
    labels = labels[keep_mask]
    n_after = images.shape[0]

    print(
        "After voltage cut: kept "
        f"{n_after}/{n_before} events "
        f"({(n_after / max(n_before, 1)) * 100.0:.1f}%). "
        f"Cut = {MIN_TILES_PER_EVENT} tiles with ≥{MIN_SIPMS_PER_TILE} SiPMs "
        f"at ≥{VOLT_THRESHOLD_V:.3f} V."
    )

    if n_after < 10:
        raise RuntimeError(
            "Too few events after cuts. Loosen VOLT_THRESHOLD_V / "
            "MIN_SIPMS_PER_TILE / MIN_TILES_PER_EVENT."
        )

    # Normalize inputs (per-dataset standardization)
    # (This is optional but usually helps training.)
    img_mean = images.mean()
    img_std = images.std() if images.std() > 0 else 1.0
    images_norm = (images - img_mean) / img_std

    # ---------------------------------------------------------
    # Targets.
    #   theta  -> standardised, as before.
    #   phi    -> left in DEGREES.  Standardising a circular variable is
    #             meaningless: labels.mean() on phi spread over the full
    #             circle returns roughly the arithmetic centre of the range,
    #             which is not the circular mean and carries no information.
    #             The loss turns phi into (sin, cos) itself.
    # ---------------------------------------------------------
    theta_mean = float(labels[:, 0].mean())
    theta_std  = float(labels[:, 0].std())
    if theta_std == 0.0:
        theta_std = 1.0

    phi_all = labels[:, 1]
    phi_lo = -180.0 if phi_all.min() < -1e-6 else 0.0   # match the input range

    targets = np.stack([(labels[:, 0] - theta_mean) / theta_std,
                        phi_all], axis=1).astype(np.float32)

    # circular mean / resultant of the truth, purely as a sanity print
    ph = np.radians(phi_all.astype(float))
    R_true = float(np.hypot(np.sin(ph).mean(), np.cos(ph).mean()))
    print(f"theta: mean={theta_mean:.3f} std={theta_std:.3f} deg")
    print(f"phi  : range [{phi_all.min():.1f}, {phi_all.max():.1f}] deg, "
          f"wrapped into [{phi_lo:.0f}, {phi_lo + 360:.0f}); "
          f"resultant R={R_true:.4f} "
          f"({'flat in phi' if R_true < 0.05 else 'NOT flat in phi'})")

    print("loss : dtheta^2 + [(s/n - sin)^2 + (c/n - cos)^2] + "
          "%g (n-1)^2,   n = |(s,c)|" % W_NORM)
    print("       both angular terms in radians -> 1 deg of phi costs exactly "
          "1 deg of theta")
    print("       (s,c) normalised before the phi term: no radial degree of "
          "freedom, so")
    print("       |(s,c)| is pinned near 1 and is NOT a confidence")

    labels_norm = targets

    # Train/test split (80/20)
    N = images.shape[0]
    indices = np.random.permutation(N)
    n_train = int(0.8 * N)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_images = images_norm[train_idx]
    train_labels = labels_norm[train_idx]
    test_images = images_norm[test_idx]
    test_labels = labels_norm[test_idx]

    train_dataset = MuonDataset(train_images, train_labels)
    test_dataset = MuonDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)

    # CUDA first (cluster), then Apple MPS (laptop), then CPU.  getattr guards
    # torch builds that have no mps backend at all.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif getattr(getattr(torch.backends, "mps", None), "is_available", bool)():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    
    print(f"Using device: {device}")

    model = MuonDirectionCNN().to(device)
    #criterion = nn.MSELoss()
    criterion = AngularLoss(theta_mean, theta_std, W_NORM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=NUM_EPOCHS)

    num_epochs = NUM_EPOCHS
    train_losses = []
    test_losses = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Epoch {epoch:3d}/{num_epochs}: "
              f"train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")

    # Final evaluation: get predictions on test set (denormalized)
    _, y_true_raw, y_pred_raw = evaluate(model, test_loader, criterion, device)

    # y_true_raw : (N, 2) = theta_norm, phi_deg
    # y_pred_raw : (N, 3) = theta_norm, s, c
    theta_true = y_true_raw[:, 0] * theta_std + theta_mean
    phi_true   = wrap_to(y_true_raw[:, 1], phi_lo)

    theta_pred = y_pred_raw[:, 0] * theta_std + theta_mean
    s_pred, c_pred = y_pred_raw[:, 1], y_pred_raw[:, 2]
    phi_pred = decode_phi(s_pred, c_pred, phi_lo)

    # |(s, c)| is NOT a confidence in this variant: the (n-1)^2 anchor pins it
    # near 1.  It is kept as a health check -- a narrow spike at 1 means the
    # anchor is holding; a broad or collapsing distribution means weight_decay
    # is winning and the phi gradient is being scaled down again, which is the
    # failure this whole variant exists to avoid.
    phi_conf = np.hypot(s_pred, c_pred)
    print(f"output norm |(s,c)|: median={np.median(phi_conf):.3f} "
          f"[{np.percentile(phi_conf, 5):.3f}, "
          f"{np.percentile(phi_conf, 95):.3f}]   "
          f"(anchored to 1; NOT a confidence)")
    if not 0.9 < float(np.median(phi_conf)) < 1.1:
        print("  WARNING  the anchor is not holding -- raise W_NORM "
              "(currently %g) or lower weight_decay" % W_NORM)

    # -----------------------
    # Residuals (needed by both the export and the plots)
    # -----------------------
    theta_res = theta_pred - theta_true
    phi_res   = delta_phi_deg(phi_pred, phi_true)

    # -----------------------
    # Export the trained network BEFORE plotting, so a failure in any of the
    # cosmetics below cannot cost you the model.
    # -----------------------
    export_trained_model(
        outdir, model, script_path, filename, device,
        img_mean, img_std, theta_mean, theta_std, phi_lo, W_NORM,
        len(train_idx), len(test_idx), train_losses, test_losses,
        theta_true, phi_true, theta_pred, phi_pred, phi_conf,
        theta_res, phi_res,
    )

    # -----------------------
    # Diagnostic plots
    # -----------------------

    # 1. Loss vs epoch (train + test)
    plt.figure(figsize=(8, 6))
    epochs = np.arange(1, num_epochs + 1)
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, test_losses, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (normalized targets)")
    plt.title("Training and test loss vs epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss.pdf"), bbox_inches="tight")

    # 2. Resolution plots (theta, phi) for test sample

    # (a) 1D residuals: pred - true (theta) and wrapped for phi
    #     (theta_res / phi_res are computed above, before the export)

    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    add_residual_hist_with_fit(
        ax1,
        theta_res,
        bins=100,
        xlabel=r"$\theta_{\rm pred} - \theta_{\rm true}$ [degrees]",
        title=r"$\theta$ residuals",
        xlim=(-30, 30),
        fit_range=THETA_FIT_RANGE
    )

    ax2 = plt.subplot(1, 2, 2)
    add_residual_hist_with_fit(
        ax2,
        phi_res,
        bins=100,
        xlabel=r"$\phi_{\rm pred} - \phi_{\rm true}$ [degrees]",
        title=r"$\phi$ residuals",
        xlim=(-30, 30),
        fit_range=PHI_FIT_RANGE
    )

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "resolution1D.pdf"), bbox_inches="tight")

###     # ---------------------------------------------------
###     # 2a. Theta/phi residuals in bins of true theta
###     #     Bins: 0–15, 15–30, 30–45, 45–60, 60–75, >75 deg
###     # ---------------------------------------------------
###     theta_edges = [0, 15, 30, 45, 60, 75, 1e9]  # last bin is >75
###     bin_labels = [
###         r"$0^\circ < \theta_{\rm true} < 15^\circ$",
###         r"$15^\circ < \theta_{\rm true} < 30^\circ$",
###         r"$30^\circ < \theta_{\rm true} < 45^\circ$",
###         r"$45^\circ < \theta_{\rm true} < 60^\circ$",
###         r"$60^\circ < \theta_{\rm true} < 75^\circ$",
###         r"$\theta_{\rm true} > 75^\circ$",
###     ]
### 
###     fig, axes = plt.subplots(
###         2, 6, figsize=(14, 5), sharex=False, sharey=False
###     )

    # ---------------------------------------------------
    # 2a. Theta/phi residuals in bins of true theta
    #     Bins: 0–5, 5-15, 15-25, 25–35, 35–45, 45–55, 55-65, 65–75, >75 deg
    # ---------------------------------------------------
    theta_edges = [0, 5, 15, 25, 35, 45, 55, 65, 75, 1e9]  # last bin is >75
#    bin_labels = [
#        r"\theta_{\rm true} < 5^\circ$",
#        r"$5^\circ < \theta_{\rm true} < 15^\circ$",
#        r"$15^\circ < \theta_{\rm true} < 25^\circ$",
#        r"$25^\circ < \theta_{\rm true} < 35^\circ$",
#        r"$35^\circ < \theta_{\rm true} < 45^\circ$",
#        r"$45^\circ < \theta_{\rm true} < 55^\circ$",
#        r"$55^\circ < \theta_{\rm true} < 65^\circ$",
#        r"$65^\circ < \theta_{\rm true} < 75^\circ$",
#        r"$\theta_{\rm true} > 75^\circ$",
#    ]

    fig, axes = plt.subplots(
        2, 9, figsize=(20, 5), sharex=False, sharey=False
    )

    for i in range(9):
        lo = theta_edges[i]
        hi = theta_edges[i + 1]

        if i < 8:
            mask = (theta_true >= lo) & (theta_true < hi)
            bin_label = fr"{lo:.0f}° < $\theta_{{\rm true}}$ < {hi:.0f}°"
        else:
            mask = (theta_true >= lo)
            bin_label = r"$\theta_{\rm true} > 75^\circ$"

        theta_res_bin = theta_res[mask]
        phi_res_bin   = phi_res[mask]

        # --- Top row: theta residuals (no x-label, short title) ---
        ax_theta = axes[0, i]
        add_residual_hist_with_fit_binned(
            ax_theta,
            theta_res_bin,
            bins=50,
            xlabel=r"$\theta_{\rm pred} - \theta_{\rm true}$ [degrees]",
            title=bin_label,
            xlim=(-30, 30),
            max_sigma=20,
        )
        if i > 0:
            ax_theta.set_ylabel("")  # only leftmost has y-label

        # --- Bottom row: phi residuals (shared x-label text) ---
        ax_phi = axes[1, i]
        add_residual_hist_with_fit_binned(
            ax_phi,
            phi_res_bin,
            bins=50,
            xlabel=r"$\phi_{\rm pred} - \phi_{\rm true}$ [degrees]",
            title="",                # no extra title; bin label is above
            xlim=(-30, 30),
            max_sigma=60,
        )
        if i > 0:
            ax_phi.set_ylabel("")    # only leftmost has y-label

    # Set explicit y-labels on the left column
    axes[0, 0].set_ylabel("Entries")
    axes[1, 0].set_ylabel("Entries")

    fig.suptitle(
        "Residuals vs true $\\theta$ (top: $\\theta$, bottom: $\\phi$)",
        y=0.96,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(outdir, "resolutionBinned.pdf"), bbox_inches="tight")

    # ---------------------------------------------------
    # 2b. "Canvas" plot: resolution (Gaussian sigma) vs true theta
    #     Left: theta resolution vs true theta
    #     Right: phi resolution vs true theta
    #
    #     - y = best-fit Gaussian sigma in each true-theta bin
    #     - yerr = fit uncertainty on sigma
    #     - x = bin center, xerr = half bin width
    #     - Legend: "multiple SiPMs"
    #     - Theta panel only: overlay "single SiPM" reference histogram (4 bins)
    # ---------------------------------------------------
    #os.makedirs("plots", exist_ok=True)

    # Use your existing theta_edges. Skip the last open-ended bin (>75) for this plot.
    x_centers_theta = []
    x_centers_phi = []
    x_errs_theta    = []
    x_errs_phi    = []
    theta_sig = []
    theta_sig_err = []
    phi_sig   = []
    phi_sig_err = []

    for i in range(len(theta_edges) - 2):  # up to the second-to-last edge (i.e., last finite bin)
        lo = float(theta_edges[i])
        hi = float(theta_edges[i + 1])

        mask = (theta_true >= lo) & (theta_true < hi)
        th_bin = theta_res[mask]
        ph_bin = phi_res[mask]

        xc = 0.5 * (lo + hi)
        xe = 0.5 * (hi - lo)

        s_th, s_th_err, n_th = fit_gaussian_sigma_and_uncertainty(th_bin, bins=80, max_sigma=20.0)
        s_ph, s_ph_err, n_ph = fit_gaussian_sigma_and_uncertainty(ph_bin, bins=50, max_sigma=60.0)

        # --- theta ---
        if np.isfinite(s_th):
            x_centers_theta.append(xc)
            x_errs_theta.append(xe)
            theta_sig.append(s_th)
            theta_sig_err.append(s_th_err)
            
        # --- phi ---
        if np.isfinite(s_ph):
            x_centers_phi.append(xc)
            x_errs_phi.append(xe)
            phi_sig.append(s_ph)
            phi_sig_err.append(s_ph_err)

    x_centers_theta = np.array(x_centers_theta, dtype=float)
    x_errs_theta    = np.array(x_errs_theta, dtype=float)
    x_centers_phi   = np.array(x_centers_phi, dtype=float)
    x_errs_phi      = np.array(x_errs_phi, dtype=float)

    theta_sig     = np.array(theta_sig, dtype=float)
    theta_sig_err = np.array(theta_sig_err, dtype=float)

    phi_sig     = np.array(phi_sig, dtype=float)
    phi_sig_err = np.array(phi_sig_err, dtype=float)

    # Build the 2-panel figure
    fig2, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

    axL.errorbar(
        x_centers_theta, theta_sig,
        yerr=theta_sig_err,
        xerr=x_errs_theta,
        fmt="o",
        capsize=3,
        label="cylinder + SiPMs",
    )

    axL.set_xlabel(r"true $\theta$ [degrees]",fontsize=14)
    axL.set_ylabel(r"$\sigma_\theta$ [degrees]",fontsize=14)
    axL.set_title(r"$\theta$ resolution vs true $\theta$",fontsize=16)
    axL.grid(True, alpha=0.3)
    axL.set_ylim(0,10)
    axL.set_xlim(0,75)

    # Overlay "single SiPM" reference as a 4-bin step histogram on the theta panel
    single_edges = np.array([5.0, 15.0, 25.0, 35.0, 45.0], dtype=float)
    single_vals  = np.array([2.0, 2.6, 4.2, 4.5], dtype=float)

    # Step plot that looks like a histogram:
    axL.step(
        single_edges,
        np.r_[single_vals, single_vals[-1]],
        where="post",
        linewidth=2,
        label="tile + SiPM",
    )

    axL.legend(loc="best")

    axR.errorbar(
        x_centers_phi, phi_sig,
        yerr=phi_sig_err,
        xerr=x_errs_phi,
        fmt="o",
        capsize=3,
        label="cylinder + SiPMs",
    )

    axR.set_xlabel(r"true $\theta$ [degrees]",fontsize=14)
    axR.set_ylabel(r"$\sigma_\phi$ [degrees]",fontsize=14)
    axR.set_title(r"$\phi$ resolution vs true $\theta$",fontsize=16)
    axR.grid(True, alpha=0.3)
    axR.legend(loc="best")
    axR.set_ylim(0,10)
    axR.set_xlim(0,75)
    
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, "resolution_vs_true_theta.pdf"), bbox_inches="tight")

    
    # (b) 2D scatter plots: true vs pred
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    #plt.scatter(theta_true, theta_pred, s=1, alpha=0.7)
    hb = plt.hexbin(theta_true, theta_pred, gridsize=100, bins='log')
    plt.colorbar(hb, label='log10(N)')
    min_t = min(theta_true.min(), theta_pred.min())
    max_t = max(theta_true.max(), theta_pred.max())
    plt.plot([min_t, max_t], [min_t, max_t], "r--")
    plt.xlabel(r"$\theta_{\rm true}$ [degrees]")
    plt.ylabel(r"$\theta_{\rm pred}$ [degrees]")
    #plt.title(r"$\theta$ scatter plot")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 90)
    plt.ylim(0, 90)
    #plt.legend()

    plt.subplot(1, 2, 2)
    #plt.scatter(phi_true, phi_pred, s=1, alpha=0.7)
    hb = plt.hexbin(phi_true, phi_pred, gridsize=100, bins='log')
    plt.colorbar(hb, label='log10(N)')
    min_p = min(phi_true.min(), phi_pred.min())
    max_p = max(phi_true.max(), phi_pred.max())
    plt.plot([min_p, max_p], [min_p, max_p], "r--")
    plt.xlabel(r"$\phi_{\rm true}$ [degrees]")
    plt.ylabel(r"$\phi_{\rm pred}$ [degrees]")
    #plt.title(r"$\phi$ scatter plot")
    plt.grid(True, alpha=0.3)
    # full circle now; a sin/cos head should fill the diagonal corner to
    # corner with no gap at the wrap point
    plt.xlim(phi_lo, phi_lo + 360)
    plt.ylim(phi_lo, phi_lo + 360)
    #plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "resolutionScatter.pdf"), bbox_inches="tight")

    # ---------------------------------------------------
    # 3. |(s, c)| -- a HEALTH CHECK, not a confidence.  The (n-1)^2 anchor
    #    pins the output norm near 1, so the left panel should be a narrow
    #    spike at 1 and the right panel should be FLAT.  A spread-out left
    #    panel means the anchor is losing to weight_decay; a falling right
    #    panel means a radial degree of freedom has crept back in and the
    #    m = 4 artifact may be back with it.
    # ---------------------------------------------------
    fig3, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5))

    axA.hist(phi_conf, bins=60, histtype="step", linewidth=2)
    axA.axvline(1.0, color="0.4", ls="--", lw=1.0)
    axA.set_xlabel(r"$|(s,c)|$   (anchored to 1, NOT a confidence)",
                   fontsize=12)
    axA.set_ylabel("Entries")
    axA.set_title(r"output norm $|(s,c)|$  --  want a spike at 1")
    axA.grid(True, alpha=0.3)

    nb = 10
    edges = np.linspace(phi_conf.min(), phi_conf.max(), nb + 1)
    xs, ys, ns = [], [], []
    for k in range(nb):
        m = (phi_conf >= edges[k]) & (phi_conf < edges[k + 1])
        if m.sum() >= 20:
            xs.append(0.5 * (edges[k] + edges[k + 1]))
            ys.append(float(np.percentile(np.abs(phi_res[m]), 68.0)))
            ns.append(int(m.sum()))
    if xs:
        axB.plot(xs, ys, "o-", linewidth=2)
        for x, y, nn_ in zip(xs, ys, ns):
            axB.annotate("%d" % nn_, (x, y), textcoords="offset points",
                         xytext=(0, 7), ha="center", fontsize=6, color="0.4")
    axB.set_xlabel(r"$|(s,c)|$", fontsize=12)
    axB.set_ylabel(r"68% of $|\Delta\phi|$ [degrees]", fontsize=12)
    axB.set_title(r"$\phi$ resolution vs $|(s,c)|$  --  want it flat")
    axB.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(os.path.join(outdir, "phi_confidence.pdf"),
                 bbox_inches="tight")



#    plt.show()



# ============================================================================
# Model export
#
# Everything needed to run the trained network on other data is written into
# <outdir>/model/.  The point is that nothing in that directory depends on
# this training script: model_scripted.pt is TorchScript, so it carries its
# own architecture, and model_config.json carries every constant needed to
# turn raw volts into degrees.
# ============================================================================

PREDICT_PY = '''#!/usr/bin/env python3
"""
predict.py -- run the trained muon-direction network on new events.

    python predict.py EVENTS.txt
    python predict.py EVENTS.txt -o mypreds.csv
    python predict.py EVENTS.txt --only-passing      # drop events failing the
                                                     # training voltage cut

Everything it needs sits next to this file:

    model_scripted.pt   the network itself (TorchScript: no class definition
                        and no training script required)
    model_config.json   the constants -- input normalisation, theta
                        standardisation, phi range, event-selection cut

Output CSV columns
------------------
    event            index in the input file (before any filtering)
    theta_pred       degrees
    phi_pred         degrees, wrapped into the training range
    phi_conf         |(s, c)|.  This model's loss anchors it near 1, so it
                     is a health check on the network, NOT a per-event
                     confidence.  Do not cut on it.
    passed_cut       1/0, the training voltage cut applied to this event
    theta_true, phi_true, dtheta, dphi      only if the file has truth labels

IMPORTANT: the network only ever saw events passing the voltage cut, so a
prediction for an event with passed_cut = 0 is an extrapolation.  Those rows
are kept by default and flagged, not silently dropped.

Use it as a library instead of a CLI:

    from predict import load_model, read_events, predict
    model, cfg = load_model()
    images, truth = read_events("EVENTS.txt", cfg)
    theta, phi, conf = predict(model, cfg, images)
"""

import argparse
import json
import os
import sys

import warnings

import numpy as np
import torch

warnings.simplefilter("ignore", FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(model_dir=None):
    """Return (torchscript_model, config_dict)."""
    d = model_dir or HERE
    cfg_path = os.path.join(d, "model_config.json")
    if not os.path.isfile(cfg_path):
        sys.exit("no model_config.json in %s" % d)
    with open(cfg_path) as f:
        cfg = json.load(f)

    ts = os.path.join(d, "model_scripted.pt")
    if os.path.isfile(ts):
        model = torch.jit.load(ts, map_location="cpu")
    else:
        sys.exit(
            "no model_scripted.pt in %s.\\n"
            "Fall back to model_state_dict.pt, which needs the class "
            "definitions from the training script copied into the parent "
            "directory." % d)
    model.eval()
    return model, cfg


def passes_voltage_cut(image, cfg):
    sel = cfg["selection"]
    above = image >= sel["volt_threshold_V"]
    per_tile = np.sum(above, axis=0)
    return int(np.sum(per_tile >= sel["min_sipms_per_tile"])
               >= sel["min_tiles_per_event"])


def read_events(path, cfg, max_events=None):
    """Parse the same text format the network was trained on.

    Returns (images (N,4,T) float32 in VOLTS, truth (N,2) or None).
    """
    fmt = cfg["format"]
    nrow = fmt["n_rows"]
    ntile = fmt["num_tiles"]
    nskip = fmt["skip_leading_blocks"]

    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]

    images, truth = [], []
    have_truth = True
    i, n = 0, len(lines)
    while i < n:
        head = lines[i].split()
        if len(head) < 2:
            raise ValueError("malformed header at line %d: %s" % (i + 1,
                                                                  lines[i]))
        try:
            truth.append([float(head[0]), float(head[1])])
        except ValueError:
            have_truth = False
            truth.append([np.nan, np.nan])

        for _ in range(nskip * nrow):        # blocks the trainer skipped
            i += 1
            if i >= n:
                raise ValueError("file ends inside a skipped block")

        rows = []
        for _ in range(nrow):
            i += 1
            if i >= n:
                raise ValueError("file ends inside an image block")
            v = [float(x) for x in lines[i].split()]
            if len(v) != ntile:
                raise ValueError(
                    "line %d: %d values, expected %d.  This model was trained "
                    "on %d tiles." % (i + 1, len(v), ntile, ntile))
            rows.append(v)
        images.append(rows)
        i += 1
        if max_events is not None and len(images) >= max_events:
            break

    images = np.asarray(images, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32) if have_truth else None
    return images, truth


def predict(model, cfg, images, batch_size=256):
    """images (N,4,T) in VOLTS -> (theta_deg, phi_deg, phi_conf)."""
    pre = cfg["preprocessing"]
    post = cfg["postprocessing"]

    x = (np.asarray(images, dtype=np.float32) - pre["img_mean"]) \\
        / pre["img_std"]
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
    conf = np.hypot(s, c)
    return theta, phi, conf


def delta_phi_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("eventfile", help="text file in the training format")
    p.add_argument("-o", "--out", default=None,
                   help="output CSV (default: EVENTFILE_pred.csv)")
    p.add_argument("--model-dir", default=None,
                   help="directory holding model_scripted.pt and "
                        "model_config.json (default: next to this script)")
    p.add_argument("--only-passing", action="store_true",
                   help="write only events that pass the training voltage "
                        "cut (default: write all, flagged by passed_cut)")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    a = p.parse_args(argv)

    model, cfg = load_model(a.model_dir)
    print("model      %s  (trained %s)"
          % (cfg["provenance"]["source_script"],
             cfg["provenance"]["trained_at"]))
    print("expects    %d rows x %d tiles, volts"
          % (cfg["format"]["n_rows"], cfg["format"]["num_tiles"]))

    images, truth = read_events(a.eventfile, cfg, a.max_events)
    print("read       %d events from %s" % (images.shape[0], a.eventfile))

    passed = np.array([passes_voltage_cut(im, cfg) for im in images],
                      dtype=int)
    print("cut        %d/%d pass the training voltage cut (%.1f%%)"
          % (passed.sum(), passed.size,
             100.0 * passed.sum() / max(passed.size, 1)))

    theta, phi, conf = predict(model, cfg, images, a.batch_size)

    idx = np.arange(images.shape[0])
    keep = passed.astype(bool) if a.only_passing else np.ones_like(passed,
                                                                  dtype=bool)

    cols = ["event", "theta_pred", "phi_pred", "phi_conf", "passed_cut"]
    data = [idx[keep], theta[keep], phi[keep], conf[keep], passed[keep]]
    if truth is not None:
        tt, pt = truth[:, 0], truth[:, 1]
        pt = (pt - cfg["postprocessing"]["phi_lo"]) % 360.0 \\
            + cfg["postprocessing"]["phi_lo"]
        cols += ["theta_true", "phi_true", "dtheta", "dphi"]
        data += [tt[keep], pt[keep], (theta - tt)[keep],
                 delta_phi_deg(phi, pt)[keep]]

    out_path = a.out or (os.path.splitext(a.eventfile)[0] + "_pred.csv")
    with open(out_path, "w") as f:
        f.write(",".join(cols) + "\\n")
        for r in range(int(keep.sum())):
            f.write(",".join(("%d" % col[r]) if cols[j] in ("event",
                                                            "passed_cut")
                             else ("%.6f" % col[r])
                             for j, col in enumerate(data)) + "\\n")

    if truth is not None:
        m = passed.astype(bool)
        dth = (theta - truth[:, 0])[m]
        dph = delta_phi_deg(phi, truth[:, 1])[m]
        if dth.size:
            print("resolution on the %d events passing the cut:" % dth.size)
            print("  theta   bias %+7.3f   RMS %7.3f   68%% of |res| %7.3f deg"
                  % (dth.mean(), dth.std(), np.percentile(np.abs(dth), 68)))
            print("  phi     bias %+7.3f   RMS %7.3f   68%% of |res| %7.3f deg"
                  % (dph.mean(), dph.std(), np.percentile(np.abs(dph), 68)))
    print("wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


MODEL_README = '''# Trained muon-direction network

Self-contained: nothing here needs the training script.

## Files

| file | what it is |
|---|---|
| `model_scripted.pt`   | the network as TorchScript -- architecture + weights in one file, loads with `torch.jit.load`, no class definitions needed |
| `model_state_dict.pt` | plain weights; needs `MuonDirectionCNN` from the training script (a copy of which sits in the parent directory) |
| `model_config.json`   | every constant needed to go from raw volts to degrees, plus training provenance and the resolutions achieved |
| `predict.py`          | standalone CLI / importable module that does the whole chain |
| `model_summary.txt`   | layer list, parameter counts, tensor shape at each stage |
| `training_history.csv`| train/test loss per epoch |
| `test_predictions.csv`| per-event predictions on the held-out test set |

## Run it on new events

```
python predict.py EVENTS.txt
python predict.py EVENTS.txt -o mypreds.csv --only-passing
```

`EVENTS.txt` must be in the same format the network was trained on: a header
line (`theta phi z`), then the leading block(s) the trainer skipped, then the
4 x N image. If the header carries real truth values the script also prints
the achieved resolution; if it does not, the prediction columns are still
written.

## Use it from your own code

```python
from predict import load_model, read_events, predict
model, cfg = load_model()                       # reads this directory
images, truth = read_events("EVENTS.txt", cfg)  # (N, 4, T) in volts
theta, phi, conf = predict(model, cfg, images)  # all in degrees
```

## Or drive the network directly

```python
import json, numpy as np, torch
cfg   = json.load(open("model_config.json"))
model = torch.jit.load("model_scripted.pt", map_location="cpu").eval()

x = (images - cfg["preprocessing"]["img_mean"]) / cfg["preprocessing"]["img_std"]
out = model(torch.from_numpy(x[:, None, :, :]).float()).detach().numpy()

theta = out[:, 0] * cfg["postprocessing"]["theta_std"] \\
        + cfg["postprocessing"]["theta_mean"]
phi   = np.degrees(np.arctan2(out[:, 1], out[:, 2]))
phi   = (phi - cfg["postprocessing"]["phi_lo"]) % 360 \\
        + cfg["postprocessing"]["phi_lo"]
conf  = np.hypot(out[:, 1], out[:, 2])
```

## Three things that will bite you

1. **Normalise the input.** The network was trained on
   `(volts - img_mean) / img_std` with the constants in `model_config.json`,
   which came from the training sample. Feeding raw volts gives nonsense.
2. **theta is standardised, phi is not.** Output 0 is `theta_norm` and needs
   `* theta_std + theta_mean`. Outputs 1 and 2 are `(sin phi, cos phi)` and
   need `atan2`, not a rescale.
3. **The voltage cut is part of the model.** Training only ever saw events
   passing it, so predictions on failing events are extrapolation.
   `predict.py` flags them in `passed_cut`.
'''


def export_trained_model(outdir, model, source_script, input_file, device,
                         img_mean, img_std, theta_mean, theta_std, phi_lo,
                         w_norm, n_train, n_test, train_losses, test_losses,
                         theta_true, phi_true, theta_pred, phi_pred,
                         phi_conf, theta_res, phi_res):
    """Write everything needed to re-run this network elsewhere."""
    mdir = os.path.join(outdir, "model")
    os.makedirs(mdir, exist_ok=True)

    # always export from a CPU copy, so the files are portable off this
    # machine (an MPS or CUDA tensor will not load on a plain CPU box)
    m = copy.deepcopy(model).to("cpu").eval()

    torch.save(m.state_dict(), os.path.join(mdir, "model_state_dict.pt"))

    scripted = None
    example = torch.zeros(2, 1, 4, NUM_TILES, dtype=torch.float32)
    # torch >= 2.14 emits a FutureWarning for torch.jit; it still works and is
    # by far the most portable single-file export, so keep it and stay quiet
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            scripted = torch.jit.script(m)
            how = "torch.jit.script"
        except Exception as e_script:
            try:
                with torch.no_grad():
                    scripted = torch.jit.trace(m, example)
                how = ("torch.jit.trace (script failed: %s)"
                       % type(e_script).__name__)
            except Exception as e_trace:
                how = "FAILED: %s / %s" % (e_script, e_trace)
        if scripted is not None:
            scripted.save(os.path.join(mdir, "model_scripted.pt"))
            # prove the exported file reproduces the live model
            with torch.no_grad():
                a = m(example).numpy()
                reloaded = torch.jit.load(os.path.join(mdir,
                                                       "model_scripted.pt"))
                b = reloaded(example).numpy()
            max_dev = float(np.max(np.abs(a - b)))
        else:
            max_dev = float("nan")

    # ---- architecture summary, with the shape at each stage ----
    shapes = []
    hooks = []

    def hook(name):
        def fn(_mod, _inp, out):
            if isinstance(out, torch.Tensor):
                shapes.append((name, tuple(out.shape)))
        return fn

    for name, child in m.named_children():
        hooks.append(child.register_forward_hook(hook(name)))
    with torch.no_grad():
        m(example)
    for h in hooks:
        h.remove()

    n_par = sum(p.numel() for p in m.parameters())
    n_trn = sum(p.numel() for p in m.parameters() if p.requires_grad)

    with open(os.path.join(mdir, "model_summary.txt"), "w") as f:
        f.write("MuonDirectionCNN\n")
        f.write("=" * 70 + "\n")
        f.write("input  : (B, 1, 4, %d)   normalised volts\n" % NUM_TILES)
        f.write("output : (B, 3) = theta_norm, sin_phi, cos_phi\n")
        f.write("params : %d total, %d trainable\n" % (n_par, n_trn))
        f.write("export : %s   (max |scripted - live| = %.3g)\n"
                % (how, max_dev))
        f.write("=" * 70 + "\n\n")
        f.write("tensor shape after each top-level stage "
                "(batch of 2, note the circular pad 4 -> 7 rows):\n")
        for name, shp in shapes:
            f.write("    %-10s %s\n" % (name, shp))
        f.write("\n" + "=" * 70 + "\n\n")
        f.write(str(m) + "\n\n")
        f.write("=" * 70 + "\n")
        f.write("%-52s %12s %10s\n" % ("parameter", "shape", "count"))
        for name, prm in m.named_parameters():
            f.write("%-52s %12s %10d\n"
                    % (name, "x".join(str(v) for v in prm.shape), prm.numel()))

    # ---- training history ----
    with open(os.path.join(mdir, "training_history.csv"), "w") as f:
        f.write("epoch,train_loss,test_loss\n")
        for k, (a_, b_) in enumerate(zip(train_losses, test_losses), 1):
            f.write("%d,%.8g,%.8g\n" % (k, a_, b_))

    # ---- per-event predictions on the held-out test set ----
    with open(os.path.join(mdir, "test_predictions.csv"), "w") as f:
        f.write("theta_true,phi_true,theta_pred,phi_pred,phi_conf,"
                "dtheta,dphi\n")
        for k in range(len(theta_true)):
            f.write("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n"
                    % (theta_true[k], phi_true[k], theta_pred[k], phi_pred[k],
                       phi_conf[k], theta_res[k], phi_res[k]))

    # ---- resolutions, so the config records what this model actually does ----
    s_th, s_th_e, _ = fit_gaussian_sigma_and_uncertainty(theta_res, bins=100,
                                                         max_sigma=90.0)
    s_ph, s_ph_e, _ = fit_gaussian_sigma_and_uncertainty(phi_res, bins=100,
                                                         max_sigma=180.0)

    def _f(v):
        return None if v is None or not np.isfinite(v) else float(v)

    cfg = {
        "model": {
            "class": "MuonDirectionCNN",
            "description": "ResNet-18 style, circular pad on the azimuthal "
                           "(4-row) axis, global average pool, 3 outputs",
            "input_shape": [1, 4, NUM_TILES],
            "input_units": "volts, standardised with preprocessing below",
            "outputs": ["theta_norm", "sin_phi", "cos_phi"],
            "n_parameters": int(n_par),
            "torchscript_export": how,
            "torchscript_max_deviation": _f(max_dev),
        },
        "format": {
            "n_rows": 4,
            "num_tiles": int(NUM_TILES),
            "skip_leading_blocks": 1,
            "header_fields": ["theta_deg", "phi_deg", "z"],
            "note": "one event = header line, then skip_leading_blocks "
                    "blocks of n_rows lines, then the n_rows x num_tiles "
                    "image; blank lines ignored",
        },
        "preprocessing": {
            "img_mean": float(img_mean),
            "img_std": float(img_std),
            "formula": "x = (volts - img_mean) / img_std",
        },
        "postprocessing": {
            "theta_mean": float(theta_mean),
            "theta_std": float(theta_std),
            "phi_lo": float(phi_lo),
            "theta_formula": "theta_deg = out[0] * theta_std + theta_mean",
            "phi_formula": "phi_deg = (degrees(atan2(out[1], out[2])) "
                           "- phi_lo) % 360 + phi_lo",
            "confidence_formula": "phi_conf = hypot(out[1], out[2]). ANCHORED near 1 by the loss -- a health check, NOT a per-event confidence.",
        },
        "selection": {
            "volt_threshold_V": float(VOLT_THRESHOLD_V),
            "min_sipms_per_tile": int(MIN_SIPMS_PER_TILE),
            "min_tiles_per_event": int(MIN_TILES_PER_EVENT),
            "note": "applied to RAW volts before normalisation; the network "
                    "never saw failing events",
        },
        "training": {
            "epochs": int(NUM_EPOCHS),
            "batch_size": int(BATCH_SIZE),
            "optimizer": "Adam",
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR(T_max=%d)" % NUM_EPOCHS,
            "loss": "dtheta^2 + [(s/n - sin phi)^2 + (c/n - cos phi)^2] "
                    "+ w_norm * (n-1)^2,  n = |(s,c)|;  dtheta = standardised "
                    "theta residual * theta_std * pi/180.  Both angular terms "
                    "in radians, 1:1 per degree, no w_phi.  (s,c) normalised "
                    "before the phi term so the angular gradient does not "
                    "scale with |(s,c)|.",
            "w_norm": float(w_norm),
            "seed": 42,
            "n_train": int(n_train),
            "n_test": int(n_test),
            "final_train_loss": _f(train_losses[-1] if train_losses else None),
            "final_test_loss": _f(test_losses[-1] if test_losses else None),
        },
        "performance_on_test_set": {
            "n_events": int(len(theta_true)),
            "theta_sigma_gauss_deg": _f(s_th),
            "theta_sigma_gauss_err_deg": _f(s_th_e),
            "theta_bias_deg": float(np.mean(theta_res)),
            "theta_rms_deg": float(np.std(theta_res)),
            "theta_68pct_abs_deg": float(np.percentile(np.abs(theta_res), 68)),
            "phi_sigma_gauss_deg": _f(s_ph),
            "phi_sigma_gauss_err_deg": _f(s_ph_e),
            "phi_bias_deg": float(np.mean(phi_res)),
            "phi_rms_deg": float(np.std(phi_res)),
            "phi_68pct_abs_deg": float(np.percentile(np.abs(phi_res), 68)),
            "phi_conf_median": float(np.median(phi_conf)),
        },
        "provenance": {
            "trained_at": datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"),
            "source_script": os.path.basename(source_script),
            "training_file": os.path.abspath(input_file),
            "output_dir": os.path.abspath(outdir),
            "device": str(device),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    with open(os.path.join(mdir, "model_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    pp = os.path.join(mdir, "predict.py")
    with open(pp, "w") as f:
        f.write(PREDICT_PY)
    try:
        os.chmod(pp, 0o755)
    except OSError:
        pass

    with open(os.path.join(mdir, "README.md"), "w") as f:
        f.write(MODEL_README)

    print("")
    print("=" * 70)
    print("Model exported to %s" % mdir)
    print("  model_scripted.pt      %s" % how)
    if np.isfinite(max_dev):
        print("                         verified: max |scripted - live| "
              "= %.3g" % max_dev)
    print("  model_config.json      normalisation, decoding, cut, provenance")
    print("  predict.py             python predict.py EVENTS.txt")
    print("  README.md              how to use all of it")
    print("  model_summary.txt      %d parameters" % n_par)
    print("  training_history.csv   %d epochs" % len(train_losses))
    print("  test_predictions.csv   %d events" % len(theta_true))
    print("=" * 70)
    return mdir

def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))





def add_residual_hist_with_fit_binned(ax, data, bins, xlabel, title,
                                      xlim=None, max_sigma=100):
    """
    Version of the residual+Gaussian plot tuned for theta_true-binned panels.
    - Fixed x-limits (via xlim)
    - Compact legend in upper-right
    - Smaller fonts to avoid crowding
    - Reject pathological Gaussian fits (sigma > max_sigma or <= 0)
    """
    data = np.asarray(data)
    entries = data.size
    if entries == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Entries", fontsize=8)
        ax.set_title(title, fontsize=9)
        return

    # Histogram (counts, not density)
    counts, edges, _ = ax.hist(
        data,
        bins=bins,
        histtype="step",
        linewidth=1.2,
        color="tab:blue",
        density=False,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Basic stats
    mean = data.mean()
    rms = data.std(ddof=1)

    hist_text = f"N={entries}, μ={mean:.2f}, RMS={rms:.2f}"

    # Try Gaussian fit
    p0 = [counts.max(), mean, rms if rms > 0 else 1.0]
    fit_text = None
    try:
        popt, _ = curve_fit(gauss, centers, counts, p0=p0, maxfev=10000)
        A_fit, mu_fit, sigma_fit = popt

        # Reject pathological fits
        if 0 < sigma_fit < max_sigma:
            x_fit = np.linspace(edges[0], edges[-1], 400)
            y_fit = gauss(x_fit, A_fit, mu_fit, sigma_fit)
            ax.plot(x_fit, y_fit, color="orange", linewidth=1.5)
            fit_text = f"fit μ={mu_fit:.2f}, σ={sigma_fit:.2f}"
            #print(xlabel)
            if "theta" in xlabel:
                print(sigma_fit)


        ax.set_ylim(0, 1.4 * counts.max())
    except Exception:
        ax.set_ylim(0, 1.4 * counts.max())
        pass  # just skip the fit

    # Axes cosmetics
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.4 * max(counts.max(), 1))
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Entries", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=8)

    # Compact legend
    legend_lines = [hist_text]
    if fit_text is not None:
        legend_lines.append(fit_text)

    ax.legend(
        legend_lines,
        loc="upper right",
        fontsize=5,
        frameon=True,
        framealpha=0.9,
        handlelength=1.0,
        borderpad=0.2,
    )


def add_residual_hist_with_fit(ax, data, bins, xlabel, title, xlim=None, fit_range=None):
    """
    Plot a 1D residual histogram, fit a Gaussian, and show a legend with:
      - Hist: N, mean, RMS
      - Fit:  mu, sigma
    """
    data = np.asarray(data)
    entries = data.size
    if entries == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center")
        return

    # Histogram (counts, not density)
    counts, edges, _ = ax.hist(
        data,
        bins=bins,
        histtype="step",
        linewidth=2,
        label=None,
        density=False,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Basic stats from histogram/data
    mean = data.mean()
    rms = data.std(ddof=1)

    # If requested, restrict the Gaussian fit to a limited x-range
    centers_fit = centers
    counts_fit  = counts
    fit_lo, fit_hi = edges[0], edges[-1]

    if fit_range is not None:
        fit_lo, fit_hi = fit_range
        m = (centers >= fit_lo) & (centers <= fit_hi)   # inclusive range
        centers_fit = centers[m]
        counts_fit  = counts[m]

        # If the requested window is too narrow / empty, skip the fit gracefully
        if centers_fit.size < 3 or np.count_nonzero(counts_fit) < 3:
            raise RuntimeError("Too few histogram bins in fit_range for a stable fit.")

    
    # Gaussian fit (in y=counts)
    #p0 = [counts.max(), mean, rms if rms > 0 else 1.0]
    p0 = [counts_fit.max(), mean, rms if rms > 0 else 1.0]
    
    try:
        #popt, _ = curve_fit(gauss, centers, counts, p0=p0, maxfev=10000)
        popt, _ = curve_fit(gauss, centers_fit, counts_fit, p0=p0, maxfev=10000)
        A_fit, mu_fit, sigma_fit = popt

        #x_fit = np.linspace(edges[0], edges[-1], 400)
        x_fit = np.linspace(fit_lo, fit_hi, 400)
        y_fit = gauss(x_fit, A_fit, mu_fit, sigma_fit)
        ax.plot(
            x_fit,
            y_fit,
            color="orange",
            linewidth=2,
            label=f"Gauss fit (μ={mu_fit:.2f}, σ={sigma_fit:.2f})",
        )

        ax.set_ylim(0, 1.2 * counts.max())
        
        hist_label = f"Hist (N={entries}, mean={mean:.2f}, RMS={rms:.2f})"
        ax.legend(title=hist_label, loc="upper right")

    except RuntimeError:
        # Fit failed – still show hist stats
        hist_label = f"Hist (N={entries}, mean={mean:.2f}, RMS={rms:.2f})"
        ax.legend(title=hist_label, loc="upper right")
        ax.set_ylim(0, 1.2 * counts.max())

    # Axes cosmetics
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Entries")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

def fit_gaussian_sigma_and_uncertainty(data, bins, max_sigma=100.0):
    """
    Fit a Gaussian to a residual distribution histogram and return:
      sigma, sigma_uncertainty, N_entries

    Uses curve_fit on histogram counts vs bin centers.
    Returns (np.nan, np.nan, N) if fit fails or is pathological.
    """
    data = np.asarray(data, dtype=float)
    n = data.size
    if n < 10:
        return np.nan, np.nan, n

    counts, edges = np.histogram(data, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Require enough nonzero bins to fit sensibly
    nonzero = counts > 0
    if np.count_nonzero(nonzero) < 3:
        return np.nan, np.nan, n

    mean = data.mean()
    rms = data.std(ddof=1) if n > 1 else 0.0
    if not np.isfinite(rms) or rms <= 0:
        rms = 1.0

    p0 = [counts.max(), mean, rms]

    try:
        popt, pcov = curve_fit(gauss, centers, counts, p0=p0, maxfev=20000)
        _, mu_fit, sigma_fit = popt

        if (not np.isfinite(sigma_fit)) or sigma_fit <= 0 or sigma_fit > max_sigma:
            return np.nan, np.nan, n

        # Uncertainty on sigma from covariance matrix
        sigma_err = np.sqrt(pcov[2, 2]) if (pcov is not None and pcov.shape == (3, 3) and pcov[2, 2] >= 0) else np.nan
        if not np.isfinite(sigma_err):
            sigma_err = np.nan

        return float(sigma_fit), float(sigma_err), n

    except Exception:
        return np.nan, np.nan, n

    
if __name__ == "__main__":
    main()
