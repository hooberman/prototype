# 16-Channel SiPM Array Board — Readme

## Overview
- 16x onsemi MicroFJ-60035-TSV SiPMs (6.13x6.13mm package) in a single row
- Board size: 339.13mm x 15.5mm, 4-layer
- Each SiPM has its own independent anode and cathode connection (32 signals total)
- Fast output pin is not used/routed (per request)
- All 32 connections terminate in two rows of through-hole solder pads at one end (connector "J1"), sized for hand-soldered wire leads

## Geometry
- SiPM pitch: 19.6mm center-to-center (294mm span from SiPM 1 to SiPM 16, plus 6.13mm SiPM width = 300mm / 30cm edge-to-edge, matching the scintillator length)
- 3mm clearance maintained from the outermost SiPM edges to the board edges in both length and width
- Board width (15.5mm) is wider than the strict minimum (12.13mm) needed just for SiPM+margin — the extra width was required to route 32 independent signals with real electrical clearance (see "Why 15.5mm" below)

## Layer stack
- **F.Cu**: SiPM pads (SMD) + connector pads (through-hole) + ground pour
- **In1.Cu**: all 16 anode trunk traces
- **In2.Cu**: all 16 cathode trunk traces
- **B.Cu**: ground pour, stitched to F.Cu ground pour with a few vias in clear areas

Anode and cathode vias from each SiPM are **blind vias** (F.Cu <-> In1.Cu for anode, F.Cu <-> In2.Cu for cathode) — this is a normal, moderate-cost option at most fabs for a 4-layer board (sometimes called "1+2+1" build), and it's what makes the tight trace spacing electrically safe: a blind via only has copper on its two specified layers, not the whole board thickness.

## Channel-to-pad mapping (important!)
To route 32 independent signals without any traces crossing on the same layer, the connector pad **order does not match the physical SiPM order**. Each pad is silkscreened with its channel number (A1-A16 for anodes, K1-K16 for cathodes) — always solder by label, not by position along the row.

## Why 15.5mm wide
With 16 independent anode + 16 independent cathode lines converging on one end, each pair of adjacent lines needs real clearance (trace width + electrical clearance, calculated at 0.15mm/0.15mm here — comfortably inside what any standard fab, e.g. JLCPCB/PCBWay, supports). At the originally-implied 12.13mm width (SiPM size + 3mm margins only), there wasn't physically enough room to fit 32 such lines with safe clearance between the SiPM's via positions and the board edges. 15.5mm provides that room with a small margin to spare.

## Fabrication notes
- 4-layer board, standard FR4, 1.6mm thickness assumed (adjust in fab's order form if you want thinner)
- Minimum trace/space used: 0.15mm / 0.15mm (JLCPCB "standard" capability tier, no premium options needed)
- Minimum via: 0.35mm finished diameter / 0.2mm drill (also standard capability)
- Blind vias (F.Cu<->In1.Cu and F.Cu<->In2.Cu): flag this explicitly when ordering — most fabs need "HDI" or "blind/buried via" selected as an option
- Connector pads: 1.0mm diameter / 0.6mm drill, through-hole, suitable for 22-26AWG hookup wire

## Assembly notes
- The bare TSV SiPM package has 36 tiny solder bumps on a 0.565mm/1.0mm pitch grid — this requires solder paste + stencil + reflow, not hand soldering. Only 2 of the 36 pads per chip are actually used electrically (C1=anode, A1=cathode); the rest are either redundant duplicates of those same nets (D1, F6) or unconnected pads tied to the ground pour for thermal relief, per the datasheet's recommendation.
- The 32 connector pads ARE meant for hand-soldered wires.

## Files
- `sipm_array_16ch.kicad_pcb` — full KiCad 7 project file (open in KiCad to inspect, run DRC yourself, or make changes)
- `sipm_array_16ch_gerbers.zip` — Gerbers + drill files, ready to upload to a fab (JLCPCB, PCBWay, OSH Park, etc.)

## What I verified programmatically before generating these files
- Every SiPM's anode/cathode pad connects to the correct unique net (no cross-wiring) — checked all 16 channels
- Every connector pad's net matches its printed label
- A geometric clearance check (accounting for real via/trace/pad diameters, not just centerlines, and accounting for blind vias spanning only their intended layers) found zero violations among all 130 copper features

I'd still recommend opening the .kicad_pcb in KiCad and running a full DRC yourself before ordering — my check covers the specific short-circuit risk this design faced, but KiCad's built-in DRC checks many other rules (drill-to-copper, silkscreen-over-pad, etc.) that are worth a final look.
