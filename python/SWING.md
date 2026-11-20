`swing.py` is a separate, capture-specific processor for
`/mnt/dataz/pictures/linescan/2026-09-03-00-31-12/`.

```sh
uv run python python/swing.py
```

The default destination is `swing_processed/` inside that capture. A run writes
15 numbered, full-resolution RGBA PNG wedges and corresponding 16-bit linear
RGB DNGs, plus `contact_sheet.jpg`, `uncorrected_overview.jpg`, `motion_fit.png`,
and `swing.json`. The JSON records the annotations, measured correlations,
motion parameters, pixel geometry, and display settings. All wedges have the
same 0.56-radian span (about 32.1 degrees), canvas size, and pivot position.
The default contact sheet is 4,680 by 10,200 pixels, with up to 1,920-pixel-tall
thumbnails; `--contact-height` controls their resolution. Choose a new
`--output-dir` for another run; existing output directories are protected.

```sh
# Fast visual check of selected passages; fitting still uses all 15.
uv run python python/swing.py --scale 0.25 --contact-height 800 --passes 4 5 8 --output-dir /tmp/swing-preview

# Compare the constant-speed approximation over each short passage.
uv run python python/swing.py --motion constant --output-dir /tmp/swing-constant
```

The 15 passage centers were annotated by inspecting an uncorrected Bayer-cell
preview. Time is measured in Bayer-binned columns, matching the coordinates in
`speed_constraints.py`; the capture does not record the line rate needed to
convert these to seconds. Same-direction crossings give a roughly 6,020-column
period. `--centers` and `--period` override these motion annotations and period.
Separate, visually annotated `FRAMING_CENTERS` center the head/seat/feet
silhouette in each wedge; `--framing-centers` overrides them. They select an
angular crop within the original motion model without moving its speed maximum.
`--angle-width` sets the common full angular span in radians. For compatibility,
`--half-window` can instead choose that common span using the specified time
margin at the median passage speed. Each slice then uses its own required time
interval, so faster and slower passages retain identical angular bounds.

The existing green-plane correlation estimator measures signed temporal lag
on short windows in five height bands covering the chains and seat. Lag is
the reciprocal of tangential speed in full sensor pixels per time column.
A robust joint fit uses `speed(y) = angular_speed * (y - pivot_y)` to infer one
fixed projected pivot above the scanning line and an amplitude for each pass.
The default model varies angular speed as a cosine about each crossing and
integrates it to obtain angle; it never integrates through a reversal. The fit
for this capture puts the pivot about 1,534 full sensor pixels above the image,
with a median relative lag residual around 2.1%. The default then applies a
modest 200-pixel downward visual adjustment and refits all passage amplitudes at
the adjusted shared pivot. This slightly reduces the relative width at the top
and increases it at the feet. It is an aesthetic correction, not stronger
correlation evidence for the pivot. Both the unconstrained estimate and applied
adjustment are recorded in JSON. `--pivot-offset 0` restores the correlation
estimate; `--pivot-y` overrides the final pivot with an absolute negative row.
These are approximate geometric estimates; overlapping windows do not provide
independent uncertainty bounds.

Inverse polar mapping makes stationary background rows into concentric circular
arcs. Both travel directions use the same camera orientation. Each Bayer plane
is sampled separately, including its vertical offset and the radius-dependent
delay between scan planes, before the two greens are averaged. At `--scale 1`,
output pixels have the full sensor's spatial pitch. Since the pivot is outside
the scan, the measured area is an annular sector with a curved top and bottom.

PNG previews share white balance, exposure, and gamma. The DNG pixels retain
camera-linear values, with color and white balance stored as metadata; these
are demosaiced/resampled linear DNGs rather than Bayer mosaics. Unmeasured pixels
are transparent in PNGs and black in DNGs. A small neutral walkway region is
used for this capture's white balance. The child's independent limb motion is
not modeled, and overlaps with the second swing remain visible.

The processor reads existing helper functions and adds no behavior to the train
or panorama scripts. It stages a complete output directory before publishing
it, and does not replace existing renders or editing sidecars.

```sh
PYTHONPATH=python uv run python -m unittest python/test_swing.py
```
