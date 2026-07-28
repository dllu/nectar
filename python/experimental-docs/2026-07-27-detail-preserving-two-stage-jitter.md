# Detail-preserving two-stage jitter correction — 2026-07-27

> **Updated later on 2026-07-27.** The two-stage estimator remains in use, but
> its unrestricted additive application still flattened low-signal bellows
> and wheel detail. See
> `2026-07-27-shadow-detail-jitter-protection.md` for the reproduced failure
> and the scene-aware application limit.
>
> **Geometry correction, 2026-07-28.** The raw-geometry isolation described
> below was incorrect and has been superseded by
> `2026-07-28-radiometric-first-geometry.md`.

## 1. Regression report

The local multiscale estimator documented as final in
`2026-07-27-clipped-roof-regression.md` was not safe enough to retain. Although
it substantially reduced stripes, it could not distinguish narrow scene
content from a narrow exposure error. The supplied MTR comparisons exposed
three severe consequences:

- recognizable wheel assemblies became smooth, ambiguous patches;
- window and cab edges acquired purple block-like color splotches; and
- the articulated bellows between cars lost most of its pleated structure.

These are algorithmic artifacts, not JPEG compression. The estimator was
replacing each row band's current value with a short temporal-median target.
Any real feature narrower than that median window could therefore be erased.
The structure mask reduced the failure but could not make the operation safe:
small dark details and low-contrast color edges do not always cross a fixed
edge threshold.

The MTR geometry regression was separate but related. At the time, it was
attributed to allowing a radiometric correction to affect the geometry fit.
The later capture replay showed the opposite cause: common capture-column
stripes bias an uncorrected green-plane correlation toward zero displacement.

## 2. Rejected alternatives

The following experiments were not retained:

- Direct local median correction removed the most stripe energy but reproduced
  the reported wheel, window, and bellows destruction.
- Common-sign and majority-channel gates still accepted scene features that
  happened to appear in several Bayer channels.
- A causal charge-state model left visible transitions and depended on scene
  order.
- PCA of only the redundant green difference could not predict the
  channel-specific red and blue response reliably.
- A low-rank model alone preserved detail, but left fine pair-specific stripes
  visible in the MTR wheels.

The useful result from the PCA experiments was a safe separation: recurrent
sensor behavior has broad row/channel support and recurs throughout a capture,
whereas a wheel edge, window frame, bellows pleat, or ventilation grill is
sparse in sensor-row support, time, or color channels.

## 3. Recurrent low-rank stage

The first stage still forms per-channel candidates from centered 31-column and
1023-column temporal medians in 256 independent vertical sensor bands.
Candidates are normalized by each channel's measured range and are capped at
0.5% before model fitting.

Before covariance estimation, candidates are robustly winsorized separately
for the two physical Bayer-column groups: red/green-1 and green-2/blue.
Eigenmodes are retained only when they:

1. exceed the covariance noise floor by at least 5×;
2. participate in at least 10% of sampled capture columns;
3. occupy at least 5% of the row/channel feature space; and
4. carry at least 5% of their power in every Bayer channel.

At most 64 modes are retained, learned from at most 32,768 columns distributed
across the complete selected capture. Projection accounts for clipped and
masked features rather than filling their values. This stage is deliberately
conservative: it removes recurrent sensor modes without locally fitting the
scene.

## 4. Motion-aligned fine residual stage

The low-rank stage leaves some high-rank, pair-specific line jitter. A second
stage revisits only the 31-column residual after the conservative correction.
It is enabled after radiometric-first motion estimation supplies the
displacement between green-1 and green-2.

For every vertical band:

1. Green-2 gradients are sampled at the raw subpixel displacement corresponding
   to each green-1 column.
2. A scene edge is protected only when both aligned greens corroborate its
   direction. Gradient thresholds are normalized by robust, capture-derived
   scales rather than a fixed exposure-dependent raw value.
3. Weak edges receive a one-column guard. Strong edges protect the complete
   15-column temporal-median radius so a narrow plateau cannot be hollowed out
   between its two boundaries.
4. A residual must have meaningful power in at least two Bayer channels.
   Single-channel color detail is never eligible.
5. Eligible corrections are capped at 0.6% of that channel's range and still
   receive the existing per-pixel black/highlight fade.

The recurrent and fine corrections are stored as separate components.
Vertical interpolation owner-gates each component independently. This matters
at a wheel, window, bellows, or clipped-roof boundary: a fine correction from
an adjacent sensor band cannot leak into a protected band merely because the
recurrent component is supported there.

## 5. Radiometric-first geometry

This section's original raw-geometry design has been superseded. Speed now
uses the motion-independent recurrent radiometric model, and skew uses the same
model while sampling at the fitted speed positions. Both fits bypass the
display-only scene-relative shadow cap. The motion-aligned fine residual and
scene-protection mask remain downstream rendering safeguards and are not fed
back into geometry. See `2026-07-28-radiometric-first-geometry.md` for the
capture replay and regression measurements.

## 6. Focused production replays

The exact production-selected MTR range (files 168–789, 159,232 Bayer-pair
columns) retained 53 recurrent modes. The complete two-stage model took about
89 seconds to estimate: about 48 seconds for candidates/low-rank fitting and
36 seconds for the guarded fine residual pass.

For the supplied wheel region, a rendered column-residual metric was computed
after subtracting each row's local median. Lower is less coherent vertical
striping:

| Model | B/G/R RMS (DN) | B/G/R p95 (DN) |
| --- | --- | --- |
| No correction | 2.815 / 1.307 / 1.358 | 6 / 3 / 3 |
| Conservative low-rank only | 1.113 / 0.747 / 0.637 | 2 / 1 / 1 |
| Two-stage production | 0.807 / 0.665 / 0.504 | 1.5 / 1 / 1 |
| Destructive local median | 0.703 / 0.586 / 0.462 | 1 / 1 / 1 |

The two-stage result approaches the stripe residual of the destructive method
without making the wheels unrecognizable. Native aligned comparisons also
showed the cab/window edges and pilot silhouette intact, without the new
purple patches. Against the conservative detail-preserving render, the
two-stage top-decile edge-magnitude ratio was 1.000 in the cab (correlation
0.993) and 0.933 in the bellows (correlation 0.967). The bellows' vertical
pleats remain visible.

The exact Pato selected range (files 2–165, 167,936 Bayer-pair columns) retained
32 recurrent modes and took 80.6 seconds for both stages. Its guarded fine
component is active in 56.6% of band/channel samples, but the independently
owner-gated edge masks preserve roof lines, lettering, fasteners, hoses, and
wheel structure in the inspected native render. Broad and narrow non-scene
bands in the dark undercarriage are substantially reduced.

## 7. Automated regression coverage

Sixteen tests now pass. In addition to the clipping, gradient, block-boundary,
structure-mask, and motion-fit cases from the previous iteration, the suite
now covers:

- localized multi-channel vertical detail without a supplied structure mask;
- repeated local grill detail;
- repeated and isolated single-channel color detail;
- a narrow scene feature shifted by one line between the two sensor columns;
- reduction of high-rank pair-specific sensor noise by the fine stage; and
- exact equality of both model stages across internal processing blocks.

The test suite, Python compilation, and `git diff --check` pass. Full-capture
validation results are recorded below after fresh end-to-end renders.

## 8. Fresh end-to-end validation

The full validation runs use scratch directories containing symlinks to the
original raw files. No `.bin` file or capture setting is modified.

All four requested captures completed a fresh selection-through-render
pipeline:

| Capture | Stacked preview | Native audit |
| --- | --- | --- |
| MTR | `13793 × 12288` | three full-width slabs plus four exact regression crops |
| Pato | `16162 × 10240` | four full-width slabs plus the exact reported crop |
| Maglev | `15765 × 8192` | four full-width slabs |
| Panorama | `8722 × 2048` | four overlapping tiles |

The MTR exact wheel crop retained the recognizable wheels and suspension
structure while reducing the combined rendered stripe RMS by about 79%
relative to no correction. The new B/G/R RMS was
`0.547 / 0.364 / 0.172 DN`, versus `2.634 / 1.230 / 1.003 DN` in the
uncorrected render. The exact cab crop retained the window, pilot silhouette,
and reflections without purple patches. The exact articulated-connection crop
retained the complete bellows pleat structure. The previously regressed
clipped-roof crop now has a smooth surrounding background without either the
wild vertical bands of the old correction or loss of roof detail. Native
slabs spanning the train showed consistent door, window, bogie, coupler, and
roof-grill geometry from the first car to the last.

The Pato crop matching the reported image retains the Bombardier and Talgo
lettering, step, bolts, hoses, and wheel assemblies while the recurrent
vertical bands in the dark chassis are nearly absent. Four native slabs
covering both noses, the power car, passenger cars, and couplers retained
small serial text, grilles, wiring, and undercarriage structure. Window and car
spacing remained uniform across the train, with no correction-driven
left-to-right stretch.

The Maglev native slabs retained repeated windows, doors, logos, ventilation
grilles, front and rear geometry, and small undercarriage details against its
bright background. The panorama tiles retained smooth sky and dark foreground
as well as buildings, scaffolding, window grids, poles, foliage, and vehicles.
Neither capture showed coherent correction stripes, smearing, or color
splotches; fine repetitive detail remained sharp.

The MTR plus panorama concurrent run took 676.2 seconds and peaked at about
12.5 GiB RSS. The Pato plus Maglev concurrent run took 645.6 seconds and peaked
at about 20.7 GiB RSS. These wall times include raw selection, masks, raw
speed/skew estimation, model construction, and all DNG, PNG, JPEG, and preview
writes. Focused production model construction took 89 seconds for MTR and
80.6 seconds for Pato. The complete render therefore remains above the
aspirational five-minute target and should be optimized separately without
weakening the detail-preservation safeguards.

The first concurrent validation attempt also exposed Matplotlib's interactive
Tk backend being destroyed from a worker thread. Preview now selects the
non-interactive `Agg` backend explicitly; both subsequent concurrent runs
completed cleanly.
