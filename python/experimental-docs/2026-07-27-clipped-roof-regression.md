# Clipped-roof and long-cadence stripe regressions — 2026-07-27

## 1. Reproducing the clipped-roof failure

The correction committed on 2026-07-26 was not safe around the clipped MTR
roof. The supplied screenshot was located in the production render at source
coordinates approximately `x=15202`, `y=234`, with a source crop of
`984 x 1194` pixels. Matching the supplied half-resolution screenshot against
that render gave a normalized correlation of 0.998.

The production path was replayed from its actual selected range, files 168–789
(622 files and 159,232 Bayer-pair columns). A focused replay of the regressed
algorithm matched the screenshot at 0.975 correlation despite the JPEG round
trip. Disabling the correction removed the dense field of colored lines above
and below the roof.

The previous implementation coupled unrelated sensor rows in three ways:

1. It replaced a clipped or otherwise invalid vertical band with a value pooled
   from other rows and Bayer channels.
2. One global sign-confidence value decided whether a column was corrected
   anywhere on the sensor.
3. A whole-height cosine projection spread the pooled values back across the
   sensor.

The endpoint fade preserved samples exactly at black and saturation, but it
could not protect the surrounding non-clipped pixels from that cross-row
projection. The 2026-07-26 validation had not replayed the reported production
coordinate, so its clipping-safety conclusion was wrong.

## 2. Why the first local fix was insufficient

Removing the invalid-band fill and whole-height projection made the roof safe.
It also exposed a second problem in Pato: a repeating stripe with a cadence of
about 142 pixels in the half-resolution preview, or about 283 Bayer-pair
columns. A 31-column temporal window cannot estimate that broad cycle.

The following alternatives were evaluated and rejected:

- Larger windows with only 64 or 128 vertical bands left row-dependent
  residuals in Pato.
- A 1023-column correction by itself cleaned Pato but again treated the clipped
  roof plateau as an exposure error.
- Dilating the scene mask protected more scene content but also hid too much of
  the stripe from the estimator. An undilated mask was the best boundary.
- Requiring a shared sign across Bayer channels was unsafe. Measured
  cross-channel correction correlations were only about 0.51–0.73, and valid
  responses sometimes had different signs.
- Filling unsupported bands, even from immediate neighbors, could carry a
  correction back across the clipped-roof boundary.

`bottleneck.move_median` replaced the much slower generic median filter so that
the larger model could still run over complete captures.

## 3. Final stripe estimator

The retained model is absolute rather than cumulative and uses 256 independent
vertical bands. For each Bayer channel and band:

1. A 31-column centered median estimates fine line jitter.
2. A 1023-column centered median estimates the broad exposure cycle. Its
   contribution is separated from the fine component.
3. Fine and broad candidates are limited to 2% and 1% of that channel's
   measured range, respectively. Larger residuals are treated as scene changes.
4. A band is unsupported when more than 10% of its samples are near black or
   saturation, or when the undilated scene-structure mask intersects it.
5. Unsupported candidates are exactly zero. There is no value fill, sign vote,
   cosine projection, or cross-height smoothing.
6. During application, interpolation may taper a supported band toward its
   neighbor, but an owner-band gate prevents a neighbor's correction from
   entering an unsupported band.
7. Correction strength fades to zero at each channel's estimated black and
   white endpoints, so clipped samples remain bit-for-bit unchanged.

Block processing includes a 511-column halo, so internal 4096-column block
boundaries are equivalent to estimating the same region in one block. The
model is stored as `float16`; arithmetic during estimation and application is
`float32`.

## 4. Exact MTR roof result

Baseline, regressed, and final multiscale focused replays used identical motion
samples, deskew, white balance, color transform, sharpening, exposure, and crop
coordinates. For the top 500 rows above the roof, the metric takes each
column's median red–green and blue–green values, subtracts a 31-column centered
median from both traces, and computes their RMS magnitude. It measured:

| Model | Median | p95 | p99 |
| --- | ---: | ---: | ---: |
| No stripe correction | 1.275 | 2.915 | 3.673 |
| Regressed global/DCT model | 1.458 | 5.148 | 8.062 |
| Final multiscale model | 0.707 | 1.581 | 3.162 |

The final model beats the no-correction baseline at all three percentiles while
the regressed model is substantially worse. The aligned 1:1 images also show
that real horizontal scene edges and the clipped roof shape are unchanged.

The final multiscale model was then run through the complete 622-file pipeline.
The supplied screenshot aligned to `preview_stack.jpg` at
`x=7234:7718`, `y=4213:4811`. In that exact dark band, the regressed-to-final
column-residual changes in rendered DN were:

| Component | Standard deviation | p95 |
| --- | ---: | ---: |
| Luma | 6.270 → 0.174 | 12.234 → 0.337 |
| Red–green | 2.804 → 0.292 | 5.895 → 0.547 |
| Blue–green | 3.413 → 0.286 | 7.448 → 0.569 |

Thirty-five full-width native dark-strip tiles and 80 native tiles across the
complete stacked preview were inspected. No non-scene stripe was found. Median
luma/red–green/blue–green p95 values in the dark-strip tiles were
0.352/0.589/0.574 DN; the maxima coincided with real scene objects. The clipped
roof and surrounding shadows remained clean under an aggressively expanded
display.

## 5. Pato result

The final Pato output was generated from selected files 2–165. The user's
reported view aligned at `x=8275:9025`, `y=9721:10235` in the
`16112 x 10240` stacked preview.

In the upper background of that exact view, the old-to-final residual changed
as follows:

| Metric | Old | Final | Reduction |
| --- | ---: | ---: | ---: |
| Luma standard deviation | 0.307 | 0.047 | 84.7% |
| Luma MAD | 0.200 | 0.033 | 83.5% |
| Luma p95 | 0.580 | 0.091 | 84.3% |
| Red–green standard deviation | 0.182 | 0.116 | 36.3% |
| Blue–green standard deviation | 0.411 | 0.128 | 68.9% |

The strong 142-preview-pixel stripe signature had correlation 0.776 before the
change and was absent afterward. A weaker 208-pixel peak at correlation 0.322
followed real scene structure rather than crossing surfaces. Every native tile
and the exact user region was inspected; no coherent non-scene vertical stripe
remained.

## 6. Motion-fit guard found during Maglev validation

Stripe removal itself passed on Maglev, but the first full render exposed an
independent motion regression: repeated windows narrowed from roughly 384 to
300 pixels and the rear cab had horizontal/vertical scale near 0.75. The fitted
speed magnitude drifted by 2.81× across a train whose repeated rigid features
showed that such a change was impossible. Robust residual weighting could not
reject the error because the low-light correlation bias varied smoothly with
the scene.

Two safeguards were added:

- If a fit crosses zero, it is retried using the dominant direction's
  high-confidence samples and is clamped outside their supported interval.
- If the fitted speed-magnitude range exceeds 1.5×, it falls back to the
  high-confidence weighted median. Maglev selected `-0.283190`.

The guard is deliberately above the observed ranges for Pato (1.16×), MTR
(1.14×), and the panorama (1.33×), so their accepted motion fits are unchanged.

The guarded Maglev rerun produced a `15818 x 8192` stacked preview. Eleven
repeated windows in one end measured 378–384 pixels (median 382); twelve in the
other measured 384–391 pixels (median 385). Their median ratio is 1.008.
Mirrored-cab affine fits gave horizontal/vertical scale ratios of 1.001–1.007,
instead of about 0.75. The progressive 22–25% squeeze is gone.

Sixty-four native Maglev tiles were also audited for stripes. Across 32 sky
tiles, the median/maximum luma column-residual p95 was 0.159/0.167 DN. Fine
texture in the very dark guideway was localized, uncorrelated with the sky and
between output strips, and present in the no-correction baseline; it was not a
coherent column artifact.

## 7. Complete-capture validation

All four requested captures were processed from raw selection through the
saved preview artifacts:

| Capture | Selection | Stacked preview | Stripe pass | Whole pipeline | Result |
| --- | --- | ---: | ---: | ---: | --- |
| Pato | files 2–165 | 16112×10240 | ~43 s | 490.24 s | Stripe pass; severe end stretch absent |
| MTR | files 168–789 | 14379×10240 | ~48 s | 537.73 s | Exact roof and full-width stripe pass |
| Maglev | files 52–86 | 15818×8192 | ~9 s | 254.24 s | Stripe and geometry pass |
| Panorama | files 0–153 | 9245×2048 | ~9 s | 74.91 s | Stripe and geometry pass |

The stripe estimator itself remains below one minute on every capture. The two
longest complete pipelines exceed five minutes because the timing also
includes the existing motion search, skew search, demosaic, render, and
PNG/DNG/JPEG writes. Quality was not reduced to hide that wall-clock result.

The panorama was split into four native-width tiles for inspection. Sky, dark
foreground, buildings, scaffolding, poles, foliage, and vehicles retained
their real edges; no vertical correction field crossed those surfaces.

## 8. Automated regression checks

`python/test_jitter_correction.py` contains ten passing tests covering:

- exact fixed-point behavior on a monotonic scene gradient;
- reduction of channel- and row-dependent fine stripes;
- reduction of a broad synthetic stripe cadence;
- equality across internal processing-block boundaries;
- exact preservation of clipped samples;
- exact isolation of a clipped roof-like transition from unaffected rows;
- preservation of a structure-masked scene feature;
- rejection of neighbor interpolation inside an unsupported owner band;
- rejection and endpoint clamping of direction-reversal aliases;
- constant-speed fallback for catastrophic dynamic range while preserving a
  supported moderate speed change.

`python -m py_compile` and `git diff --check` also pass. No source `.bin` file,
capture setting, or C++ capture path was modified.
