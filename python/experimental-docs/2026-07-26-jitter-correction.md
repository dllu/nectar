# Vertical-stripe correction experiments — 2026-07-26

## 1. Baseline inspection

The previous correction fitted an affine transform between every adjacent pair
of line columns and composed those transforms from left to right. Inspection of
the four captures confirmed the failure described in `jitter_prompt.md`:

- Smooth scene gradients produced small relative estimates that accumulated
  into broad brightness drift.
- One shared RGB model could not represent the distinct red, green-1, green-2,
  and blue responses.
- Exact zero and saturated regions strongly biased the fit and could spread
  colored vertical streaks into neighboring dark rows.
- Its fixed residual sigma behaved differently in the bright Pato capture and
  the underexposed Maglev capture.

The MTR capture was the clearest stress case. Raw crops around pair columns
67,200 and 122,821 showed fine common-mode stripes over the train body and
stronger bands associated with the clipped roof.

## 2. Non-cumulative estimator experiments

A centered temporal median was tested as an absolute local reference. For each
channel, every line column was summarized in vertical bands, and the difference
between the current band median and a centered temporal median was used as the
candidate additive correction.

This reference has an important property: a monotonic sequence is its own
centered median. Consequently, smooth rising or falling scene brightness
produces zero correction rather than a sequence of steps that can accumulate.

Experiments compared:

- 32 versus 64 vertical bands;
- temporal windows of 9 and 31 columns;
- direct band corrections versus smooth cosine-basis projections;
- 8, 12, 20, and 32 cosine coefficients; and
- additive corrections versus intensity-affine fits based on three quantiles.

The 31-column window removed more line-frequency variation without damaging
the inspected gradients. Sixty-four bands captured the row dependence. Twelve
cosine coefficients retained that response while avoiding hard band
boundaries. Higher basis sizes began to follow scene detail in individual
bands. The quantile-affine model slightly worsened the robust residual metrics
and was less stable near clipping, so it was rejected.

## 3. Clipping and scene-structure safeguards

An early cosine-basis version improved the aggregate metric but spread color
from the clipped MTR roof into darker rows. The final estimator therefore:

1. estimates separate black and white endpoints for all four Bayer channels;
2. rejects a vertical band when more than 10% of its samples are near either
   endpoint;
3. requires a common correction sign across many bands and at least two
   channels;
4. fills an invalid band's estimate from the robust cross-band/channel median
   before vertical smoothing; and
5. fades application to zero at the black and white endpoints, leaving exact
   clipped samples unchanged.

Consensus replaces the old fixed exposure-dependent sigma. Local scene edges
normally affect only a minority of row/channel features, while sensor exposure
jitter is common across them. No top, bottom, or static-background region is
assumed.

The model stores 64 smooth band-center corrections per channel as `float16` and
interpolates them by sensor row during sampling. This keeps the largest tested
model at about 118 MiB and makes skewed per-pixel lookup inexpensive. Raw-file
caching is byte-bounded at 512 MiB.

## 4. Full-width validation

All columns of each requested capture were modeled. Times below cover the
stripe-model estimation pass.

| Capture | Pair columns | Time | Model size |
| --- | ---: | ---: | ---: |
| Pato `18-10-06-13-52-16-utc` | 173,056 | 37.5 s | 84.5 MiB |
| Maglev `18-10-18-02-01-33-utc` | 148,480 | 32.7 s | 72.5 MiB |
| MTR `2026-01-13-09-49-37` | 242,432 | 63.8 s | 118.4 MiB |
| Panorama `2026-01-13-09-32-44` | 39,424 | 9.1 s | 19.3 MiB |

For quantitative comparisons, each channel was split into the same 64 vertical
bands. The metric is the median absolute common residual after subtracting a
31-column centered median. The 95th percentile is included to expose strong
outlier stripes.

| Inspected crop | Common MAD, before → after | Common p95, before → after |
| --- | ---: | ---: |
| Pato, pair column 86,528 | 24.0 → 7.41 | 76.0 → 22.69 |
| Maglev, pair column 73,728 | 20.0 → 4.37 | 96.0 → 22.92 |
| MTR, pair column 122,821 | 48.0 → 8.70 | 152.0 → 40.56 |
| Panorama, pair column 8,000 | 8.0 → 5.72 | 72.0 → 32.55 |
| Panorama, pair column 20,000 | 8.0 → 2.19 | 64.0 → 20.41 |

Before/after RGB crops were inspected both with normal tone mapping and an
aggressive dark-region lift, including 1:1 crops:

- The MTR roof and body lost the dense non-scene bands. Saturated roof pixels
  stayed saturated, and their correction did not bleed into the undercarriage.
- Pato's smooth body gradients and bright regions remained smooth; dark-window
  line patterning was reduced without broad correction drift.
- Maglev retained its exposure noise and scene edges but lost most of the
  coherent column residual.
- The panorama retained building, foliage, pole, and vehicle edges despite
  having no static background region.

The correction is orientation-independent. Final rendering was also exposed as
`--flip-output-180` / `--no-flip-output-180`, so the older non-flipped captures
and newer flipped captures no longer require editing a source constant.

## 5. Automated checks

`python/test_jitter_correction.py` covers:

- exact zero correction for a smooth monotonic gradient;
- reduction of synthetic channel- and row-dependent stripes;
- identical estimates across internal processing-block boundaries;
- bit-for-bit preservation of exact black and saturated samples; and
- agreement between contiguous-window and skewed per-pixel model lookup.

No `.bin` file or capture settings file was modified during these experiments.
