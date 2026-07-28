# Scene-aware shadow-detail protection — 2026-07-27

> **Geometry correction, 2026-07-28.** Speed and skew no longer use untouched
> Bayer planes. The motion-independent radiometric model is applied before
> both fits; see `2026-07-28-radiometric-first-geometry.md`.

## 1. Remaining failure

The recurrent plus motion-aligned fine jitter model removed most visible
stripes, but its combined additive correction was still unrestricted away
from black and white clipping. In the supplied MTR articulated connection,
the correction could approach 680 raw DN. That is small relative to the
sensor's complete range, but it is large relative to a bellows fold carrying
only a few thousand black-subtracted DN. The folds therefore converged toward
the same dark color even though neither model stage explicitly blurred the
rendered image.

This explains why clipped samples and bright details could pass while the
bellows, wheels, and other deep shadows looked washed out. A correction bound
expressed only as a fraction of full channel range does not bound its effect
on local shadow contrast.

## 2. Rejected application strategies

Three application changes were replayed against the exact production MTR
selection:

- A global cap at 5% of each sample's black-subtracted signal restored most
  dark detail, but also weakened correction in genuinely flat dark regions.
  Faint original stripes returned above the clipped roof.
- A hard binary protection mask kept detail near detected edges, but its
  abrupt row-band and column boundaries became visible as block-like grids.
- Protecting only the exact edge pixels did not cover low-contrast interiors
  such as the space between the two boundaries of a bellows pleat.

The useful distinction is not simply bright versus dark. Full correction is
desirable in a flat shadow, while a much smaller correction is desirable
around corroborated scene structure in that same shadow.

## 3. Retained scene-aware limit

The existing motion fit supplies a displacement between the two green sensor
columns. A real scene edge appears in both green planes after applying that
displacement; a sensor-column exposure stripe remains tied to the capture
column and does not align as scene structure.

The estimator now reuses that evidence as follows:

1. It selects only strong, same-direction green edges at least three robust
   gradient scales above the local band noise.
2. Each strong seed protects 256 capture columns on either side. The wider
   neighborhood includes low-contrast interiors bounded by a pair of strong
   edges, rather than protecting only the boundary.
3. The 256-column expansion remains specific to each sensor-row band. A roof
   edge therefore cannot weaken stripe correction in unrelated flat background
   many rows above it.
4. A Gaussian feather of 3 columns and 0.75 row bands removes hard transitions
   at the protection boundary without a broad vertical dilation.
5. Where protection is one, the *combined* recurrent and fine additive
   correction is limited to 5% of the sample's black-subtracted signal. Where
   protection is zero, the complete learned correction is retained. Fractional
   confidence blends continuously between those two cases.
6. Existing black and highlight fades are applied afterward. Exact black
   remains a fixed point.

If no motion-aligned scene evidence is available, the model falls back to the
conservative 5% signal-relative limit everywhere. This favors recoverable
scene detail over aggressive shadow correction when the two cannot be
distinguished.

The protection confidence is stored as `float16` alongside the two correction
components and interpolated over row bands during final sampling. This
scene-relative application limit is used for rendering only. Geometry sampling
bypasses it and applies the complete motion-independent radiometric correction.

## 4. Focused production replays

The exact MTR production model, raw motion fit, tone curve, and sampling
positions were replayed without changing geometry. On strong baseline edges
in the articulated connection, retained edge energy was:

| Application | Edge-energy ratio |
| --- | ---: |
| Unrestricted two-stage correction | 0.701 |
| Global 5% shadow cap | 0.919 |
| Scene-aware protection | 0.912 |

The scene-aware result recovers almost all of the global cap's bellows detail,
with baseline correlation 0.957. In the cab/window crop, the unrestricted,
global-cap, and scene-aware edge ratios were 0.832, 0.922, and 0.901
respectively; the scene-aware correlation was 0.971. Purple block artifacts
did not return. The wheel crop also improved over unrestricted application
while retaining recognizable wheel and suspension geometry.

The clipped-roof audit is the important counterexample. A global cap restored
faint vertical stripes in the flat background. Because scene-aware protection
does not vertically dilate the wide protected neighborhood, that background
stays close to the fully corrected result while the roof edge and neighboring
bellows remain protected. Increasing the horizontal neighborhood from 64 to
256 columns raised bellows edge retention from 0.807 to 0.912; increasing it
further to 512 yielded only 0.916 and was not retained.

The exact Pato production selection was independently replayed. In a dark
chassis ROI, horizontal strong-edge retention relative to the uncorrected
render was 0.862 for unrestricted application, 0.972 for a global 5% cap, and
0.924 for scene-aware protection. Flat columns continue to receive the full
stripe correction, so the result avoids the blanket cap's broader stripe
regression.

## 5. Regression coverage

The jitter test suite now checks:

- limiting positive and negative combined corrections relative to local
  black-subtracted signal;
- exact preservation of samples at or below black;
- full correction in a flat dark region with zero scene confidence;
- construction of motion-aligned protection over scene detail;
- smooth feathering across columns and neighboring row bands; and
- exact equality of correction and protection components across internal
  4096-column processing blocks.

The focused MTR bellows, cab, wheel, and clipped-roof replays and the independent
Pato shadow replay use production raw data. Source `.bin` files, capture
settings, geometry fits, and C++ capture code are unchanged.
