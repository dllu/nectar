# Radiometric-first geometry fitting — 2026-07-28

## 1. Regression

The detail-preservation change in `ee38a66` made speed and skew estimation use
uncorrected Bayer data. Its rationale was that allowing a radiometric model to
affect geometry could stretch the rendered train.

That separation was backwards. The column correction is a property of the raw
capture coordinates. A strong exposure stripe occurring in both green planes
at the same capture column creates a false zero-displacement match. Estimating
motion from those uncorrected planes therefore lets the radiometric error
determine geometry.

## 2. Restored dependency order

Production processing now has this acyclic order:

1. Estimate the recurrent radiometric correction in raw capture coordinates.
   This estimator has no speed or skew input.
2. Apply the complete recurrent correction to both green planes, then estimate
   and fit speed.
3. Resample all Bayer planes at the speed-corrected positions while applying
   the same complete radiometric correction, then fit skew.
4. Once geometry is final, use the fitted green displacement to construct the
   optional fine residual and scene-protection data used only for final
   rendering.

The 5% scene-relative shadow limit protects output appearance; it is not part
of the sensor correction. Speed and skew explicitly bypass that limit and use
the full learned correction. Black and highlight confidence fades remain in
place.

The motion-aware fine stage is deliberately downstream of both geometry fits.
It is a conservative rendering safeguard around wheels, bellows, windows, and
other real scene structure and is not fed back into speed or skew. There is
therefore no radiometry/geometry cycle.

## 3. Capture replay

The exact contiguous moving selection for
`/mnt/dataz/pictures/linescan/2026-01-13-09-54-37` was files 135–573: 439 raw
files and 112,384 Bayer-pair columns. The recurrent estimator retained 47
sensor modes.

Using identical correlation windows, the geometry changed materially:

| Measurement | Uncorrected Bayer | Radiometric first |
| --- | ---: | ---: |
| Sample displacement median | 0.392 | 0.578 |
| Sample displacement range | 0.148–0.693 | 0.487–0.794 |
| Fitted displacement median | 0.394 | 0.576 |
| Fitted displacement range | 0.365–0.503 | 0.528–0.593 |
| Fitted max/min ratio | 1.378× | 1.123× |

The median fitted step moved by 0.184 columns, about 47% of the uncorrected
estimate. This explains the visible horizontal stretch: raw stripe coherence
was dominating part of the green-plane correlation. The corrected fit is also
substantially more uniform from one end of the train to the other.

As a sensitivity check, adding the later fine rendering residual changed the
recurrent-corrected fitted speed by only −0.004 columns at the median, 0.010 at
the 95th percentile, and 0.018 in the worst window. This confirms that the
large geometry repair comes from putting the motion-independent radiometric
model before speed, rather than from the rendering safeguard.

On this capture, the production-order skew pass—corrected-speed positions plus
the uncapped recurrent model—selected −0.01618, the same grid point as the
uncorrected-radiometry control. The ordering is still retained because the
size and direction of a stripe-induced skew bias are scene-dependent.

## 4. Regression coverage

The test suite now includes a synthetic pair of green planes with a true
two-column displacement and a much stronger common capture-column stripe. Raw
correlation collapses to zero displacement; radiometric-first correlation
recovers 1.990 columns.

A separate sampling test verifies that output rendering keeps the
scene-relative shadow limit while geometry sampling receives the complete
raw-coordinate correction. A replay of the accepted MTR model also matched all
163,053,568 fine-correction values bit-for-bit after the estimator was split,
so the geometry repair does not alter its wheel, window, bellows, or shadow
detail behavior. The full Python suite contains 21 passing tests.
