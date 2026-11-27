Nectar
====

![Shinkansen N700A](https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Line_scan_photo_of_Shinkansen_N700A_Series_Set_G13_in_2017%2C_car_01.png/1280px-Line_scan_photo_of_Shinkansen_N700A_Series_Set_G13_in_2017%2C_car_01.png)

![Caltrain](https://i.dllu.net/caltrain317_4c6c80561b5156c3.jpg)

![Screenshot](https://i.dllu.net/2024-09-01-10-03-37_c1f08f5150abd503.png)

Some simple programs for capturing images from Alkeria Necta line scan camera.

## Prerequisites

1. Alkeria Necta USB3 camera with proprietary binary drivers version 1.11. If you are using Arch Linux, you can install them using the PKGBUILD in the `arch-linux` directory. If on Ubuntu, just install using the deb file that you download from the Alkeria Download Area.
2. [imgui](https://github.com/ocornut/imgui) 1.91.9 and above
3. SDL2

## Previewing slow subjects

Run `python/preview.py` as usual. The preview probes the displacement between
the two green planes at powers-of-two column strides and uses the selected
scale for motion detection, speed correlation, and skew estimation. If there
are too few supported measurements, it doubles the stride and tries again.
The fitted step and reconstruction aperture are expressed in original Bayer
columns. A slow subject therefore contributes multiple original exposures to
each full-resolution output pixel.

The first pass reads the raw files sequentially and keeps them in RAM for the
later passes. The default cache budget is half the available RAM on Linux,
shared across simultaneous captures. If the raw files exceed the budget, the
preview uses lossless LZ4 compression and reads any frames that still do not
fit from disk. Use `--raw-cache-gib 12` to set a 12 GiB budget or
`--raw-cache-gib 0` to disable the capture-wide cache. Jitter correction and
rendering use additional working memory.

## Speed-constraint annotations

The raw speed-constraint editor displays the Bayer-binned capture and autosaves
circle, equal-width, and sequence-cut annotations beside the raw data:

```sh
uv run python python/speed_constraints.py /path/to/capture
```

Use the column selector in the toolbar to show all columns or one in every
2, 4, 8, and so on. You can also start with `--column-stride 8`. The default
detail window spans 4096 displayed columns, so at 1/8 it covers 32,768
original Bayer columns. Click the navigator to center the window. The editor
reads each raw frame once at startup and keeps the complete Bayer-binned RGB
image in RAM (three bytes per pixel). Seeking and changing the display stride
use that image directly.
The navigator wraps into multiple rows as needed, and its divider can be
dragged vertically. Use Left/Right to slide by 256 displayed columns,
Ctrl+mouse-wheel to zoom, F to enable persistent fitting, F8 to hide the
navigator, and F9 to hide
the annotation panel. Fit mode is enabled initially and refits whenever the
window is resized; manual zoom or 100% mode disables it. The **Rotate 180°**
checkbox rotates both detailed and
navigator views without changing saved coordinates; it starts checked for
pre-2026 captures and unchecked for 2026-and-later captures so both camera
mounting eras appear upright. The default output is `speed_constraints.json`
in the capture directory; an existing file is loaded when the tool starts.

`preview.py` loads that file automatically. Circle and equal-width annotations
refine the correlation-derived speed spline without changing its direction,
using a fixed, uniformly spaced spline basis. Each equal-width group has a
jointly fitted common output width. The dimension uncertainty is 2.5 displayed
pixels. For example, at 1/8 it is 20 Bayer columns horizontally and 2.5 Bayer
rows vertically. The editor records the stride for each annotation and saves
its coordinates at the original Bayer scale. Older annotations use stride 1.
Sequence positions replace motion-region detection and define one output
DNG/JPEG per adjacent pair. The first and last positions also bound speed
fitting: only complete correlation windows inside those bounds are eligible.
Featureless background windows are rejected as undefined unless they contain
both real horizontal structure and an absolute, distinct correlation peak, so
padded background cannot pull the manual fit. Pass `--no-speed-constraints`
to render with the legacy automatic behavior for comparison. Pass
`--equal-length-chunks` to use only the first
and last annotated positions and split that corrected output range into nearly
equal chunks smaller than `OUTPUT_CHUNK_TARGET`, ignoring intermediate cut
positions. Pass `--pad-output-to-equal-widths` instead to retain every annotated
cut and pad the resulting slices to the width of the widest one.

Pass `--justify` to give all full-resolution DNG and JPEG slices a common
width. The stacked JPEG uses the same slices at half resolution. The fit keeps
the annotated start and end fixed and balances interior cut movement against
horizontal scaling. Each cut has a prior of 20 displayed columns at its own
annotation stride, and the scale prior is 4%. Each interior slice gets a
constant offset to its log-speed spline during raw sampling. The first and
last slices keep their native scale and fill any missing width with randomly
sampled columns from their outside edges.
For a print layout, choose cuts that give the middle slices similar native
lengths and leave both end slices shorter. This keeps horizontal scaling
small and lets the captured background fill the ends.

With `--justify --justify-padding original`, the end slices use contiguous
background from outside the annotated cuts. The renderer extends the speed
spline at a constant speed from each end cut. If the capture ends early, the
renderer fills the remaining width by randomly sampling whole columns from
the outermost 128 available columns, including the slice edge when fewer than
128 background columns are available. Use the default
`--justify-padding random` when the captured margins contain other moving
objects you want to omit.

Output orientation follows the camera mounting date: captures before 2026 are
not rotated, while captures from 2026 onward are rotated 180 degrees. Override
the automatic choice with `--flip-output-180` or `--no-flip-output-180`.
The half-resolution `_stack.jpg` places the earliest part of the capture on
top. Full-resolution JPEGs and DNGs are numbered from left to right so they
can be concatenated horizontally in filename order.

Each capture is rendered into a temporary staging directory. Only after the
complete DNG/JPEG set and stack preview succeed does `preview.py` replace that
capture's prior outputs, including chunks that no longer exist. Existing
Darktable `.dng.xmp` sidecars are carried forward by `NECTA` chunk number and
renamed for the new DNG hash; if several sidecars exist for one chunk, the most
recently modified one is used. Choose a separate `--output-dir` when an older
render needs to be preserved.

## Color and linear DNGs

The DNG stores three reconstructed 16-bit camera channels. White balance is
editable through `AsShotNeutral`. The D65 color matrix includes the sensor's
unequal daylight response, while the forward matrix takes white-balanced
camera RGB to D50 XYZ. The JPEG uses the DNG's 16-bit samples and camera
profile, followed by display exposure, sharpening and sRGB encoding. The
individual JPEGs and stack retain full-resolution chroma (4:4:4, quality 99).
To derive the nominal profile,
we combined the published sensor quantum efficiency and solid RGB filter
transmission curves at 5 nm intervals from 400 to 700 nm. Under D65 this
predicts a native white of `(1.16927, 1, 0.72297)`. We fitted the matrix to
the average reflectances of 24 ColorChecker patches, using the CIE 1931 2°
observer and constraining balanced white to stay neutral. Lens transmission
and the individual camera can differ from this model. [Alkeria NECTA User Guide, figures 3.2 and 3.3](https://www.argocorp.com/linescancamera/Alkeria/common/download/NECTA/document/DOC-NECTA-User_Guide-v2.4.pdf)
and [BabelColor spectral data](https://babelcolor.com/colorchecker-2.htm).

Green reconstruction aligns the two phases in capture time and sensor height.
A signal-weighted mean keeps the surviving measurement when one phase clips
to zero. A five-column median estimates the local difference in phase response
separately at each sensor height. Native-row detail that varies within that
window contributes additional contrast, weighted by the camera pitch at the
exposures that supplied it.

If spot-balancing a white surface leaves a darker neutral surface tinted, place
`color_reference.json` beside that capture's raw frames. Measure two unclipped
neutral areas under the same illumination in **reconstructed camera counts**
from an initial DNG. The following values are from dim and bright lower steam
in the November 22 Big Boy capture.

```json
{
  "dark_rgb": [3634, 2956, 1359],
  "light_rgb": [31375, 28010, 20371],
  "shadow_chroma_threshold": 3000
}
```

The difference between the patches sets white balance. Their relative offsets
correct level-dependent color in both exports. Where a channel has clipped to
black, the correction is bounded by the signal in the other channels. Color
in that interval is an estimate. The optional `shadow_chroma_threshold` tapers
saturation where even the strongest white-balanced channel falls below the
threshold. For this capture, 3,000 counts is about twice the largest fitted
channel offset in green-equivalent units. Omitting the field preserves full
shadow saturation. Captures without a reference use the automatic white balance
and their original black point.

When the highlights already have a good white balance, a shadow reference can
retain it. Use the initial DNG's `AsShotNeutral`, a measured dark neutral, and
`shadow_fade_end` in green-equivalent 16-bit camera counts. The offset has full
strength at the dark patch and fades smoothly to zero at the endpoint. A pixel
above that level in any balanced channel retains all three original counts.
The preview also retains the original exposure metering. The optional chroma
taper is useful when a channel clips to zero in a dark neutral.

The [measured capture references](python/color_references/) record the source
DNG, patch coordinates, and gain and shutter settings. The preview uses a
bundled reference when the capture directory name and recorded settings match.
A local `color_reference.json` takes precedence. The November 21 reference
uses the silver trucks of a passenger car, with a chroma taper for the
black-clipped blue channel. The June 16 reference uses a shaded Nankai bogie.
Its blue channel retains detail, so it uses only the relative offsets.

To reproduce the four-row layout from the Nectar repository, run

```sh
capture=/mnt/dataz/pictures/linescan/2026-11-22-17-17-19
uv run python python/preview.py "$capture" \
  --speed-constraints-file "$capture/four-row-print/speed_constraints.json" \
  --justify --justify-padding original --output-dir "$capture/four-row-print-color"
```

## Camera vibration compensation

`preview.py` corrects pitch vibration by default. It tracks stationary ground
in the bottom 128 sensor rows of the upright image using normalized vertical
cross-correlation on one raw green plane. The fit removes a slow baseline
(Gaussian sigma of 256 capture-time columns) and tapers the correction to zero
where the ground lacks a distinct match. Each Bayer plane is sampled at its
fractional source height before the horizontal dewarp combines scan columns.

Use `--vibration-rows N` to adjust the ground strip height in full-resolution
pixels (minimum 32). Choose a strip containing stationary detail. For a capture
where the subject fills the bottom edge, or for a comparison render, pass
`--no-vibration-compensation`. The separate `--jitter` option controls
exposure stripes.
