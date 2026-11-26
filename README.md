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

Each capture is rendered into a temporary staging directory. Only after the
complete DNG/JPEG set and stack preview succeed does `preview.py` replace that
capture's prior outputs, including chunks that no longer exist. Existing
Darktable `.dng.xmp` sidecars are carried forward by `NECTA` chunk number and
renamed for the new DNG hash; if several sidecars exist for one chunk, the most
recently modified one is used. Choose a separate `--output-dir` when an older
render needs to be preserved.

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
