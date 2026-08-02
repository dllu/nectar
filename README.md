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

## Speed-constraint annotations

The raw speed-constraint editor displays consecutive Bayer-binned capture
chunks and autosaves circle, equal-width, and sequence-cut annotations beside
the raw data:

```sh
uv run python python/speed_constraints.py /path/to/capture
```

It opens 16 chunks at a time by default. A bounded-memory navigator is built
at the bottom by sampling each raw chunk one at a time; click it to center the
detailed sliding window. The navigator wraps into multiple rows as needed, and
its divider can be dragged vertically. Use Left/Right to slide by one chunk,
Ctrl+mouse-wheel to zoom, F to enable persistent fitting, F8 to hide the
navigator, and F9 to hide the annotation panel. Fit mode is enabled initially
and refits whenever the window is resized; manual zoom or 100% mode disables
it. The **Rotate 180°** checkbox rotates both detailed and
navigator views without changing saved coordinates; it starts checked for
pre-2026 captures and unchecked for 2026-and-later captures so both camera
mounting eras appear upright. The default output is `speed_constraints.json`
in the capture directory; an existing file is loaded when the tool starts.

`preview.py` loads that file automatically. Circle and equal-width annotations
refine the correlation-derived speed spline without changing its direction,
and sequence positions replace motion-region detection and define one output
DNG/JPEG per adjacent pair. The first and last positions also bound speed
fitting: only complete correlation windows inside those bounds are eligible.
Featureless background windows are rejected as undefined unless they contain
both real horizontal structure and an absolute, distinct correlation peak, so
padded background cannot pull the manual fit. Pass `--no-speed-constraints`
to render with the legacy automatic behavior for comparison. Pass
`--equal-length-chunks` to use only the first
and last annotated positions and split that corrected output range into nearly
equal chunks smaller than `OUTPUT_CHUNK_TARGET`, ignoring intermediate cut
positions.

Output orientation follows the camera mounting date: captures before 2026 are
not rotated, while captures from 2026 onward are rotated 180 degrees. Override
the automatic choice with `--flip-output-180` or `--no-flip-output-180`.
