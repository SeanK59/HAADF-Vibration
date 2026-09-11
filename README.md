# HAADF Vibration Simulator

Interactive simulation of a HAADF scan of a perfect crystal under vibration and drift, with the
vibration spectrum read back from the FFT satellites by the same method the lab's `haadf_vib.py`
uses -- so you can dial in a motion and see what the measurement makes of it.

**https://seankung.ca/HAADF-Vibration/**

One self-contained `index.html`, no build step, no npm, no network. Open it directly in Chrome, Edge or
Firefox (from disk it computes on the main thread, since Chrome refuses a Web Worker for a `file://`
page), or serve the folder with anything static and it runs in a worker:

```bash
python -m http.server 8124 --directory .
```

then open <http://localhost:8124/>. Append `?selftest` to run the built-in checks. The state lives in
the URL hash, so a link reproduces a run exactly. A file viewer that shows a static snapshot of a local
HTML file will show the page with no controls or figures, and a note saying the script is not running.

The defaults are a single frame of a real atomic-resolution acquisition: 512 &times; 512 over 16 nm
(31.25 pm/px), 8 &micro;s dwell, 32 &micro;s flyback, so 4128 &micro;s lines and a 242.2 Hz line rate.

**Noise and background, matched to real frames.** A raw single frame at these settings is a faint lattice on a large pedestal:
smoothed, the valley between columns is 0.91 of the column peak, the pixel noise is nearly the same at every
level, and the whole peak-to-valley modulation is only 2.5 &sigma;. Shot noise on a uniform background nine
times the column signal behaves exactly like that, so the default is background 9 &times; with 75 counts of
column signal. Run through the same measurement as the real frames:

| | valley/peak | mean/&sigma; | (peak &minus; valley)/&sigma; | Bragg/bg, strongest reflections |
|---|---|---|---|---|
| real frames | 0.90&ndash;0.91 | 23&ndash;26 | 2.5 | 100&ndash;117 |
| simulator, dose 60 | 0.92 | 23.6 | 1.9 | 95&ndash;110 |
| simulator, dose 100 | 0.93 | 30.5 | 2.4 | 121&ndash;143 |

Whether the real pedestal is specimen background or partly a detector offset cannot be told from these
frames -- the intensity range is too narrow -- but Bragg/bg, the number the vibration floor depends on, is
independent of any offset, and the readout reports it for every reflection read. A one-atom square lattice
has fewer strong reflections than a real crystal with several columns per cell, so its higher orders
(Bragg/bg ~15-20) are weaker than the real ones (60-80).

## What is simulated

**Specimen.** A square lattice of constant *a*, rotated by &theta; relative to the scan, one atom per
cell. HAADF is incoherent, so the image is the atom positions convolved with the probe intensity; the
probe is a Gaussian of half width at half maximum *r* (FWHM = 2*r*). For a square lattice that is
separable in the lattice frame, *I*(x&prime;, y&prime;) = *g*(x&prime;)&middot;*g*(y&prime;), with *g* a 1-D periodic sum of
Gaussians, tabulated once per run.

**Scan timing** -- the part that creates everything else. Pixel (i, j) of frame n is recorded at

```
t = n*T_frame + i*t_line + (j + 1/2)*dwell,    t_line = M*dwell + flyback,    T_frame = N*t_line
```

and records the lattice at its nominal position minus the specimen displacement at that instant,
**s**(t) = drift + vibration. Nothing else is imposed: the satellites, the loss inside a row, aliasing
at the line rate and the drift shear all follow from that timing.

**Vibration.** Up to four modes, each a Lorentzian of centre *f*, full width *w* and rms amplitude
*a*, along a direction or isotropic. A mode is generated as &xi;(t) = Re[z(t) e<sup>2&pi;ift</sup>] with z a
complex Ornstein-Uhlenbeck envelope of rate &pi;*w*, sampled finely enough that it barely moves
between samples and interpolated at every pixel time. Its one-sided power spectrum is exactly

```
S(f') = a^2 [ L(f' - f) + L(f' + f) ],     L(x) = (w/2pi) / (x^2 + (w/2)^2)
```

so the total rms is *a*; *w* = 0 is a pure tone of peak *a*&radic;2. An isotropic mode splits *a*
between independent x and y processes. Every random stream is seeded per mode and per axis, so changing
one knob does not re-roll the others.

**Drift** is a constant velocity, accumulated through the frames.

**Noise** is Poisson on the column signal (a given count at a column centre) plus a uniform background
given in units of that signal, or off.

## How the spectrum is read back

A line-for-line port of `vibration.py` (as of 2026-09-11), function names kept:

| JS | Python | what it does |
|---|---|---|
| `windowedMag` | `_sideband_power`'s `P` | \|fft2((a - mean) &times; Hann)\| |
| `detectGs` | `detect_gs` | strongest peaks, DC radius 5 bins, `fx*M >= 3`, min separation 4 |
| `fitReflections` | `fit_reflections` | up to 6 from a pool of 12, each with its FFT column to itself |
| `nDirections` | `n_directions` | distinct directions, 15&deg; tolerance |
| `sidebandPower` | `_sideband_power` | walk each column: Bragg, mean of the &plusmn;k satellites, background from column ix + M/3 |
| `invertBesselRatio` | `_invert_bessel_ratio` | J1/J0 table on 0-2.30 |
| `xOnly` / `tensor` | `_x_only` / `_tensor` | x-only reading; the x-and-y fit, weights (A/bg)&#8308; |
| `rowResponse` | `_row_response` | the row's Hann-weighted time average, divided back out (clamped at 0.25) |
| `SpecAcc` | `sequence_spectrum` | power average over frames, &radic;(Sxx+Syy), x-only fallback under 3 directions |
| `reduceSpectrum` | `ronchi_vib.reduce_spectrum` | floor, bands, peaks, with `haadf_vib`'s seven bands |
| `crosstalkHz` | `haadf_vib._crosstalk_from_reflections` | via `lattice._find_reflections` |

**Cross-checked against the Python:** frames simulated here and run through `vibration.sequence_spectrum`
give the same reflections, the same x+y / x-only decision, spectra equal to 1e-13, and identical floors,
bands and peaks. The one place they can differ is a tie: a noise-free axis-aligned lattice has
mirror-image peaks of exactly equal power, and which one an argmax picks then depends on the last bit of
the FFT.

## The curves

* **set, per bin** -- the spectrum you dialled in, integrated over each FFT bin (width line rate / N)
  and spread by the Hann window's [&frac14;, 1, &frac14;] power kernel: what an ideal satellite reading
  would show, drawn past Nyquist too.
* **expected reading** -- what the scan can report: that spectrum after each row's Hann-weighted time
  average (the exact discrete response of `np.hanning(M)` over pixels one dwell apart), folded at the
  line rate, then divided by the method's own correction at the frequency it appears at. Below
  Nyquist that undoes the row average and leaves the set spectrum; above it, the loss happened at the
  true frequency and the correction at the folded one, so aliased lines come back smaller.
* **extracted** -- the port above, run on the simulated frames. It should scatter about *expected reading*.
* **x-only reading** -- the method before 2026-09-10, which divides every reflection by gx.

## Things the simulator shows about the method

* **The in-row correction had the wrong shape -- found here, fixed in `vibration.py` on 2026-09-11.**
  The code divided by sinc(f&middot;t_active), the response of a uniform average over the row. But the
  satellite reading goes through the 2-D Hann window, which also weights the pixels along the row, so
  the average is Hann-weighted and its response is sinc(x)/(1 - x&sup2;). Readings came out high by
  1/(1 - x&sup2;): 7 % at x = 0.25, 25 % at x = 0.45, 33 % at Nyquist when the flyback is short. A
  tone at x = 0.426 read 34.85 pm where the Hann response predicted 34.58 and the uniform one 28.28.
  `_row_response` now divides out the exact discrete Hann response, and the same tone reads its true
  28.3 pm.
* **The x-only reading mis-books y.** A pure slow-axis tone of 28.46 pm read 17.18 pm by the old
  method, 28.45 by the x-and-y fit.
* **An axis-aligned square lattice cannot separate y.** (1,&plusmn;1) and (2,0)/(2,&plusmn;1) share FFT columns,
  the reflections left point one or two ways, and the fit falls back to x only. Rotate a few degrees.
* **Accuracy depends on how many cells the field holds.** At 32 cells across, a tone is recovered to
  0.1 %; at 16 cells, typically within 2 %; at 8 cells, several %, and at particular rotations more,
  where other reflections' leakage lands in a column being read.
* **Off-centre reflections respond slightly differently.** Within-row motion slides the +k and -k
  satellites &plusmn;x bins apart in fx, so a Bragg reflection sitting &delta; off a bin centre reads
  [W(&delta;+x) + W(&delta;-x)] / 2W(&delta;) instead of W(x): under 0.5 % below Nyquist, a few % for tones
  far above it.
* **Drift and the reflection's own window leakage** raise the lowest bins even with no vibration.

## Licence

MIT -- see `LICENSE`.
