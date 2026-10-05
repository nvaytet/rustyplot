# rustyplot

GPU-accelerated plotting for Python, with a Rust core. The same renderer runs natively
(Vulkan/Metal/DX12) and in a Jupyter notebook (WebGPU/WebGL2 via WebAssembly).

```python
import rustyplot as rp
rp.scatter(x, y)
```

## Why this exists

The GPU does the drawing regardless of host language, so Rust is not what makes the
pixels fast. The actual wins, and the things to protect when making decisions:

1. One portable core that compiles to both native and wasm, so notebook and desktop
   share every line of plotting logic.
2. Fast data reduction (decimation, binning, LOD, triangulation) with no Python overhead.
3. Interaction that never crosses the Python boundary — pan, zoom and hit-testing all
   happen in Rust, so they stay smooth regardless of kernel load or network latency.

Closest prior art: `fastplotlib`/`pygfx` (Python + wgpu, notebook via `jupyter_rfb`) and
`rerun` (Rust + wgpu, notebook via a wasm viewer). We differ from the former by keeping
the core in Rust and rendering client-side rather than streaming frames from the kernel.

## Repository layout

```
crates/rustyplot-core     scene description, view maths, interaction. No deps at all.
crates/rustyplot-render   wgpu implementation of the Backend trait. WGSL shaders.
crates/rustyplot-app      native winit window (shader iteration, desktop use)
crates/rustyplot-wasm     wasm-bindgen shell exposing Plot to JavaScript
python/rustyplot          Python API and the anywidget subclass
python/rustyplot/static   widget.js/.css plus the generated wasm bundle (git-ignored)
examples/                 demo.html (standalone browser check), quickstart.ipynb
tests/                    Python tests, including a node --check of the frontend
```

## Architectural constraints

These are load-bearing. Breaking one silently forecloses a planned feature.

- **`rustyplot-core` has no dependencies and must never gain one on `wgpu`, PyO3 or any
  windowing crate.** It owns the scene description and the `Backend` trait. This is what
  keeps a future SVG/vector backend possible without a rewrite. There is a test that will
  need to enforce this in CI.
- **The Rust crates must be usable without Python.** A Rust user should be able to depend
  on `rustyplot-core` + `rustyplot-render` and get a working plot. Python is one frontend,
  not the foundation.
- **Array data never travels as JSON.** numpy arrays go to the browser as raw bytes
  through anywidget's `Bytes` traits. Serialising a million floats as JSON would dominate
  every other cost.
- **Interaction logic lives in `rustyplot-core::interaction`**, not in JavaScript and not
  in Python, so the desktop and the notebook behave identically.
- **No npm.** The frontend is a single hand-written ES module. There is no `package.json`,
  no bundler and no JS dependencies. Node is used only as a local dev tool for
  `node --check`; it is not a build, CI or runtime dependency.

## Build and test

```bash
cargo test -p rustyplot-core     # pure logic, fast, no GPU needed
cargo check --workspace          # native build
./build.sh                       # wasm bundle -> python/rustyplot/static/
python3 -m pytest -q             # Python + frontend syntax checks
cargo run -p rustyplot-app --release   # native demo window (RUSTYPLOT_POINTS=5000000)
```

`build.sh` runs `cargo build --target wasm32-unknown-unknown` then `wasm-bindgen`. It
requires `rustup target add wasm32-unknown-unknown` and `cargo install wasm-bindgen-cli`.
The `wasm-bindgen` CLI version and the `wasm-bindgen` crate version must match exactly;
the crate is pinned to `=0.2.128`. `wasm-opt` is used if present and skipped otherwise.

Manual verification, in increasing order of coverage:

1. `cargo run -p rustyplot-app --release` — native window.
2. `python3 -m http.server 8123`, open `/examples/demo.html` — wasm in a browser without
   Jupyter. Reports backend, first-draw time and ms/frame. `?n=1000000` to stress it.
3. `jupyter lab examples/quickstart.ipynb` — the real target. Restart the kernel after
   rebuilding the wasm, since the bundle is cached in widget state.

## Traps discovered the hard way

- **Pin one graphics backend before touching the canvas.** Probing WebGPU calls
  `canvas.getContext("webgpu")`, which binds that canvas permanently. If the probe then
  fails — common on Linux, where `navigator.gpu` exists but yields no adapter — the WebGL2
  fallback can no longer create its context and you get a misleading "no suitable adapter"
  error. `widget.js` probes on a *throwaway* canvas and passes `use_webgpu` into
  `Plot::attach`, which restricts `wgpu::Instance` to a single backend.
- **wgpu 30 differs substantially from older releases.** `Instance::new` takes
  `InstanceDescriptor` by value and it has no `Default` (fields: `backends`, `flags`,
  `memory_budget_thresholds`, `backend_options`, `display`). `get_current_texture` returns
  a `CurrentSurfaceTexture` enum, not a `Result`. `present` moved to `queue.present(frame)`.
  `PipelineLayoutDescriptor` takes `&[Option<&BindGroupLayout>]` and `immediate_size`
  instead of push constant ranges. `VertexState::buffers` is `&[Option<VertexBufferLayout>]`.
  `multiview` became `multiview_mask`. When in doubt, read the source in
  `~/.cargo/registry/src/*/wgpu-30*/` rather than guessing from older documentation.
- **WebGL2 needs reduced limits**: `Limits::downlevel_webgl2_defaults()` on wasm.
- **The frontend has no compile step**, so a syntax error only surfaces as an opaque
  anywidget failure in the browser. `tests/test_frontend.py` runs `node --check` to catch
  this; keep it passing.
- **Return a teardown function from `render()`** in the widget. Without it, every cell
  re-run leaks a GPU context until the browser refuses to create more.

## Roadmap

Phase 0 is complete and verified. Later phases are the agreed plan, not yet built.

**Phase 0 — spike (done).** Scatter rendering, wheel zoom, drag pan, click callbacks
reaching Python, running both natively and in JupyterLab. 1M points interactive.

**Phase 1 — 2D primitives.** The retained, backend-agnostic draw list in core, plus a
serde scene-delta protocol for Python→wasm sync. `Axes2d` and figure layout land here,
not later: a scene is a grid of axes from the start, since retrofitting multiple
viewports onto a single-view core is far more invasive than building it in. Lines, and
`heatmap` for images and meshes (see [Heatmap design](#heatmap-design)). Axes
decorations, ticks, tick formatting, titles, legend and colorbar, with text via
`glyphon`.

**Phase 2 — interaction.** Box zoom, axis-locked variants, GPU id-buffer picking to
replace the brute-force hit test, a fuller event API (`on_move`, `on_zoom`) with
throttling, and composition with other ipywidgets.

**Phase 3 — 3D.** Orbit/trackball camera, depth and lighting, point clouds, surface and
mesh plots, isosurfaces, volume ray-casting.

**Phase 4 — polish.** Themes, PNG export via headless offscreen render, documentation and
an example gallery.

Deferred deliberately, with the architecture kept open for them: SVG/PDF vector export
(needs a second `Backend` implementation), pandas/xarray/scipp input (goes in
`python/rustyplot/_data.py`, the single seam all array input passes through), and a
server-side rendering path for datasets too large to ship to the browser.

Excluded from v1: statistical chart types, geospatial projections, animation export, and
a matplotlib compatibility shim.

## API direction

Matplotlib's *object-oriented* API, minus the global state and minus the getter/setter
pairs. The structure users already know — `Figure`, `Axes`, artists added to an existing
`Axes` — but with attribute assignment where matplotlib would use `set_*`.

```python
fig, ax = rp.subplots(1, 2)
pts = ax[0].scatter(x, y, size=6)
ax[0].xlim = (-1, 2)
ax[0].xlabel = "time"
pts.color = "red"
```

Adopted from matplotlib:

- `Figure` owns a grid of `Axes`; `rp.subplots()` returns `(fig, ax)`.
- Artists are created by methods on an `Axes` (`scatter`, `plot`, `pcolormesh`,
  `imshow`) and returned as handles that stay mutable.
- Familiar names for familiar things. No gratuitous renaming.

Rejected from matplotlib:

- No global current figure: no `pyplot` module, no `gca()`, no `plt.plot()`.
- No `get_xlim()`/`set_xlim()` pairs. Properties instead: `ax.xlim`, `ax.title`,
  `line.linewidth`.
- No keyword aliases (`c` for `color`, `lw` for `linewidth`).

Undecided: matplotlib's generic `Artist.set(**kwargs)` / `get(name)`. These are worth
keeping even though the per-attribute getters and setters are not. `set` is the natural
way to batch several changes into one delta — `pts.set(color="red", size=10)` is one
sync where two assignments are two — and `get` gives a uniform way to read a property
whose name is only known at run time. If they are kept they stay thin wrappers over the
same properties, with no separate code path and no attribute names that exist only there.

### Property rules

Properties are the main mutation surface, so their semantics have to be pinned down.

- **Properties are for state, methods are for actions.** Anything needing arguments
  beyond the new value stays a method: `ax.autoscale()`, `fig.savefig(path)`.
- **Return immutable values.** `ax.xlim` returns a tuple, never a list, so that
  `ax.xlim[0] = 3` fails loudly instead of silently not redrawing.
- **Some properties are bidirectional.** `ax.xlim` is not merely a Python attribute: the
  Rust core mutates the view during pan and zoom and syncs the result back. A read must
  reflect what the user is currently looking at, not the last value Python wrote.
- **Each assignment is one scene delta.** `with fig.hold():` batches a group of writes
  into a single sync and a single redraw.

### Mapping onto the Rust core

This is why the API shape is a core concern and not only a Python one.

| Python | `rustyplot-core` |
| --- | --- |
| `Figure` | `Scene` — background, layout, the list of axes |
| `Axes` | `Axes2d` — viewport rect, its own `View2d`, its own draw list |
| artist handle | opaque `ArtistId` (index + generation) into an axes' draw list |
| property write | one entry in a scene delta applied to the retained scene |

The Python objects hold ids and forward mutations; they own no drawing state themselves.
Layout — where each `Axes` sits within the figure — is computed in core, so the desktop
window and the notebook lay out identically.

A matplotlib compatibility shim (accepting `set_xlim` and friends) may come later to ease
migration. It is not a design goal, and nothing in the core should bend to accommodate it.

## Heatmap design

One method, `heatmap`, replaces matplotlib's `imshow`/`pcolormesh` split. It dispatches
on the shape of its arguments: `Z` alone is a uniform grid, 1D `x`/`y` are rectilinear,
2D `X`/`Y` are curvilinear. The split is an implementation detail users should not have
to know about, so it is not in the API.

### Unify on `locate()`, not on the data layout

Every variant has the same fragment tail: fetch the value for a cell, map it through the
colormap, write it. The only thing that differs is answering "which cell is this pixel
in?". So there is one pipeline drawing one quad, and a single swappable function:

```wgsl
fn locate(world: vec2f) -> vec2u
```

| `locate` variant | Cost | Covers |
| --- | --- | --- |
| affine | O(1) | uniform grids, plain images |
| analytic (log, and similar) | O(1) | log-spaced grids |
| binary search over edge textures | O(log n), 12 steps at 4096 | arbitrary rectilinear |
| 2D index texture (precomputed inverse map) | O(1), resolution-approximate | curvilinear |

**Binary search is the baseline; the others are opportunistic.** The general case needs
the edges sorted and nothing else — spacing may be large, then small, then large again.
The classifier that picks a cheaper `locate` must be *conservative*: if it cannot prove
the spacing is affine or log within tolerance, it falls back to binary search. A bug in
detection then costs 12 shader steps instead of 1 and can never produce a wrong image.
Keep that property; the temptation will be to make detection cleverer over time.

Classification is one O(nx) scan of a 1D array, in core, so it is testable without a GPU.

### Normalise at the boundary

Core converts every accepted input into one canonical form — **ascending edges plus an
index-flip flag** — and the shader never learns that centres or descending order exist.

- **Centres or edges** is decided by shape: `len(x) == nx + 1` is edges, `len(x) == nx`
  is centres. Interior edges are midpoints; the two outer edges extrapolate the end
  half-cells, as matplotlib does for `shading="nearest"`. `n == 1` has no spacing to
  infer and needs an explicit rule.
- **Descending coordinates flip the edges, never the data.** Reverse the 16 kB edge array
  at upload and return `nx - 1 - i` from `locate`. Reversing `Z` instead would copy 64 MB
  to produce an identical picture. Each axis is independent.
- **Rejected with actionable messages:** coordinates that are neither ascending nor
  descending (matplotlib renders nonsense here), non-finite coordinates, and lengths
  matching neither `nx` nor `nx + 1` — that error should name both shapes that would work.
- Duplicate coordinates are allowed: zero-width cells simply do not rasterise.

2D coordinates follow the same rule, `(ny+1, nx+1)` corners or `(ny, nx)` centres.
Descending does not apply to them.

### Why not a mesh

Per cell, a mesh costs 4 vertices x (2 f32 position + 1 f32 value) plus 6 u32 indices.
Sharing vertices between neighbours only halves that, because the index buffer does not
shrink and becomes 67% of the total — and sharing forces Gouraud shading, since a shared
corner carries one value for four cells. Flat per-cell colour requires unshared corners.

| 4096² grid | Memory |
| --- | --- |
| Z texture | 64 MB |
| instanced quads (corner textures, no vertex or index buffer) | 192 MB |
| mesh, shared vertices | 576 MB |
| mesh, flat shading | 1152 MB |

The flat mesh does not merely run slowly on the WebGL2 fallback: its 768 MB vertex buffer
exceeds `max_buffer_size` (256 MiB) and is rejected outright. Textures are bounded by
dimension rather than that limit, so they degrade gracefully instead.

Curvilinear grids are the only case needing real geometry, and in practice they come from
instrument or model geometry at ≤1000² , where instanced quads cost ~12 MB. Grids with
16M+ cells are essentially always uniform rasters. **The sizes do not overlap**, which is
what makes this affordable — but a uniform grid must never be routed through the
instanced path: that is 2 triangles versus 33M.

### Colormapping belongs in the shader

Z stays in a texture, the colormap is a small 1D LUT texture, and `vmin`/`vmax` are
uniforms. Changing the colormap or the limits is then a uniform write rather than
recomputing and re-uploading every colour — the difference between a smooth clim slider
and a stuttering one. Store a per-vertex or per-instance *scalar* for curvilinear grids,
never a per-vertex colour, for the same reason.

### Changing coordinates does not recreate the artist

`image.x = non_uniform_x` may move the renderer from the affine path to the binary-search
one. The `ArtistId`, its place in the draw list and the Python handle are unaffected;
which pipeline satisfies a draw-list entry is renderer-internal, and forcing a
recreate would leak backend detail into the API.

**Z is a `(ny, nx)` texture in every variant.** Only the coordinate-side resources differ,
so switching paths re-uploads a few kB of edges and never touches the large array.
Conversely, writing `z` never touches coordinate resources.

Expose the chosen path read-only for introspection; a one-line assignment that moves a
large grid onto a much more expensive path deserves to be visible.

### Capabilities, LOD and interaction

Whether data fits in a texture is a question of data size and backend limits only, never
of display resolution. Check per axis against `max_texture_dimension_2d`, check the
upload against `max_buffer_size`, and treat total bytes against a configurable budget —
there is no portable way to query free VRAM, so that part is policy, not a query.

The `Backend` trait reports these upward as capabilities so that core owns the "reduce by
4x in x" decision and it stays unit-testable. Reduce **to the texture limit, not to the
display resolution**: reducing to ~700 px looks right until the user zooms and no detail
ever appears.

Interaction then costs nothing in the common case. The quad's corners are in data space
and the view is a uniform, so pan and wheel zoom are a uniform write and a redraw —
exactly the path scatter and lines already take. When data has been reduced, the gesture
still never waits: redraw immediately at the resident LOD and swap in a sharper texture
on a debounce after the gesture settles. Hover readout is one binary search per
mouse-move, not per pixel, and belongs in `core::interaction`.

### Measured dead ends

Both of these look attractive and are not:

- **Resampling to canvas resolution on the CPU each frame.** For a 4096² grid: 12.6 ms at
  700x500, 131 ms at 1920x1080, 538 ms at 3840x2160 (native; wasm is slower still). A
  700x500 CSS canvas is 1400x1000 physical pixels on a retina display, so the realistic
  case is the middle row. Zoomed out is the slow case, because adjacent pixels land in
  distant cells and every lookup misses cache. A fragment shader is this same algorithm
  across thousands of cores, with no per-frame upload.
- **Replacing the binary search with an O(1) bucket LUT.** Lovely for near-uniform data
  (1 step, 16 kB) and useless in general: log-spaced edges need 549 forward-scan steps at
  16 kB and still 63 at 256 kB. Binary search is 12 steps *always*, which is the better
  trade when a whole warp waits for its slowest lane. Log spacing wants the analytic
  `locate`, not a bigger table.

CPU work earns its place as amortised data *reduction* — once, when the data or LOD level
changes, with a reducer you control — not as per-frame rendering.

### Traps specific to this

- **`FLOAT32_FILTERABLE` is an optional feature**, needing `OES_texture_float_linear` on
  the GL path. Linear filtering and mipmapping of an `R32Float` texture are not guaranteed
  on the WebGL2 fallback. Check it and fall back, or use a narrower format.
- **Mip the values, not the colours.** Averaging colours after the colormap gives a
  different and wrong answer. Zoomed-out aliasing reads as noise in the data, which is
  worse than blurriness because people believe it.
- **WebGL2 has no storage buffers at all** (`max_storage_buffers_per_shader_stage: 0`),
  so everything here must travel as textures.
- `downlevel_webgl2_defaults()` caps `max_texture_dimension_2d` at 2048, but
  `.using_resolution(adapter.limits())` — which `rustyplot-render` already calls — raises
  it to what the adapter really supports, typically 4096–16384. It does **not** raise
  `max_buffer_size`.

### Undecided

- **RGB(A) input.** `(ny, nx, 3|4)` needs no colormap, `vmin`/`vmax` or colorbar, and
  `heatmap` is a poor name for it. Either keep `heatmap` scalar-only and add `image`, or
  find a name covering both.
- **NaN and masked cells.** Trivial to make transparent in the shader, painful to retrofit
  into the colormap path later. The scientific target argues for designing it in now.
- **Curvilinear: exact or approximate.** Instanced quads are exact at any zoom; a
  precomputed index texture makes it a single quad like everything else but goes blocky
  past its raster resolution and needs re-rasterising on a debounce. Pick on whether users
  zoom deep into curvilinear grids.

## Conventions

- Comments explain what the code cannot say for itself. No restating the next line.
- Every behavioural change comes with a test. Core logic is testable without a GPU;
  prefer putting logic in `rustyplot-core` partly for that reason.
- Errors that a user can act on should say what to do, not just what failed. See the
  missing-wasm-bundle error in `_figure.py` and the `_data.py` messages for the tone.
- The notebook is the primary target. A feature that only works on the desktop is
  incomplete.
