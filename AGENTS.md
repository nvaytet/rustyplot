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
serde scene-delta protocol for Python→wasm sync. Line, image and `pcolormesh`: rectilinear
non-uniform grids as one quad with bin-edge lookup textures and a binary search in the
shader; curvilinear grids as a triangle mesh with per-vertex colours. Axes, ticks, tick
formatting, titles, legend and colorbar, with text via `glyphon`.

**Phase 2 — interaction.** Box zoom, axis-locked variants, GPU id-buffer picking to
replace the brute-force hit test, a fuller event API (`on_move`, `on_zoom`) with
throttling, and composition with other ipywidgets.

**Phase 3 — 3D.** Orbit/trackball camera, depth and lighting, point clouds, surface and
mesh plots, isosurfaces, volume ray-casting.

**Phase 4 — polish.** Subplot/grid layout, themes, PNG export via headless offscreen
render, documentation and an example gallery.

Deferred deliberately, with the architecture kept open for them: SVG/PDF vector export
(needs a second `Backend` implementation), pandas/xarray/scipp input (goes in
`python/rustyplot/_data.py`, the single seam all array input passes through), and a
server-side rendering path for datasets too large to ship to the browser.

Excluded from v1: statistical chart types, geospatial projections, animation export, and
a matplotlib compatibility shim.

## API direction

Explicit and object-oriented, in the spirit of `plopp` rather than `pyplot`. Familiar
method names (`scatter`, `pcolormesh`, `imshow`), but no global figure state, no `gca()`.
Artists are objects with mutable properties that trigger a redraw. A matplotlib
compatibility layer may come later for migration; it is not a design goal.

## Conventions

- Comments explain what the code cannot say for itself. No restating the next line.
- Every behavioural change comes with a test. Core logic is testable without a GPU;
  prefer putting logic in `rustyplot-core` partly for that reason.
- Errors that a user can act on should say what to do, not just what failed. See the
  missing-wasm-bundle error in `_figure.py` and the `_data.py` messages for the tone.
- The notebook is the primary target. A feature that only works on the desktop is
  incomplete.
