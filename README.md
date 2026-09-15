# rustyplot

GPU-accelerated plotting for Python, with a Rust core.

```python
import numpy as np
import rustyplot as rp

x = np.random.normal(size=1_000_000)
y = 0.6 * x + np.random.normal(size=1_000_000)

fig, ax = rp.subplots()
ax.scatter(x, y, size=2.0)
ax.xlabel = "x"
ax.title = "A million points"
fig
```

Renders in Jupyter and in a native desktop window from the same Rust code: the core is
compiled to WebAssembly for the notebook, where it draws through WebGPU (falling back to
WebGL2). Pan and zoom are handled inside the WebAssembly module, so navigating a figure
never round-trips to the kernel.

> **Status: early prototype.** Scatter and line plots, axes decorations and a pan/zoom
> toolbar. No images, `pcolormesh`, legends or colorbars yet, and no 3D.

## Installing

```bash
pip install rustyplot
```

The wheel ships the prebuilt WebAssembly bundle, so no Rust toolchain is needed to use
it. To build from source instead, see [Development](#development).

## Usage

The API follows matplotlib's *object-oriented* interface: a `Figure` owns a grid of
`Axes`, and artists are created by methods on an `Axes`. There is no global current
figure — no `pyplot`, no `gca()`. Displaying a figure means ending a cell with it.

### Figures and axes

`subplots()` returns `(fig, ax)`, with `ax` a single `Axes` for a 1x1 figure or a
row-major tuple otherwise:

```python
fig, ax = rp.subplots(1, 2)
ax[0].scatter(x, y)
ax[1].plot(t, signal, line={"style": "solid"})
fig
```

For a one-liner, `rp.scatter(x, y)` and `rp.plot(x, y)` build a single-axes figure and
return it directly.

### Properties instead of getters and setters

Where matplotlib has `set_xlim()`/`get_xlim()`, rustyplot has a property. Reads return
immutable values, and `ax.xlim` reflects what you are *currently looking at*, so it
follows along as you pan and zoom:

```python
ax.title = "Measured response"
ax.xlabel = "time (s)"
ax.ylabel = "amplitude"
ax.xlim = (-1.0, 2.0)

ax.xlim          # -> (-1.0, 2.0), a tuple
```

### Lines

`plot()` needs at least one of `line` or `marker`, and each call *adds* a line rather
than replacing the previous one:

```python
ax.plot(t, raw, line={"style": "dashed", "width": 1.0}, color="#999999")
ax.plot(t, fitted, line={"style": "solid", "width": 2.0}, marker={"style": "o", "size": 4})
```

`scatter()` differs: an axes holds a single scatter series, so a second call replaces it.

### Mutable artist handles

Both `scatter()` and `plot()` return a handle that stays live. Assigning to it updates
the existing GPU buffers in place instead of rebuilding the plot:

```python
pts = ax.scatter(x, y)
pts.y = np.sin(x)                   # redraws without re-sending x
pts.set(color="#ff0000", size=10)   # several changes, one sync
```

Scatter handles expose `x`, `y`, `xy`, `size` and `color`; line handles expose `x`, `y`
and `xy`. Colours are `#rrggbb`/`#rrggbbaa` strings, an RGB(A) tuple, or a per-point
array; named colours are not supported yet.

### Batching updates

Each assignment is normally one message to the browser. `fig.hold()` collapses a group
of them into a single sync and a single redraw:

```python
with fig.hold():
    ax.xlabel = "time"
    ax.ylabel = "value"
    ax.xlim = (0, 10)
```

### Clicks

```python
@fig.on_click
def report(event):
    print(event["axes"], event["x"], event["y"], event.get("index"))
```

The event gives the axes clicked and the data coordinates under the cursor, plus the
index of the nearest point when one was hit.

## Layout

| Path | Contents |
| --- | --- |
| `crates/rustyplot-core` | Scene description, view maths, interaction. No GPU, no Python. |
| `crates/rustyplot-render` | `wgpu` renderer, WGSL shaders. |
| `crates/rustyplot-app` | Native window, for developing without a browser. |
| `crates/rustyplot-wasm` | WebAssembly bindings driven from JavaScript. |
| `python/rustyplot` | Python API and the anywidget frontend. |

`rustyplot-core` deliberately has no dependencies, so that a non-GPU backend (SVG output,
for instance) can be added later without disturbing it.

## Development

Building from source needs the Rust toolchain, the `wasm32-unknown-unknown` target and
`wasm-bindgen-cli`. The CLI version must match the `wasm-bindgen` crate version pinned in
`crates/rustyplot-wasm/Cargo.toml` exactly:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli
```

Then build the WebAssembly bundle and install the package in editable mode:

```bash
./build.sh
pip install -e '.[test]'
```

`build.sh` writes the bundle into `python/rustyplot/static/`, which is git-ignored and
rebuilt rather than committed. Restart the Jupyter kernel after rebuilding, since the
bundle is cached in widget state.

```bash
cargo test --workspace     # Rust
python3 -m pytest          # Python and frontend syntax
cargo run -p rustyplot-app --release   # native window, RUSTYPLOT_POINTS=5000000 to stress it
```

There is no JavaScript build step: the frontend is a single hand-written ES module, and
no `npm` packages are involved. `node` is used only to syntax-check that file during
tests, and is optional.

`examples/demo.html` renders the same plot outside Jupyter; serve the repository over
HTTP and open it to check the browser path on its own.

## Licence

BSD-3-Clause.
