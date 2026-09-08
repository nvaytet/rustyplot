# rustyplot

GPU-accelerated plotting for Python, with a Rust core.

```python
import numpy as np
import rustyplot as rp

x = np.random.normal(size=1_000_000)
y = 0.6 * x + np.random.normal(size=1_000_000)

rp.scatter(x, y, size=2.0)
```

Renders in Jupyter and in a native desktop window from the same Rust code: the core is
compiled to WebAssembly for the notebook, where it draws through WebGPU (falling back to
WebGL2). Pan and zoom are handled inside the WebAssembly module, so navigating a figure
never round-trips to the kernel.

> **Status: early prototype.** Scatter plots only. No axes, ticks or labels yet.

## Installing

Requires the Rust toolchain, the `wasm32-unknown-unknown` target and `wasm-bindgen-cli`:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli
```

Then build the WebAssembly bundle and install the Python package:

```bash
./build.sh
pip install -e .
```

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

```bash
cargo test --workspace     # Rust
python3 -m pytest          # Python and frontend syntax
cargo run -p rustyplot-app --release   # native window, RUSTYPLOT_POINTS=5000000 to stress it
```

There is no JavaScript build step: the frontend is a single hand-written ES module, and
no `npm` packages are involved. `node` is used only to syntax-check that file during tests,
and is optional.

`examples/demo.html` renders the same plot outside Jupyter; serve the repository over HTTP
and open it to check the browser path on its own.

## Licence

BSD-3-Clause.
