"""The notebook figure widget."""

from __future__ import annotations

import pathlib
from typing import Callable

import anywidget
import traitlets

from ._data import to_colors, to_float32, to_sizes

_STATIC = pathlib.Path(__file__).parent / "static"
_WASM_JS = _STATIC / "rustyplot_wasm.js"
_WASM_BINARY = _STATIC / "rustyplot_wasm_bg.wasm"


def _load_wasm() -> tuple[str, bytes]:
    if not _WASM_JS.exists() or not _WASM_BINARY.exists():
        raise FileNotFoundError(
            "The rustyplot WebAssembly bundle is missing. Build it with "
            "`./build.sh` from the repository root."
        )
    return _WASM_JS.read_text(), _WASM_BINARY.read_bytes()


class Figure(anywidget.AnyWidget):
    """A GPU-rendered figure.

    Pan with the left mouse button, zoom with the wheel. Both are handled inside
    the WebAssembly module, so they never round-trip to the kernel.
    """

    _esm = _STATIC / "widget.js"
    _css = _STATIC / "widget.css"

    _wasm_js = traitlets.Unicode("").tag(sync=True)
    _wasm_binary = traitlets.Bytes(b"").tag(sync=True)

    _x = traitlets.Bytes(b"").tag(sync=True)
    _y = traitlets.Bytes(b"").tag(sync=True)
    _size = traitlets.Bytes(b"").tag(sync=True)
    _color = traitlets.Bytes(b"").tag(sync=True)
    # Bumped to tell the frontend that the buffers above have all been replaced.
    _revision = traitlets.Int(0).tag(sync=True)

    width = traitlets.Int(700).tag(sync=True)
    height = traitlets.Int(450).tag(sync=True)
    background = traitlets.List(
        trait=traitlets.Float(), default_value=[1.0, 1.0, 1.0, 1.0], minlen=4, maxlen=4
    ).tag(sync=True)
    #: Current view as ``[x_min, x_max, y_min, y_max]``, updated as the user navigates.
    view = traitlets.List(
        trait=traitlets.Float(), default_value=[0.0, 1.0, 0.0, 1.0], minlen=4, maxlen=4
    ).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._wasm_js, self._wasm_binary = _load_wasm()
        self._click_callbacks: list[Callable[[dict], None]] = []
        self.on_msg(self._handle_frontend_msg)

    def scatter(self, x, y, size=6.0, color=None) -> "Figure":
        """Draw points at `x`, `y` with the given pixel `size` and RGBA `color`."""
        xs = to_float32(x, "x")
        ys = to_float32(y, "y")
        if xs.size != ys.size:
            raise ValueError(f"`x` has {xs.size} elements but `y` has {ys.size}.")

        with self.hold_sync():
            self._x = xs.tobytes()
            self._y = ys.tobytes()
            self._size = to_sizes(size, xs.size).tobytes()
            self._color = to_colors(color, xs.size).tobytes()
            self._revision += 1
        return self

    def on_click(self, callback: Callable[[dict], None]) -> Callable:
        """Register `callback`, called with a dict describing each click.

        The dict holds the data coordinates under the cursor (``x``, ``y``) and,
        when a point was hit, its ``index`` and exact position.
        """
        self._click_callbacks.append(callback)
        return callback

    def _handle_frontend_msg(self, _widget, content, _buffers) -> None:
        if not isinstance(content, dict) or content.get("type") != "click":
            return
        event = {k: v for k, v in content.items() if k != "type"}
        for callback in self._click_callbacks:
            callback(event)


def scatter(x, y, size=6.0, color=None, **kwargs) -> Figure:
    """Create a figure showing a scatter plot of `x` against `y`."""
    return Figure(**kwargs).scatter(x, y, size=size, color=color)
