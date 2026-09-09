"""The notebook figure widget: a grid of axes rendered by the wasm module.

All mutable state (scatter data, labels, view) lives in traitlets traits, not
in ephemeral messages: a widget's traits are replayed in full the first time
it is displayed, so `ax.scatter(...)` followed later by `fig` (the common
pattern in a notebook cell) still shows the data. A custom `send()` message,
by contrast, only reaches a browser-side model that already exists -- one
sent before the widget has ever been displayed is silently dropped. Custom
messages are used here only for the one thing that cannot happen before
display anyway: click events.
"""

from __future__ import annotations

import pathlib
from typing import Callable

import anywidget
import traitlets

from ._axes import Axes

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
    """A GPU-rendered figure: a grid of [`Axes`][rustyplot.Axes].

    Pan with the left mouse button, zoom with the wheel. Both are handled inside
    the WebAssembly module, so they never round-trip to the kernel.
    """

    _esm = _STATIC / "widget.js"
    _css = _STATIC / "widget.css"

    _wasm_js = traitlets.Unicode("").tag(sync=True)
    _wasm_binary = traitlets.Bytes(b"").tag(sync=True)

    nrows = traitlets.Int(1).tag(sync=True)
    ncols = traitlets.Int(1).tag(sync=True)

    width = traitlets.Int(700).tag(sync=True)
    height = traitlets.Int(450).tag(sync=True)
    background = traitlets.List(
        trait=traitlets.Float(), default_value=[1.0, 1.0, 1.0, 1.0], minlen=4, maxlen=4
    ).tag(sync=True)

    # One entry per axes, in row-major grid order.
    titles = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    xlabels = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    ylabels = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    #: Each axes' current view as ``[x_min, x_max, y_min, y_max]``. Kept in
    #: sync by the frontend as the user pans and zooms, so a read reflects
    #: what is currently on screen, not just the last value written here.
    view = traitlets.List(trait=traitlets.List(traitlets.Float())).tag(sync=True)
    # True for axes whose `xlim`/`ylim` was explicitly set, most recently,
    # after any `scatter()` call: on it, the frontend applies `view` as
    # written instead of overwriting it with a fresh autoscale, and it does
    # not sync the (unused) autoscaled range back over the user's value.
    # `scatter()` clears this back to False, since new data reframes the
    # axes again unless the user re-sets limits afterward -- "last call
    # wins", the same rule matplotlib applies between `scatter`/`set_xlim`.
    _view_explicit = traitlets.List(trait=traitlets.Bool()).tag(sync=True)

    # Point data, one raw buffer per axes; arrays never travel as JSON.
    _x = traitlets.List(trait=traitlets.Bytes()).tag(sync=True)
    _y = traitlets.List(trait=traitlets.Bytes()).tag(sync=True)
    _size = traitlets.List(trait=traitlets.Bytes()).tag(sync=True)
    _color = traitlets.List(trait=traitlets.Bytes()).tag(sync=True)
    # Bumped whenever any axes' point data is replaced. `_dirty_axes` names
    # which axes changed since the frontend last consumed it -- a list, not
    # a single index, because `with fig.hold():` can batch `scatter()` calls
    # to several axes into one sync; a scalar "last touched axes" would lose
    # every entry but the final one when traitlets coalesces the batch into
    # a single outgoing message. The frontend resets it to `[]` once applied.
    _revision = traitlets.Int(0).tag(sync=True)
    _dirty_axes = traitlets.List(trait=traitlets.Int()).tag(sync=True)

    # Line artists, one entry per *line* (not per axes), in the order
    # `plot()` was called: unlike scatter, which replaces its axes' single
    # series on every call, lines accumulate -- each `plot()` appends here
    # rather than overwriting a previous entry. There is no removal/in-place
    # edit API in v1, so `_lines_revision` only ever needs to grow; the
    # frontend responds to it by clearing and fully rebuilding every line
    # artist on every axes (see `rebuildLines` in `widget.js`).
    _line_axes = traitlets.List(trait=traitlets.Int()).tag(sync=True)
    _line_x = traitlets.List(trait=traitlets.Bytes()).tag(sync=True)
    _line_y = traitlets.List(trait=traitlets.Bytes()).tag(sync=True)
    _line_color = traitlets.List(trait=traitlets.Bytes()).tag(sync=True)
    _line_width = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    # "" (no line), "solid" or "dashed".
    _line_style = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    # "" (no marker) or "o" (circle).
    _line_marker = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    _line_marker_size = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    _lines_revision = traitlets.Int(0).tag(sync=True)

    def __init__(self, nrows: int = 1, ncols: int = 1, **kwargs):
        if nrows < 1 or ncols < 1:
            raise ValueError(
                f"`nrows` and `ncols` must each be at least 1, got nrows={nrows}, "
                f"ncols={ncols}."
            )
        n = nrows * ncols
        super().__init__(
            nrows=nrows,
            ncols=ncols,
            titles=[""] * n,
            xlabels=[""] * n,
            ylabels=[""] * n,
            view=[[0.0, 1.0, 0.0, 1.0] for _ in range(n)],
            _view_explicit=[False] * n,
            _x=[b""] * n,
            _y=[b""] * n,
            _size=[b""] * n,
            _color=[b""] * n,
            _line_axes=[],
            _line_x=[],
            _line_y=[],
            _line_color=[],
            _line_width=[],
            _line_style=[],
            _line_marker=[],
            _line_marker_size=[],
            **kwargs,
        )
        self._wasm_js, self._wasm_binary = _load_wasm()
        self._axes = tuple(Axes(self, i) for i in range(n))
        self._click_callbacks: list[Callable[[dict], None]] = []
        self.on_msg(self._handle_frontend_msg)

    @property
    def axes(self):
        """All axes, in row-major order. A 1x1 figure returns a single `Axes`."""
        return self._axes[0] if len(self._axes) == 1 else self._axes

    def hold(self):
        """Batch every property write and `scatter()` call made inside the
        block into one sync, instead of one message per assignment.

        >>> with fig.hold():
        ...     ax.xlabel = "time"
        ...     ax.ylabel = "value"
        """
        return self.hold_sync()

    def on_click(self, callback: Callable[[dict], None]) -> Callable:
        """Register `callback`, called with a dict describing each click.

        The dict holds which axes was clicked (``axes``), the data coordinates
        under the cursor (``x``, ``y``), and, when a point was hit, its
        ``index`` and exact position.
        """
        self._click_callbacks.append(callback)
        return callback

    def _handle_frontend_msg(self, _widget, content, _buffers) -> None:
        if not isinstance(content, dict) or content.get("type") != "click":
            return
        event = {k: v for k, v in content.items() if k != "type"}
        for callback in self._click_callbacks:
            callback(event)


def subplots(nrows: int = 1, ncols: int = 1, **kwargs) -> tuple[Figure, "Axes | tuple[Axes, ...]"]:
    """Create a figure with an `nrows x ncols` grid of axes.

    Returns `(fig, ax)` where `ax` is a single [`Axes`][rustyplot.Axes] for a
    1x1 figure, or a tuple of axes (row-major) otherwise.
    """
    fig = Figure(nrows=nrows, ncols=ncols, **kwargs)
    return fig, fig.axes


def scatter(x, y, size: float = 6.0, color=None, **kwargs) -> Figure:
    """Create a single-axes figure showing a scatter plot of `x` against `y`."""
    fig, ax = subplots(**kwargs)
    ax.scatter(x, y, size=size, color=color)
    return fig


def plot(x, y, line: dict | None = None, marker: dict | None = None, color=None, **kwargs) -> Figure:
    """Create a single-axes figure showing a line plot of `x` against `y`.

    See [`Axes.plot`][rustyplot.Axes.plot].
    """
    fig, ax = subplots(**kwargs)
    ax.plot(x, y, line=line, marker=marker, color=color)
    return fig
