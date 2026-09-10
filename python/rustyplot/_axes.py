"""One subplot within a Figure."""

from __future__ import annotations

from ._artist import LineArtist, ScatterArtist
from ._data import to_colors, to_float32, to_sizes

_LINE_STYLES = ("solid", "dashed")
_MARKER_STYLES = ("o",)


def _parse_line(line: dict | None) -> tuple[float, str, object]:
    """Returns `(width, style, color)`.

    `style` is `""` if `line` is `None`. `color` is `line["color"]` if
    present, else `None` (meaning: fall back to `plot()`'s top-level `color`
    argument).
    """
    if line is None:
        return 0.0, "", None
    style = line.get("style", "solid")
    if style not in _LINE_STYLES:
        raise ValueError(
            f"unsupported line style {style!r}; supported styles are: "
            f"{', '.join(repr(s) for s in _LINE_STYLES)}."
        )
    width = float(line.get("width", 1.5))
    return width, style, line.get("color")


def _parse_marker(marker: dict | None) -> tuple[str, float]:
    """Returns `(style, size)`, where `style` is `""` if `marker` is `None`."""
    if marker is None:
        return "", 0.0
    style = marker.get("style", "o")
    if style not in _MARKER_STYLES:
        raise ValueError(
            f"unsupported marker style {style!r}; supported styles are: "
            f"{', '.join(repr(s) for s in _MARKER_STYLES)}."
        )
    size = float(marker.get("size", 6.0))
    return style, size


class Axes:
    """One subplot within a [`Figure`][rustyplot.Figure].

    Created by [`subplots`][rustyplot.subplots]; there is no public constructor.
    Every property write replaces one entry in one of the figure's list
    traits (matplotlib's "one assignment, one scene delta", `AGENTS.md`);
    wrap several in ``with fig.hold():`` to batch them into a single sync.
    """

    def __init__(self, figure, index: int) -> None:
        self._figure = figure
        self._index = index

    def scatter(self, x, y, size: float = 6.0, color=None) -> ScatterArtist:
        """Draw points at `x`, `y` with the given pixel `size` and RGBA `color`.

        Returns a handle whose data stays mutable:

        >>> pts = ax.scatter(x, y)
        >>> pts.y = new_y
        """
        xs = to_float32(x, "x")
        ys = to_float32(y, "y")
        if xs.size != ys.size:
            raise ValueError(f"`x` has {xs.size} elements but `y` has {ys.size}.")
        sizes = to_sizes(size, xs.size)
        colors = to_colors(color, xs.size)

        fig = self._figure
        fig._scatter_style[self._index] = (size, color)
        with fig.hold_sync():
            fig._x = self._replaced(fig._x, xs.tobytes())
            fig._y = self._replaced(fig._y, ys.tobytes())
            fig._size = self._replaced(fig._size, sizes.tobytes())
            fig._color = self._replaced(fig._color, colors.tobytes())
            dirty = list(fig._dirty_axes)
            if self._index not in dirty:
                dirty.append(self._index)
            fig._dirty_axes = dirty
            fig._revision += 1
            # A fresh `scatter()` always reframes the axes to fit the new
            # data (the frontend autoscales on every push): clear any
            # explicit xlim/ylim so that autoscaled range isn't immediately
            # discarded in favour of limits meant for the old data.
            fig._view_explicit = self._replaced(fig._view_explicit, False)
        return ScatterArtist(fig, self._index)

    def plot(self, x, y, line: dict | None = None, marker: dict | None = None, color=None) -> LineArtist:
        """Draw a polyline through `x`, `y`.

        Unlike [`scatter`][rustyplot.Axes.scatter], each call to `plot()`
        *adds* a new line artist to this axes rather than replacing the
        previous one -- draw several lines with several calls.

        `line` controls the stroke: `{"style": "solid"|"dashed", "width": 2,
        "color": ...}`. `marker` controls the per-point marker: `{"style": "o",
        "size": 6}`. Either may be omitted (or `None`, the default), but at
        least one must be given for anything to be visible. `line["color"]`
        takes priority over the top-level `color` argument if both are given.

        Returns a handle whose data stays mutable:

        >>> line = ax.plot(x, y)
        >>> line.y = new_y
        """
        xs = to_float32(x, "x")
        ys = to_float32(y, "y")
        if xs.size != ys.size:
            raise ValueError(f"`x` has {xs.size} elements but `y` has {ys.size}.")
        width, style, line_color = _parse_line(line)
        marker_style, marker_size = _parse_marker(marker)
        if not style and not marker_style:
            raise ValueError(
                "`plot()` needs at least one of `line` or `marker` to draw anything; "
                "pass e.g. `line={'style': 'solid'}` or `marker={'style': 'o'}`."
            )
        colors = to_colors(line_color if line_color is not None else color, 1)

        fig = self._figure
        with fig.hold_sync():
            fig._line_axes = [*fig._line_axes, self._index]
            fig._line_x = [*fig._line_x, xs.tobytes()]
            fig._line_y = [*fig._line_y, ys.tobytes()]
            fig._line_color = [*fig._line_color, colors.tobytes()]
            fig._line_width = [*fig._line_width, width]
            fig._line_style = [*fig._line_style, style]
            fig._line_marker = [*fig._line_marker, marker_style]
            fig._line_marker_size = [*fig._line_marker_size, marker_size]
            fig._lines_revision += 1
            # Matches `scatter()`: a fresh call reframes the axes to fit the
            # new data, so any explicit xlim/ylim meant for the old data is
            # cleared rather than immediately overriding the autoscale.
            fig._view_explicit = self._replaced(fig._view_explicit, False)
        return LineArtist(fig, len(fig._line_axes) - 1)

    def _replaced(self, values: list, new_value) -> list:
        """A copy of `values` with this axes' entry replaced.

        Traitlets only fires a change notification on reassignment, not on
        in-place mutation, so every write here builds a new list.
        """
        out = list(values)
        out[self._index] = new_value
        return out

    @property
    def title(self) -> str:
        return self._figure.titles[self._index]

    @title.setter
    def title(self, value: str) -> None:
        self._figure.titles = self._replaced(self._figure.titles, str(value))

    @property
    def xlabel(self) -> str:
        return self._figure.xlabels[self._index]

    @xlabel.setter
    def xlabel(self, value: str) -> None:
        self._figure.xlabels = self._replaced(self._figure.xlabels, str(value))

    @property
    def ylabel(self) -> str:
        return self._figure.ylabels[self._index]

    @ylabel.setter
    def ylabel(self, value: str) -> None:
        self._figure.ylabels = self._replaced(self._figure.ylabels, str(value))

    @property
    def xlim(self) -> tuple[float, float]:
        """Current x view range.

        Reflects whatever the user is currently looking at: the frontend
        syncs this back after every pan or zoom, not just the last value
        written here.
        """
        x_min, x_max, _, _ = self._figure.view[self._index]
        return (x_min, x_max)

    @xlim.setter
    def xlim(self, value: tuple[float, float]) -> None:
        x_min, x_max = value
        _, _, y_min, y_max = self._figure.view[self._index]
        self._set_view(x_min, x_max, y_min, y_max)

    @property
    def ylim(self) -> tuple[float, float]:
        """Current y view range; see [`xlim`][rustyplot.Axes.xlim]."""
        _, _, y_min, y_max = self._figure.view[self._index]
        return (y_min, y_max)

    @ylim.setter
    def ylim(self, value: tuple[float, float]) -> None:
        y_min, y_max = value
        x_min, x_max, _, _ = self._figure.view[self._index]
        self._set_view(x_min, x_max, y_min, y_max)

    def _set_view(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        fig = self._figure
        with fig.hold_sync():
            fig.view = self._replaced(fig.view, [x_min, x_max, y_min, y_max])
            fig._view_explicit = self._replaced(fig._view_explicit, True)
