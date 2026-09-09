"""One subplot within a Figure."""

from __future__ import annotations

from ._data import to_colors, to_float32, to_sizes


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

    def scatter(self, x, y, size: float = 6.0, color=None) -> "Axes":
        """Draw points at `x`, `y` with the given pixel `size` and RGBA `color`."""
        xs = to_float32(x, "x")
        ys = to_float32(y, "y")
        if xs.size != ys.size:
            raise ValueError(f"`x` has {xs.size} elements but `y` has {ys.size}.")
        sizes = to_sizes(size, xs.size)
        colors = to_colors(color, xs.size)

        fig = self._figure
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
        return self

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
