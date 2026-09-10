"""Handles returned by the plotting methods, kept mutable after creation.

An artist owns no drawing state of its own: it holds the ids identifying
which entry of which figure trait it refers to, and forwards every write
there (`AGENTS.md`'s "the Python objects hold ids and forward mutations").

Data updates deliberately do *not* reframe the axes, unlike the `scatter()`
or `plot()` call that created the artist. Updating data is the animation
path -- a slider driving a redraw -- where autoscaling on every frame would
make the view jitter as the data moves; matplotlib draws the same
distinction between `plot()` and `set_data()`.
"""

from __future__ import annotations

import numpy as np

from ._data import from_bytes, to_colors, to_float32, to_sizes, to_xy


class Artist:
    """Common data-mutation API for the handles `plot()`/`scatter()` return.

    Subclasses supply the storage: which trait lists hold this artist's x
    and y buffers, and what else has to change when the point count does.
    """

    def __init__(self, figure) -> None:
        self._figure = figure

    # --- subclass hooks ---------------------------------------------------

    def _read_xy(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _write_xy(self, xs: np.ndarray, ys: np.ndarray, **style) -> None:
        raise NotImplementedError

    # --- data properties --------------------------------------------------

    @property
    def x(self) -> np.ndarray:
        """This artist's x data, as a read-only float32 array."""
        return self._read_xy()[0]

    @x.setter
    def x(self, value) -> None:
        self.set(x=value)

    @property
    def y(self) -> np.ndarray:
        """This artist's y data, as a read-only float32 array."""
        return self._read_xy()[1]

    @y.setter
    def y(self, value) -> None:
        self.set(y=value)

    @property
    def xy(self) -> np.ndarray:
        """Both coordinates as one read-only `(n, 2)` array of points."""
        xs, ys = self._read_xy()
        # Backed by `bytes` rather than just flagged read-only: numpy will
        # let a caller flip `writeable` back on an array that owns its
        # memory, and the silent mutation that follows would never reach the
        # plot. An immutable buffer refuses outright.
        return np.frombuffer(np.column_stack((xs, ys)).tobytes(), dtype=np.float32).reshape(-1, 2)

    @xy.setter
    def xy(self, value) -> None:
        self.set(xy=value)

    def set(self, **kwargs) -> "Artist":
        """Change several properties in one update, and so one redraw.

        Assigning `x` and `y` separately is two updates, each drawn on its
        own; in between, the two are inconsistent, and if the new data has a
        different length the first assignment fails outright. Passing both
        here (or using `xy`) applies them together:

        >>> line.set(x=new_x, y=new_y)
        >>> pts.set(xy=np.random.random((1000, 2)), size=4)
        """
        xy = kwargs.pop("xy", None)
        x = kwargs.pop("x", None)
        y = kwargs.pop("y", None)
        if xy is not None and (x is not None or y is not None):
            raise ValueError(
                "pass either `xy` or `x`/`y`, not both; `xy` already carries "
                "both coordinates."
            )

        if xy is not None:
            xs, ys = to_xy(xy)
        else:
            old_x, old_y = self._read_xy()
            xs = to_float32(x, "x") if x is not None else old_x
            ys = to_float32(y, "y") if y is not None else old_y

        if xs.size != ys.size:
            # Naming which of the two was just written makes the common
            # mistake -- resizing one coordinate alone -- self-explanatory.
            changed = "x" if x is not None else "y" if y is not None else "xy"
            raise ValueError(
                f"`x` has {xs.size} elements but `y` has {ys.size}. Assigning "
                f"`{changed}` alone cannot change the number of points; pass both "
                f"at once, with `.xy = ...` or `.set(x=..., y=...)`."
            )

        self._write_xy(xs, ys, **kwargs)
        return self

    def _replaced(self, values: list, index: int, new_value) -> list:
        """A copy of `values` with entry `index` replaced.

        Traitlets only fires a change notification on reassignment, not on
        in-place mutation, so every write here builds a new list.
        """
        out = list(values)
        out[index] = new_value
        return out


class ScatterArtist(Artist):
    """The scatter points of one axes, returned by [`Axes.scatter`][rustyplot.Axes.scatter].

    An axes holds a single scatter series, so this handle refers to its
    axes; a second `scatter()` call replaces the data this one points at,
    and this handle then addresses the new series.
    """

    def __init__(self, figure, axes_index: int) -> None:
        super().__init__(figure)
        self._axes_index = axes_index

    def _read_xy(self) -> tuple[np.ndarray, np.ndarray]:
        fig = self._figure
        i = self._axes_index
        return from_bytes(fig._x[i]), from_bytes(fig._y[i])

    def _write_xy(self, xs: np.ndarray, ys: np.ndarray, **style) -> None:
        size = style.pop("size", None)
        color = style.pop("color", None)
        if style:
            raise TypeError(
                f"unexpected keyword argument(s) {', '.join(sorted(style))} for a "
                f"scatter artist; supported: x, y, xy, size, color."
            )

        fig = self._figure
        i = self._axes_index
        # `size`/`color` as the user last passed them, held on the figure
        # rather than on this handle: every handle for an axes addresses the
        # same single series, so a stale handle must not restore the style
        # of the `scatter()` call that created it over a later one's.
        old_size, old_color = fig._scatter_style[i]
        new_size = old_size if size is None else size
        new_color = old_color if color is None else color

        n = xs.size
        old_n = len(fig._x[i]) // 4
        # Converted before anything is committed, so a rejected style leaves
        # the artist exactly as it was rather than poisoning later updates.
        sizes = self._broadcast(to_sizes, new_size, n, old_n, "size")
        colors = self._broadcast(to_colors, new_color, n, old_n, "color")

        fig._scatter_style[i] = (new_size, new_color)
        with fig.hold_sync():
            fig._x = self._replaced(fig._x, i, xs.tobytes())
            fig._y = self._replaced(fig._y, i, ys.tobytes())
            fig._size = self._replaced(fig._size, i, sizes.tobytes())
            fig._color = self._replaced(fig._color, i, colors.tobytes())
            fig._mark_updated(scatter=i)

    def _broadcast(self, convert, value, n: int, old_n: int, name: str) -> np.ndarray:
        """Re-derive a per-point `size`/`color` array for `n` points.

        A scalar re-broadcasts silently, which is what makes changing the
        point count work without restating style. A per-point array of the
        old length cannot, so say how to fix it -- but only when the length
        really is the problem, established by re-converting at the old
        length: any other complaint (an unsupported colour spelling, say)
        is the accurate one and is left alone.
        """
        try:
            return convert(value, n)
        except ValueError as err:
            if n != old_n:
                try:
                    convert(value, old_n)
                except ValueError:
                    raise err from None
                raise ValueError(
                    f"the new data has {n} points, but this artist's `{name}` is a "
                    f"per-point array of the old length ({old_n}). Pass a new one in "
                    f"the same update, e.g. `.set(xy=..., {name}=...)`, or use a "
                    f"single {name} for every point."
                ) from None
            raise

    @property
    def size(self) -> np.ndarray:
        """Per-point marker size in pixels, as a read-only float32 array."""
        return from_bytes(self._figure._size[self._axes_index])

    @size.setter
    def size(self, value) -> None:
        self.set(size=value)

    @property
    def color(self) -> np.ndarray:
        """Per-point RGBA, as a read-only `(n, 4)` float32 array."""
        return from_bytes(self._figure._color[self._axes_index]).reshape(-1, 4)

    @color.setter
    def color(self, value) -> None:
        self.set(color=value)


class LineArtist(Artist):
    """One line, returned by [`Axes.plot`][rustyplot.Axes.plot].

    Unlike scatter, an axes can hold any number of lines, so this handle
    refers to a single one of them and is unaffected by later `plot()` calls.
    """

    def __init__(self, figure, line_index: int) -> None:
        super().__init__(figure)
        self._line_index = line_index

    def _read_xy(self) -> tuple[np.ndarray, np.ndarray]:
        fig = self._figure
        i = self._line_index
        return from_bytes(fig._line_x[i]), from_bytes(fig._line_y[i])

    def _write_xy(self, xs: np.ndarray, ys: np.ndarray, **style) -> None:
        if style:
            raise TypeError(
                f"unexpected keyword argument(s) {', '.join(sorted(style))} for a "
                f"line artist; supported: x, y, xy."
            )
        fig = self._figure
        i = self._line_index
        with fig.hold_sync():
            fig._line_x = self._replaced(fig._line_x, i, xs.tobytes())
            fig._line_y = self._replaced(fig._line_y, i, ys.tobytes())
            fig._mark_updated(line=i)
