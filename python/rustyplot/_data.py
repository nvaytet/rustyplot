"""Conversion of user data into the flat float32 buffers the renderer expects.

This module is the single seam through which all array input passes. Only numpy is
supported today; pandas, xarray and scipp support will be added here and nowhere else.
"""

from __future__ import annotations

import numpy as np

# Libraries we intend to support natively later, detected to give a useful error now.
_PLANNED = {
    "pandas": "pass `series.to_numpy()`",
    "xarray": "pass `array.values`",
    "scipp": "pass `variable.values`",
}


def _as_array(data, name: str) -> np.ndarray:
    module = type(data).__module__.split(".")[0]
    if module in _PLANNED:
        raise TypeError(
            f"rustyplot does not accept {module} objects yet: {_PLANNED[module]} "
            f"for `{name}` instead."
        )
    return np.asarray(data)


def to_float32(data, name: str) -> np.ndarray:
    """Return a contiguous 1-D float32 copy of `data`."""
    array = _as_array(data, name)
    if array.ndim != 1:
        raise ValueError(f"`{name}` must be one-dimensional, got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"`{name}` is empty; there is nothing to plot.")
    return np.ascontiguousarray(array, dtype=np.float32)


def to_sizes(size, n: int) -> np.ndarray:
    """Broadcast a scalar or per-point size into a float32 array of length `n`."""
    if np.isscalar(size):
        return np.full(n, float(size), dtype=np.float32)
    sizes = to_float32(size, "size")
    if sizes.size != n:
        raise ValueError(f"`size` has {sizes.size} elements but `x` has {n}.")
    return sizes


def to_xy(value) -> tuple[np.ndarray, np.ndarray]:
    """Split a combined x/y value into two contiguous float32 arrays.

    Which of the two accepted forms is meant is decided by the container,
    not by the shape, so the answer never changes with the number of points:

    - a `tuple` or `list` of exactly two items is a pair, `(x, y)`;
    - anything else must be an `(n, 2)` array of points -- the packed form
      numpy code tends to produce, e.g. `np.random.random((1000, 2))`.

    So a `(2, 2)` ndarray is two points, never a pair; pass the pair as a
    tuple, `(x, y)`, if that is what is meant.
    """
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return to_float32(value[0], "x"), to_float32(value[1], "y")

    array = _as_array(value, "xy")
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            f"`xy` must be an (n, 2) array of points or a pair `(x, y)`, got shape "
            f"{array.shape}."
        )
    points = np.ascontiguousarray(array, dtype=np.float32)
    return to_float32(points[:, 0], "x"), to_float32(points[:, 1], "y")


def from_bytes(data: bytes) -> np.ndarray:
    """Read a float32 buffer back out of a trait, as a read-only array.

    Read-only so that `artist.y[0] = 3` fails loudly rather than mutating a
    copy that is never drawn: the buffer the renderer sees is only replaced
    by assigning the property outright (`AGENTS.md`'s "return immutable
    values" rule, applied to arrays).
    """
    array = np.frombuffer(data, dtype=np.float32)
    array.flags.writeable = False
    return array


def _parse_hex(value: str) -> tuple[float, float, float, float]:
    text = value.lstrip("#")
    if len(text) not in (6, 8):
        raise ValueError(f"`color` must be #rrggbb or #rrggbbaa, got {value!r}.")
    channels = [int(text[i : i + 2], 16) / 255.0 for i in range(0, len(text), 2)]
    if len(channels) == 3:
        channels.append(1.0)
    return tuple(channels)


def to_colors(color, n: int) -> np.ndarray:
    """Return a flat float32 RGBA array of length `4 * n`."""
    if color is None:
        color = (0.12, 0.42, 0.78, 0.75)
    if isinstance(color, str):
        color = _parse_hex(color)

    array = _as_array(color, "color").astype(np.float32, copy=False)

    if array.ndim == 1 and array.size in (3, 4):
        rgba = np.ones(4, dtype=np.float32)
        rgba[: array.size] = array
        return np.tile(rgba, n)

    if array.ndim == 2 and array.shape[0] == n and array.shape[1] in (3, 4):
        rgba = np.ones((n, 4), dtype=np.float32)
        rgba[:, : array.shape[1]] = array
        return np.ascontiguousarray(rgba.reshape(-1))

    raise ValueError(
        f"`color` must be a single RGB(A) value or an ({n}, 3)/({n}, 4) array, "
        f"got shape {array.shape}."
    )
