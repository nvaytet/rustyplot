"""rustyplot: GPU-accelerated plotting for Python, with a Rust core.

import rustyplot as rp

fig, ax = rp.subplots()
ax.scatter(x, y, size=s, color="#000000")
fig
"""

from importlib import metadata as _metadata

from ._artist import LineArtist, ScatterArtist
from ._axes import Axes
from ._figure import Figure, plot, scatter, subplots

try:
    __version__ = _metadata.version("rustyplot")
except _metadata.PackageNotFoundError:
    # Imported straight from a checkout rather than an install, which is how
    # the test suite runs (`pythonpath = ["python"]`). The real version comes
    # from the git tag at build time and there is nothing to read here.
    __version__ = "0.0.0+unknown"
__all__ = [
    "Axes",
    "Figure",
    "LineArtist",
    "ScatterArtist",
    "plot",
    "scatter",
    "subplots",
]
