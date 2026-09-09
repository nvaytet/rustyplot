"""rustyplot: GPU-accelerated plotting for Python, with a Rust core.

import rustyplot as rp

fig, ax = rp.subplots()
ax.scatter(x, y, size=s, color="#000000")
fig
"""

from ._axes import Axes
from ._figure import Figure, plot, scatter, subplots

__version__ = "0.1.0"
__all__ = ["Axes", "Figure", "plot", "scatter", "subplots"]
