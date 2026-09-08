"""rustyplot: GPU-accelerated plotting for Python, with a Rust core.

import rustyplot as rp
rp.scatter(x, y)
"""

from ._figure import Figure, scatter

__version__ = "0.1.0"
__all__ = ["Figure", "scatter"]
