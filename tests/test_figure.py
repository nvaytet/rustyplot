import numpy as np
import pytest

import rustyplot as rp


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    return rng.normal(size=100), rng.normal(size=100)


def test_subplots_returns_a_figure_and_single_axes():
    fig, ax = rp.subplots()
    assert isinstance(fig, rp.Figure)
    assert isinstance(ax, rp.Axes)
    assert fig.nrows == 1 and fig.ncols == 1


def test_subplots_grid_returns_a_tuple_of_axes():
    fig, axes = rp.subplots(1, 2)
    assert len(axes) == 2
    assert all(isinstance(a, rp.Axes) for a in axes)


def test_non_positive_grid_dimensions_are_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        rp.subplots(0, 2)
    with pytest.raises(ValueError, match="at least 1"):
        rp.subplots(1, -1)


def test_scatter_populates_this_axes_binary_traits(data):
    x, y = data
    fig, ax = rp.subplots()
    ax.scatter(x, y)
    assert (
        np.frombuffer(fig._x[0], dtype=np.float32).tolist()
        == x.astype(np.float32).tolist()
    )
    assert (
        np.frombuffer(fig._y[0], dtype=np.float32).tolist()
        == y.astype(np.float32).tolist()
    )
    assert len(fig._color[0]) == 4 * 4 * len(x)
    assert fig._revision == 1
    assert fig._dirty_axes == [0]


def test_data_is_not_json_encoded(data):
    """Arrays must travel as raw bytes; JSON would dominate the transfer cost.

    This is also what makes the data survive being set before the widget is
    ever displayed: state held in traits (unlike a custom `send()` message)
    is replayed in full the first time a browser-side model is created.
    """
    x, y = data
    fig, ax = rp.subplots()
    ax.scatter(x, y)
    assert isinstance(fig._x[0], bytes)
    assert len(fig._x[0]) == 4 * len(x)


def test_scatter_on_one_axes_does_not_touch_another(data):
    x, y = data
    fig, (left, right) = rp.subplots(1, 2)
    left.scatter(x, y)
    assert fig._x[0] != b""
    assert fig._x[1] == b""


def test_held_scatter_on_several_axes_marks_every_axes_dirty(data):
    """`_dirty_axes` must accumulate, not just record the last write.

    `hold_sync()` coalesces every trait write inside the block into one
    outgoing sync; a scalar "last touched axes" trait would then only ever
    report the final axes, silently dropping the others' updates on the
    frontend.
    """
    x, y = data
    fig, (left, right) = rp.subplots(1, 2)
    with fig.hold():
        left.scatter(x, y)
        right.scatter(x, y)
    assert sorted(fig._dirty_axes) == [0, 1]
    assert fig._x[0] != b""
    assert fig._x[1] != b""


def test_wasm_bundle_is_attached(data):
    fig, ax = rp.subplots()
    assert fig._wasm_binary[:4] == b"\x00asm"
    assert "Plot" in fig._wasm_js


def test_mismatched_lengths_are_rejected():
    fig, ax = rp.subplots()
    with pytest.raises(ValueError, match="has 3 elements but"):
        ax.scatter([1.0, 2.0, 3.0], [1.0, 2.0])


def test_redrawing_bumps_the_revision(data):
    fig, ax = rp.subplots()
    ax.scatter(*data)
    ax.scatter(*data)
    assert fig._revision == 2


def test_title_and_labels_are_stored_per_axes():
    fig, (left, right) = rp.subplots(1, 2)
    left.title = "left plot"
    right.xlabel = "time"
    right.ylabel = "value"
    assert fig.titles == ["left plot", ""]
    assert fig.xlabels == ["", "time"]
    assert fig.ylabels == ["", "value"]
    assert left.title == "left plot"
    assert right.xlabel == "time"


def test_xlim_reads_back_the_synced_view():
    fig, ax = rp.subplots()
    fig.view = [[1.0, 2.0, 3.0, 4.0]]
    assert ax.xlim == (1.0, 2.0)
    assert ax.ylim == (3.0, 4.0)


def test_xlim_write_updates_only_that_axes_view():
    fig, (left, right) = rp.subplots(1, 2)
    left.xlim = (0.0, 5.0)
    assert fig.view[0] == [0.0, 5.0, 0.0, 1.0]
    assert fig.view[1] == [0.0, 1.0, 0.0, 1.0]


def test_xlim_write_marks_that_axes_view_explicit():
    """The frontend must not silently discard an explicit xlim/ylim under a
    fresh autoscale; it uses `_view_explicit` to tell the two apart."""
    fig, ax = rp.subplots()
    assert fig._view_explicit == [False]
    ax.xlim = (0.0, 5.0)
    assert fig._view_explicit == [True]


def test_scatter_clears_the_explicit_view_flag(data):
    """A fresh `scatter()` reframes the axes, so any previous explicit
    xlim/ylim must not immediately override the new autoscale."""
    fig, ax = rp.subplots()
    ax.xlim = (0.0, 5.0)
    assert fig._view_explicit == [True]
    ax.scatter(*data)
    assert fig._view_explicit == [False]


def test_click_callbacks_receive_the_event(data):
    fig, ax = rp.subplots()
    seen = []
    fig.on_click(seen.append)
    fig._handle_frontend_msg(
        fig, {"type": "click", "axes": 0, "x": 1.5, "y": -2.0, "index": 7}, []
    )
    assert seen == [{"axes": 0, "x": 1.5, "y": -2.0, "index": 7}]


def test_unrelated_messages_are_ignored(data):
    fig, ax = rp.subplots()
    seen = []
    fig.on_click(seen.append)
    fig._handle_frontend_msg(fig, {"type": "something_else"}, [])
    assert seen == []
