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


def test_plot_appends_a_line_artist(data):
    x, y = data
    fig, ax = rp.subplots()
    ax.plot(x, y, line={"style": "solid", "width": 2}, marker={"style": "o"})
    assert fig._line_axes == [0]
    assert (
        np.frombuffer(fig._line_x[0], dtype=np.float32).tolist()
        == x.astype(np.float32).tolist()
    )
    assert fig._line_width == [2.0]
    assert fig._line_style == ["solid"]
    assert fig._line_marker == ["o"]
    assert fig._lines_revision == 1


def test_repeated_plot_calls_accumulate_lines(data):
    """Unlike `scatter()`, every `plot()` call must add a new line artist,
    not replace the axes' previous one."""
    x, y = data
    fig, ax = rp.subplots()
    ax.plot(x, y, line={"style": "solid"})
    ax.plot(x, y, line={"style": "dashed"})
    assert fig._line_axes == [0, 0]
    assert fig._line_style == ["solid", "dashed"]
    assert fig._lines_revision == 2


def test_plot_on_one_axes_does_not_touch_another(data):
    x, y = data
    fig, (left, right) = rp.subplots(1, 2)
    left.plot(x, y, line={"style": "solid"})
    assert fig._line_axes == [0]


def test_plot_requires_a_line_or_a_marker(data):
    x, y = data
    fig, ax = rp.subplots()
    with pytest.raises(ValueError, match="at least one of"):
        ax.plot(x, y)


def test_plot_rejects_an_unsupported_line_style(data):
    x, y = data
    fig, ax = rp.subplots()
    with pytest.raises(ValueError, match="unsupported line style"):
        ax.plot(x, y, line={"style": "dotted"})


def test_plot_rejects_an_unsupported_marker_style(data):
    x, y = data
    fig, ax = rp.subplots()
    with pytest.raises(ValueError, match="unsupported marker style"):
        ax.plot(x, y, marker={"style": "x"})


def test_plot_clears_the_explicit_view_flag(data):
    fig, ax = rp.subplots()
    ax.xlim = (0.0, 5.0)
    assert fig._view_explicit == [True]
    ax.plot(*data, line={"style": "solid"})
    assert fig._view_explicit == [False]


def test_plot_line_color_overrides_top_level_color(data):
    """`line["color"]` takes priority over the separate `color=` argument."""
    x, y = data
    fig, ax = rp.subplots()
    ax.plot(x, y, line={"style": "solid", "color": "#ff0000"}, color="#0000ff")
    got = np.frombuffer(fig._line_color[0], dtype=np.float32).tolist()
    want = np.frombuffer(rp._data.to_colors("#ff0000", 1).tobytes(), dtype=np.float32).tolist()
    assert got == want


def test_plot_line_color_falls_back_to_top_level_color(data):
    x, y = data
    fig, ax = rp.subplots()
    ax.plot(x, y, line={"style": "solid"}, color="#0000ff")
    got = np.frombuffer(fig._line_color[0], dtype=np.float32).tolist()
    want = np.frombuffer(rp._data.to_colors("#0000ff", 1).tobytes(), dtype=np.float32).tolist()
    assert got == want


def test_module_level_plot_creates_a_single_axes_figure(data):
    fig = rp.plot(*data, line={"style": "solid"})
    assert isinstance(fig, rp.Figure)
    assert fig._line_axes == [0]


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


# --- artist handles and in-place data updates -------------------------------


def test_scatter_returns_a_mutable_handle(data):
    fig, ax = rp.subplots()
    pts = ax.scatter(*data)
    assert isinstance(pts, rp.ScatterArtist)
    np.testing.assert_allclose(pts.x, np.float32(data[0]))
    np.testing.assert_allclose(pts.y, np.float32(data[1]))


def test_plot_returns_a_mutable_handle(data):
    fig, ax = rp.subplots()
    line = ax.plot(*data, line={"style": "solid"})
    assert isinstance(line, rp.LineArtist)
    np.testing.assert_allclose(line.y, np.float32(data[1]))


def test_assigning_y_replaces_only_y(data):
    fig, ax = rp.subplots()
    pts = ax.scatter(*data)
    new_y = np.arange(100, dtype=np.float64)
    pts.y = new_y
    np.testing.assert_allclose(pts.y, new_y)
    np.testing.assert_allclose(pts.x, np.float32(data[0]))
    assert fig._scatter_updates == [0]


def test_returned_arrays_are_read_only(data):
    fig, ax = rp.subplots()
    pts = ax.scatter(*data)
    with pytest.raises(ValueError):
        pts.y[0] = 1.0
    with pytest.raises(ValueError):
        pts.xy[0, 0] = 1.0


def test_xy_accepts_a_packed_point_array(data):
    fig, ax = rp.subplots()
    pts = ax.scatter(*data)
    points = np.random.default_rng(0).random((250, 2))
    pts.xy = points
    np.testing.assert_allclose(pts.x, np.float32(points[:, 0]))
    np.testing.assert_allclose(pts.y, np.float32(points[:, 1]))
    assert pts.xy.shape == (250, 2)


def test_xy_accepts_a_pair_of_arrays(data):
    fig, ax = rp.subplots()
    line = ax.plot(*data, line={"style": "solid"})
    line.xy = (np.arange(5), np.arange(5) * 2.0)
    np.testing.assert_allclose(line.x, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(line.y, [0, 2, 4, 6, 8])


def test_changing_only_one_coordinate_cannot_change_the_point_count(data):
    fig, ax = rp.subplots()
    pts = ax.scatter(*data)
    with pytest.raises(ValueError, match="pass both at once"):
        pts.y = np.arange(7)


def test_a_scalar_size_rebroadcasts_when_the_point_count_changes(data):
    fig, ax = rp.subplots()
    pts = ax.scatter(*data, size=4.0)
    pts.xy = np.zeros((5, 2))
    np.testing.assert_allclose(pts.size, [4.0] * 5)
    assert pts.color.shape == (5, 4)


def test_per_point_style_of_the_old_length_asks_for_a_new_one(data):
    fig, ax = rp.subplots()
    pts = ax.scatter(*data, size=np.full(100, 3.0))
    with pytest.raises(ValueError, match="per-point array of the old length"):
        pts.xy = np.zeros((5, 2))
    pts.set(xy=np.zeros((5, 2)), size=np.full(5, 2.0))
    np.testing.assert_allclose(pts.size, [2.0] * 5)


def test_updates_do_not_reframe_an_explicitly_limited_axes(data):
    fig, ax = rp.subplots()
    pts = ax.scatter(*data)
    ax.xlim = (-1.0, 1.0)
    assert fig._view_explicit == [True]
    pts.y = np.arange(100, dtype=np.float64)
    assert fig._view_explicit == [True]
    assert fig.view[0][:2] == [-1.0, 1.0]


def test_update_lists_name_only_the_latest_change(data):
    fig, ax = rp.subplots(1, 2)
    a = ax[0].scatter(*data)
    b = ax[1].scatter(*data)
    a.y = np.arange(100, dtype=np.float64)
    assert fig._scatter_updates == [0]
    b.y = np.arange(100, dtype=np.float64)
    assert fig._scatter_updates == [1]


def test_hold_batches_updates_into_one_revision(data):
    fig, ax = rp.subplots(1, 2)
    a = ax[0].scatter(*data)
    b = ax[1].scatter(*data)
    before = fig._data_revision
    with fig.hold():
        a.y = np.arange(100, dtype=np.float64)
        b.y = np.arange(100, dtype=np.float64)
    assert fig._scatter_updates == [0, 1]
    assert fig._data_revision == before + 2


def test_line_updates_are_addressed_globally(data):
    fig, ax = rp.subplots(1, 2)
    ax[0].plot(*data, line={"style": "solid"})
    second = ax[1].plot(*data, line={"style": "solid"})
    second.y = np.arange(100, dtype=np.float64)
    assert fig._line_updates == [1]
    assert fig._scatter_updates == []


def test_setting_both_xy_and_x_is_rejected(data):
    fig, ax = rp.subplots()
    pts = ax.scatter(*data)
    with pytest.raises(ValueError, match="not both"):
        pts.set(xy=np.zeros((5, 2)), x=np.zeros(5))


def test_unknown_keywords_are_rejected(data):
    fig, ax = rp.subplots()
    line = ax.plot(*data, line={"style": "solid"})
    with pytest.raises(TypeError, match="unexpected keyword"):
        line.set(colour="red")


def test_a_stale_handle_does_not_restore_its_own_style(data):
    fig, ax = rp.subplots()
    first = ax.scatter(*data, size=1.0)
    second = ax.scatter(*data, size=5.0)
    first.y = np.arange(100, dtype=np.float64)
    np.testing.assert_allclose(second.size, [5.0] * 100)


def test_a_rejected_style_leaves_the_artist_usable(data):
    fig, ax = rp.subplots()
    pts = ax.scatter(*data, size=4.0)
    with pytest.raises(ValueError):
        pts.color = "red"  # named colours are not supported
    np.testing.assert_allclose(pts.color[0], [0.12, 0.42, 0.78, 0.75], atol=1e-6)
    pts.y = np.arange(100, dtype=np.float64)
    np.testing.assert_allclose(pts.y, np.arange(100))


def test_an_unsupported_colour_is_not_reported_as_a_length_problem(data):
    fig, ax = rp.subplots()
    pts = ax.scatter(*data)
    with pytest.raises(ValueError, match="#rrggbb"):
        pts.set(xy=np.zeros((5, 2)), color="red")


def test_returned_arrays_cannot_be_made_writeable(data):
    fig, ax = rp.subplots()
    pts = ax.scatter(*data)
    for array in (pts.x, pts.xy, pts.size, pts.color):
        with pytest.raises(ValueError):
            array.flags.writeable = True


def test_a_two_by_two_array_is_two_points_not_a_pair():
    fig, ax = rp.subplots()
    pts = ax.scatter([0.0, 1.0], [0.0, 1.0])
    pts.xy = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(pts.x, [1.0, 3.0])
    np.testing.assert_allclose(pts.y, [2.0, 4.0])
    pts.xy = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_allclose(pts.x, [1.0, 2.0])
    np.testing.assert_allclose(pts.y, [3.0, 4.0])


def test_hold_batches_lines_and_scatter_together(data):
    fig, ax = rp.subplots()
    pts = ax.scatter(*data)
    line = ax.plot(*data, line={"style": "solid"})
    with fig.hold():
        pts.y = np.arange(100, dtype=np.float64)
        line.y = np.arange(100, dtype=np.float64)
    assert fig._scatter_updates == [0]
    assert fig._line_updates == [0]


def test_nested_holds_still_batch(data):
    fig, ax = rp.subplots(1, 2)
    a = ax[0].scatter(*data)
    b = ax[1].scatter(*data)
    with fig.hold():
        a.y = np.arange(100, dtype=np.float64)
        with fig.hold():
            b.y = np.arange(100, dtype=np.float64)
        assert fig._scatter_updates == [0, 1]


def test_an_exception_inside_hold_restores_batching(data):
    fig, ax = rp.subplots(1, 2)
    a = ax[0].scatter(*data)
    b = ax[1].scatter(*data)
    with pytest.raises(RuntimeError):
        with fig.hold():
            a.y = np.arange(100, dtype=np.float64)
            raise RuntimeError("boom")
    b.y = np.arange(100, dtype=np.float64)
    # Back to one-update-at-a-time: the aborted block's entry is not carried
    # over into the next, unrelated update.
    assert fig._scatter_updates == [1]


def test_line_handles_survive_later_plot_calls(data):
    fig, ax = rp.subplots(1, 2)
    first = ax[1].plot(*data, line={"style": "solid"})
    ax[0].plot(*data, line={"style": "solid"})
    ax[1].plot(*data, line={"style": "solid"})
    first.y = np.arange(100, dtype=np.float64)
    # Still the first entry of the flat list; later `plot()` calls append.
    assert fig._line_updates == [0]
    assert fig._line_axes == [1, 0, 1]
    np.testing.assert_allclose(first.y, np.arange(100))
