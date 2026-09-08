import numpy as np
import pytest

import rustyplot as rp


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    return rng.normal(size=100), rng.normal(size=100)


def test_scatter_populates_binary_traits(data):
    x, y = data
    fig = rp.scatter(x, y)
    assert (
        np.frombuffer(fig._x, dtype=np.float32).tolist()
        == x.astype(np.float32).tolist()
    )
    assert (
        np.frombuffer(fig._y, dtype=np.float32).tolist()
        == y.astype(np.float32).tolist()
    )
    assert len(fig._color) == 4 * 4 * len(x)
    assert fig._revision == 1


def test_data_is_not_json_encoded(data):
    """Arrays must travel as raw bytes; JSON would dominate the transfer cost."""
    x, y = data
    fig = rp.scatter(x, y)
    assert isinstance(fig._x, bytes)
    assert len(fig._x) == 4 * len(x)


def test_wasm_bundle_is_attached(data):
    fig = rp.scatter(*data)
    assert fig._wasm_binary[:4] == b"\x00asm"
    assert "Plot" in fig._wasm_js


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="has 3 elements but"):
        rp.scatter([1.0, 2.0, 3.0], [1.0, 2.0])


def test_redrawing_bumps_the_revision(data):
    fig = rp.scatter(*data)
    fig.scatter(*data)
    assert fig._revision == 2


def test_click_callbacks_receive_the_event(data):
    fig = rp.scatter(*data)
    seen = []
    fig.on_click(seen.append)
    fig._handle_frontend_msg(
        fig, {"type": "click", "x": 1.5, "y": -2.0, "index": 7}, []
    )
    assert seen == [{"x": 1.5, "y": -2.0, "index": 7}]


def test_unrelated_messages_are_ignored(data):
    fig = rp.scatter(*data)
    seen = []
    fig.on_click(seen.append)
    fig._handle_frontend_msg(fig, {"type": "something_else"}, [])
    assert seen == []
