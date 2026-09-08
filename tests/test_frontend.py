"""Syntax checks for the hand-written frontend.

There is no bundler and no JS test runner, so a malformed `widget.js` would
otherwise only surface as an opaque SyntaxError inside the browser.
"""

import pathlib
import shutil
import subprocess

import pytest

STATIC = pathlib.Path(__file__).parent.parent / "python" / "rustyplot" / "static"
WIDGET_JS = STATIC / "widget.js"

node = shutil.which("node") or shutil.which("nodejs")


@pytest.mark.skipif(node is None, reason="node is not installed")
def test_widget_js_is_syntactically_valid():
    result = subprocess.run(
        [node, "--check", str(WIDGET_JS)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(node is None, reason="node is not installed")
def test_demo_page_script_is_syntactically_valid(tmp_path):
    """The demo page shares the widget's load path, so its script is worth checking too."""
    html = (pathlib.Path(__file__).parent.parent / "examples" / "demo.html").read_text()
    script = html.split('<script type="module">')[1].split("</script>")[0]
    module = tmp_path / "demo.mjs"
    module.write_text(script)
    result = subprocess.run(
        [node, "--check", str(module)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_widget_js_exports_a_render_function():
    """Guards against truncation even when node is unavailable."""
    source = WIDGET_JS.read_text()
    assert "export default" in source
    assert "function render(" in source
    assert source.count("{") == source.count("}")
