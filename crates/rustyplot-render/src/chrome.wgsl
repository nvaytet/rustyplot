// Axis chrome (spines, tick marks): filled axis-aligned rectangles in pixel
// space, unlike scatter.wgsl which works in data space. Kept as a separate,
// simpler pipeline rather than overloading the scatter shader with a second mode.

struct ChromeUniforms {
    // Size of the whole canvas in physical pixels, to convert pixel rects to NDC.
    viewport: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: ChromeUniforms;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    // x, y, width, height in pixels; (x, y) is the top-left corner.
    @location(0) rect: vec4<f32>,
    @location(1) color: vec4<f32>,
) -> VsOut {
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    let corner = corners[vertex_index];
    let px = rect.xy + corner * rect.zw;
    let ndc = vec2<f32>(
        (px.x / u.viewport.x) * 2.0 - 1.0,
        1.0 - (px.y / u.viewport.y) * 2.0,
    );

    var out: VsOut;
    out.clip_position = vec4<f32>(ndc, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
