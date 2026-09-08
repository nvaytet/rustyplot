// Draws a textured quad rotated 90 degrees counter-clockwise: used to place
// a y-axis label (shaped normally, horizontally, into an offscreen texture
// by `text.rs`, since glyphon has no rotated-text primitive of its own)
// vertically alongside the y axis.

struct RotateUniforms {
    // Size of the whole canvas in physical pixels, to convert pixel rects to NDC.
    viewport: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: RotateUniforms;
@group(1) @binding(0) var label_texture: texture_2d<f32>;
@group(1) @binding(1) var label_sampler: sampler;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    // x, y, width, height in pixels of the *destination* (already rotated)
    // quad; (x, y) is its top-left corner.
    @location(0) rect: vec4<f32>,
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
    // A 90-degree CCW rotation of the source image: destination corner
    // (cx, cy) samples source corner (1 - cy, cx). See the derivation in
    // `text.rs`'s `render_ylabel` for why this mapping puts the string's
    // first character at the bottom and its last at the top.
    out.uv = vec2<f32>(1.0 - corner.y, corner.x);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(label_texture, label_sampler, in.uv);
}
