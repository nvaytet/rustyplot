// Instanced scatter points: one triangle-strip quad per point, shaped into a
// disc by an analytic SDF so that point size costs no extra geometry.

struct Uniforms {
    scale: vec2<f32>,
    offset: vec2<f32>,
    viewport: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) center: vec2<f32>,
    @location(1) size: f32,
    @location(2) color: vec4<f32>,
) -> VsOut {
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    let corner = corners[vertex_index];

    let ndc = center * u.scale + u.offset;
    // Size is in pixels, so the quad is expanded in screen space, not data space.
    let half_extent_px = 0.5 * size * corner;
    let position = ndc + 2.0 * half_extent_px / u.viewport;

    var out: VsOut;
    out.clip_position = vec4<f32>(position, 0.0, 1.0);
    out.uv = corner;
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = length(in.uv);
    // One-pixel-wide analytic edge, independent of point size.
    let edge = fwidth(d);
    let alpha = 1.0 - smoothstep(1.0 - edge, 1.0, d);
    if alpha <= 0.0 {
        discard;
    }
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
