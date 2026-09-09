// Instanced polyline segments: one triangle-strip quad per consecutive pair
// of points. Width is a screen-space extrusion (like scatter's point size),
// so the line stays a constant pixel width regardless of pan/zoom.
//
// Segments are flush ("butt") capped -- no extension along their own
// direction -- since round joins/caps are instead drawn as separate discs
// through the scatter pipeline (see `Renderer::upload`). Extending each
// segment independently left a diamond-shaped overlap at every vertex where
// two segments met at an angle, which read as an unintended small marker.
//
// Dashing uses `start_t`/`end_t`, a cumulative screen-space arc length
// computed once per polyline at upload time (rather than restarting at
// every segment, which made most segments -- shorter than one dash+gap for
// typical dense data -- render almost solid regardless of the requested
// pattern). This phase goes stale in scale as the user pans/zooms after
// upload, since recomputing it every frame would mean re-uploading on every
// pan/zoom tick, not just on data changes; direction, width and length
// still use the live view, so only the dash spacing (not position or
// thickness) can drift slightly until the next upload.

struct Uniforms {
    scale: vec2<f32>,
    offset: vec2<f32>,
    viewport: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) t: f32,
    @location(1) dashed: f32,
    @location(2) color: vec4<f32>,
};

// Pixel dash pattern; not user-configurable in v1.
const DASH_LEN: f32 = 8.0;
const GAP_LEN: f32 = 6.0;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) start: vec2<f32>,
    @location(1) end: vec2<f32>,
    @location(2) width: f32,
    @location(3) dashed: f32,
    @location(4) color: vec4<f32>,
    @location(5) start_t: f32,
    @location(6) end_t: f32,
) -> VsOut {
    let start_ndc = start * u.scale + u.offset;
    let end_ndc = end * u.scale + u.offset;
    // NDC -> pixel (y flipped, since NDC is up-positive and pixels are
    // down-positive); only the direction and length derived from this
    // matter, so the flip is consistent as long as it is undone the same
    // way when converting the extruded corner back to NDC below.
    let start_px = vec2<f32>((start_ndc.x + 1.0) * 0.5, (1.0 - start_ndc.y) * 0.5) * u.viewport;
    let end_px = vec2<f32>((end_ndc.x + 1.0) * 0.5, (1.0 - end_ndc.y) * 0.5) * u.viewport;

    let delta = end_px - start_px;
    let len = length(delta);
    let dir = select(vec2<f32>(1.0, 0.0), delta / max(len, 1e-6), len > 1e-6);
    let normal = vec2<f32>(-dir.y, dir.x);
    let half_width = width * 0.5;

    var corners = array<vec2<f32>, 4>(
        start_px - normal * half_width,
        end_px - normal * half_width,
        start_px + normal * half_width,
        end_px + normal * half_width,
    );
    let corner_px = corners[vertex_index];

    let corner_ndc = vec2<f32>(
        corner_px.x / u.viewport.x * 2.0 - 1.0,
        1.0 - corner_px.y / u.viewport.y * 2.0,
    );

    var out: VsOut;
    out.clip_position = vec4<f32>(corner_ndc, 0.0, 1.0);
    // 0/2 are the start-side vertices, 1/3 are the end-side vertices.
    out.t = select(start_t, end_t, vertex_index == 1u || vertex_index == 3u);
    out.dashed = dashed;
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if in.dashed > 0.5 {
        let phase = in.t % (DASH_LEN + GAP_LEN);
        if phase > DASH_LEN {
            discard;
        }
    }
    return in.color;
}
