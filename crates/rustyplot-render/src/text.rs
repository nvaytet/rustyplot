//! Tick labels, axis labels and titles, drawn with `glyphon`.
//!
//! The font is compiled into the binary (see `assets/`), so text renders
//! identically on native and in the browser with no network fetch and no
//! dependency on whatever fonts happen to be installed on the host: the
//! constraint is the same one that keeps the rest of the wasm bundle
//! self-contained (see `AGENTS.md`, "No npm").

use glyphon::{
    Attrs, Buffer, Cache, Color as TextColor, Family, FontSystem, Metrics, Resolution, Shaping,
    SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport as TextViewport, fontdb,
};
use rustyplot_core::Axes2d;

use crate::chrome;

const FONT_BYTES: &[u8] = include_bytes!("../assets/RustyplotSans.ttf");
const FONT_FAMILY: &str = "Rustyplot Sans";

pub const TICK_FONT_SIZE: f32 = 11.0;
pub const AXIS_LABEL_FONT_SIZE: f32 = 13.0;
pub const TITLE_FONT_SIZE: f32 = 15.0;
const LINE_HEIGHT_FACTOR: f32 = 1.3;
const TICK_LABEL_GAP: f32 = 3.0;
const TEXT_COLOR: TextColor = TextColor::rgb(40, 40, 40);

fn line_height(font_size: f32) -> f32 {
    font_size * LINE_HEIGHT_FACTOR
}

/// Everything needed to lay out and draw text, owned by the [`crate::Renderer`].
///
/// Font loading is explicit (a hand-built [`fontdb::Database`] containing only
/// our own font) rather than `FontSystem::new()`, which would otherwise probe
/// and index every font installed on the host: slow, non-reproducible between
/// machines, and pointless work on the web where there is no such thing.
pub struct TextPipeline {
    font_system: FontSystem,
    swash_cache: SwashCache,
    viewport: TextViewport,
    atlas: TextAtlas,
    renderer: TextRenderer,

    // A y-axis label is shaped horizontally like any other text, then drawn
    // into this dedicated offscreen atlas/renderer/viewport so the resulting
    // texture can be composited as a quad rotated 90° next to the y axis
    // (glyphon itself has no rotated-text primitive). Kept fully separate
    // from `atlas`/`renderer` above rather than reused: two `prepare`/`render`
    // calls into the same atlas within one frame, before a single
    // `queue.submit()`, would risk the second call's glyph uploads landing
    // before the first call's render pass executes, corrupting it (the same
    // hazard documented on `AxesGpu` in `lib.rs`).
    ylabel_viewport: TextViewport,
    ylabel_atlas: TextAtlas,
    ylabel_renderer: TextRenderer,
}

impl TextPipeline {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let mut db = fontdb::Database::new();
        db.load_font_data(FONT_BYTES.to_vec());
        db.set_sans_serif_family(FONT_FAMILY);

        let font_system = FontSystem::new_with_locale_and_db("en-US".to_string(), db);
        let swash_cache = SwashCache::new();
        // `Cache` just holds shared pipelines/layouts (keyed internally by
        // format), so it is safe and cheap to share between both atlases.
        let cache = Cache::new(device);
        let viewport = TextViewport::new(device, &cache);
        let mut atlas = TextAtlas::new(device, queue, &cache, format);
        let renderer = TextRenderer::new(&mut atlas, device, wgpu::MultisampleState::default(), None);

        let ylabel_viewport = TextViewport::new(device, &cache);
        let mut ylabel_atlas = TextAtlas::new(device, queue, &cache, format);
        let ylabel_renderer =
            TextRenderer::new(&mut ylabel_atlas, device, wgpu::MultisampleState::default(), None);

        Self {
            font_system,
            swash_cache,
            viewport,
            atlas,
            renderer,
            ylabel_viewport,
            ylabel_atlas,
            ylabel_renderer,
        }
    }

    /// Build one shaped, measured [`Buffer`] for a single line of text.
    fn shape(&mut self, text: &str, font_size: f32) -> Buffer {
        let mut buffer =
            Buffer::new(&mut self.font_system, Metrics::new(font_size, line_height(font_size)));
        {
            let mut borrowed = buffer.borrow_with(&mut self.font_system);
            borrowed.set_size(None, None);
            borrowed.set_text(
                text,
                &Attrs::new().family(Family::Name(FONT_FAMILY)),
                Shaping::Basic,
                None,
            );
            borrowed.shape_until_scroll(false);
        }
        buffer
    }

    /// Measured width of a shaped buffer's first (only) line, in pixels.
    fn width(buffer: &Buffer) -> f32 {
        buffer
            .layout_runs()
            .next()
            .map(|r| r.line_w)
            .unwrap_or(0.0)
    }

    /// Upload glyph data and queue every label for one axes, ready for [`Self::render`].
    ///
    /// Buffers must outlive the `queue` call (glyphon borrows them), so this
    /// takes a `&mut Vec<Buffer>` to own them for the caller's frame.
    pub fn queue_axes(&mut self, axes: &Axes2d, buffers: &mut Vec<(Buffer, f32, f32, f32)>) {
        let rect = axes.rect;

        let (xticks, xlabels) = axes.xticks();
        for (v, label) in xticks.iter().zip(xlabels.iter()) {
            let buffer = self.shape(label, TICK_FONT_SIZE);
            let w = Self::width(&buffer);
            let x = chrome::xtick_x(axes, *v) - w / 2.0;
            let y = rect.y + rect.height + chrome::TICK_LENGTH + TICK_LABEL_GAP;
            buffers.push((buffer, x, y, TICK_FONT_SIZE));
        }

        let (yticks, ylabels) = axes.yticks();
        for (v, label) in yticks.iter().zip(ylabels.iter()) {
            let buffer = self.shape(label, TICK_FONT_SIZE);
            let w = Self::width(&buffer);
            let x = rect.x - chrome::TICK_LENGTH - TICK_LABEL_GAP - w;
            let y = chrome::ytick_y(axes, *v) - line_height(TICK_FONT_SIZE) / 2.0;
            buffers.push((buffer, x, y, TICK_FONT_SIZE));
        }

        if !axes.xlabel.is_empty() {
            let buffer = self.shape(&axes.xlabel, AXIS_LABEL_FONT_SIZE);
            let w = Self::width(&buffer);
            let x = rect.x + rect.width / 2.0 - w / 2.0;
            let y = rect.y
                + rect.height
                + chrome::TICK_LENGTH
                + TICK_LABEL_GAP
                + line_height(TICK_FONT_SIZE)
                + TICK_LABEL_GAP;
            buffers.push((buffer, x, y, AXIS_LABEL_FONT_SIZE));
        }

        // `ylabel` is rendered separately (see `shape_ylabel`/`render_ylabel`)
        // since it needs an offscreen texture to rotate; `title` still runs
        // through the normal horizontal path. Its baseline is derived from
        // `rect.y`, not a figure-global constant: `Axes2d::margins` reserves
        // exactly `FIGURE_PADDING + line_height(TITLE_FONT_SIZE)` above
        // `rect` for the title, so subtracting the line height back off
        // `rect.y` lands `FIGURE_PADDING` below this axes' own cell top --
        // matching every row of a multi-row grid, not just the first.
        if !axes.title.is_empty() {
            let buffer = self.shape(&axes.title, TITLE_FONT_SIZE);
            let w = Self::width(&buffer);
            let x = rect.x + rect.width / 2.0 - w / 2.0;
            let y = rect.y - line_height(TITLE_FONT_SIZE);
            buffers.push((buffer, x, y, TITLE_FONT_SIZE));
        }
    }

    /// Shape a y-axis label and report the *unrotated* pixel size it will
    /// occupy once drawn into an offscreen texture (width = text length,
    /// height = one line). The caller rotates this 90° when compositing.
    pub fn shape_ylabel(&mut self, text: &str) -> (Buffer, u32, u32) {
        let buffer = self.shape(text, AXIS_LABEL_FONT_SIZE);
        let width = Self::width(&buffer).ceil().max(1.0) as u32;
        let height = line_height(AXIS_LABEL_FONT_SIZE).ceil().max(1.0) as u32;
        (buffer, width, height)
    }

    /// The figure-absolute x position (in pixels) of the rotated y-label
    /// block, given its unrotated height (its *thickness* once rotated).
    /// Mirrors the left-margin formula in `rustyplot_core::Axes2d::margins`,
    /// but measures tick-label widths with the real font instead of that
    /// crate's dependency-free heuristic.
    pub fn ylabel_x(&mut self, axes: &Axes2d, thickness: f32) -> f32 {
        let (_, ylabels) = axes.yticks();
        let max_ylabel_width = ylabels
            .iter()
            .map(|s| Self::width(&self.shape(s, TICK_FONT_SIZE)))
            .fold(0.0f32, f32::max);
        axes.rect.x - chrome::TICK_LENGTH - TICK_LABEL_GAP - max_ylabel_width - TICK_LABEL_GAP - thickness
    }

    /// Render a shaped y-label into a fresh, exactly-sized offscreen texture
    /// (unrotated). Recorded into `encoder` before the caller's main render
    /// pass, so that pass can safely sample the finished texture.
    ///
    /// Uses a separate atlas/renderer from [`Self::render`]; see the note on
    /// `ylabel_atlas` for why they must not share one.
    pub fn render_ylabel(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        buffer: &Buffer,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ylabel offscreen texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.ylabel_viewport.update(queue, Resolution { width, height });

        let area = TextArea {
            buffer,
            left: 0.0,
            top: 0.0,
            scale: 1.0,
            bounds: TextBounds {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            },
            default_color: TEXT_COLOR,
            custom_glyphs: &[],
        };

        let prepared = self
            .ylabel_renderer
            .prepare(
                device,
                queue,
                &mut self.font_system,
                &mut self.ylabel_atlas,
                &self.ylabel_viewport,
                [area],
                &mut self.swash_cache,
            )
            .is_ok();

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ylabel offscreen pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if prepared
                && self
                    .ylabel_renderer
                    .render(&self.ylabel_atlas, &self.ylabel_viewport, &mut pass)
                    .is_err()
            {
                log::warn!("y-label text render failed");
            }
        }

        self.ylabel_atlas.trim();
        (texture, view)
    }


    /// Prepare and draw every queued label. `buffers` holds `(buffer, left, top, font_size)`.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        canvas_width: u32,
        canvas_height: u32,
        buffers: &[(Buffer, f32, f32, f32)],
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        self.viewport.update(
            queue,
            Resolution {
                width: canvas_width,
                height: canvas_height,
            },
        );

        let areas = buffers.iter().map(|(buffer, left, top, _)| TextArea {
            buffer,
            left: *left,
            top: *top,
            scale: 1.0,
            bounds: TextBounds {
                left: 0,
                top: 0,
                right: canvas_width as i32,
                bottom: canvas_height as i32,
            },
            default_color: TEXT_COLOR,
            custom_glyphs: &[],
        });

        if self
            .renderer
            .prepare(
                device,
                queue,
                &mut self.font_system,
                &mut self.atlas,
                &self.viewport,
                areas,
                &mut self.swash_cache,
            )
            .is_err()
        {
            log::warn!("text preparation failed; skipping this frame's labels");
            return;
        }

        if self.renderer.render(&self.atlas, &self.viewport, pass).is_err() {
            log::warn!("text render failed");
        }
        self.atlas.trim();
    }
}
