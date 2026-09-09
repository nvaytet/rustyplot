//! Browser frontend: a thin JavaScript-facing shell around the shared core.
//!
//! JavaScript owns only the DOM event plumbing; every transform, hit test and
//! draw call happens here, so notebook interaction never round-trips to Python.

#![cfg(target_arch = "wasm32")]

use rustyplot_core::{
    Backend, Interaction, Line, LineSeries, LineStyle, Marker, MarkerStyle, Scene, ScatterSeries,
    View2d, Viewport,
};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Warn);
}

#[wasm_bindgen]
pub struct Plot {
    renderer: rustyplot_render::Renderer,
    scene: Scene,
    interaction: Interaction,
    revision: u64,
}

fn axes_index(scene: &Scene, axes: usize) -> Result<usize, JsError> {
    if axes >= scene.axes.len() {
        return Err(JsError::new(&format!(
            "axes index {axes} out of range (figure has {})",
            scene.axes.len()
        )));
    }
    Ok(axes)
}

#[wasm_bindgen]
impl Plot {
    /// Create a figure with an `nrows x ncols` grid of axes, bound to a canvas.
    /// Async because adapter selection is async on the web.
    ///
    /// `use_webgpu` must reflect what the page has already established is available:
    /// attaching binds the canvas to one context type permanently, so there is no
    /// second chance to try the other backend.
    pub async fn attach(
        canvas: HtmlCanvasElement,
        use_webgpu: bool,
        nrows: usize,
        ncols: usize,
    ) -> Result<Plot, JsError> {
        let width = canvas.width().max(1);
        let height = canvas.height().max(1);

        let backends = if use_webgpu {
            wgpu::Backends::BROWSER_WEBGPU
        } else {
            wgpu::Backends::GL
        };
        let instance = rustyplot_render::create_instance(backends);
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| {
                JsError::new(&format!(
                    "could not create a {} surface: {e}",
                    if use_webgpu { "WebGPU" } else { "WebGL2" }
                ))
            })?;
        let renderer = rustyplot_render::Renderer::new(&instance, surface, width, height)
            .await
            .map_err(|e| {
                JsError::new(&format!(
                    "{e} (requested backend: {})",
                    if use_webgpu { "WebGPU" } else { "WebGL2" }
                ))
            })?;

        let mut scene = Scene::grid(nrows, ncols);
        scene.layout(Viewport::new(width as f32, height as f32));

        Ok(Self {
            renderer,
            scene,
            interaction: Interaction::new(),
            revision: 0,
        })
    }

    /// Replace one axes' scatter data. `color` is a flat RGBA array of length `4 * n`.
    pub fn set_scatter(
        &mut self,
        axes: usize,
        x: &[f32],
        y: &[f32],
        size: &[f32],
        color: &[f32],
    ) -> Result<(), JsError> {
        let idx = axes_index(&self.scene, axes)?;
        let n = x.len();
        if y.len() != n || size.len() != n || color.len() != 4 * n {
            return Err(JsError::new(&format!(
                "inconsistent array lengths: x={n}, y={}, size={}, color={} (expected {})",
                y.len(),
                size.len(),
                color.len(),
                4 * n
            )));
        }

        let series = ScatterSeries {
            x: x.to_vec(),
            y: y.to_vec(),
            size: size.to_vec(),
            color: color.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect(),
        };
        series.validate().map_err(|e| JsError::new(&e.to_string()))?;

        self.scene.axes[idx].scatter = vec![series];
        self.revision += 1;
        self.renderer.upload(&self.scene, self.revision);
        Ok(())
    }

    /// Remove every line artist from one axes (scatter is unaffected).
    /// There is no per-line removal API in v1: `plot()` calls only append,
    /// so a full rebuild starts by clearing everything.
    pub fn clear_lines(&mut self, axes: usize) -> Result<(), JsError> {
        let idx = axes_index(&self.scene, axes)?;
        self.scene.axes[idx].lines.clear();
        self.revision += 1;
        self.renderer.upload(&self.scene, self.revision);
        Ok(())
    }

    /// Append one line artist to an axes. `color` is a flat RGBA array of
    /// length 4. `line_style` is `""` for no line, `"solid"` or `"dashed"`.
    /// `marker` is `""` for no marker or `"o"` for a circle. At least one of
    /// `line_style`/`marker` should be non-empty for anything to be visible.
    #[allow(clippy::too_many_arguments)]
    pub fn add_line(
        &mut self,
        axes: usize,
        x: &[f32],
        y: &[f32],
        color: &[f32],
        width: f32,
        line_style: &str,
        marker: &str,
        marker_size: f32,
    ) -> Result<(), JsError> {
        let idx = axes_index(&self.scene, axes)?;
        let n = x.len();
        if y.len() != n {
            return Err(JsError::new(&format!(
                "inconsistent array lengths: x={n}, y={} (expected equal lengths)",
                y.len()
            )));
        }
        if color.len() != 4 {
            return Err(JsError::new(&format!(
                "`color` has {} elements but a line takes a single RGBA colour (4 elements)",
                color.len()
            )));
        }
        let line = match line_style {
            "" => None,
            "solid" => Some(Line {
                width,
                style: LineStyle::Solid,
            }),
            "dashed" => Some(Line {
                width,
                style: LineStyle::Dashed,
            }),
            other => {
                return Err(JsError::new(&format!(
                    "unsupported line style '{other}'; supported styles are: \"solid\", \"dashed\", or \"\" for no line"
                )));
            }
        };
        let marker = match marker {
            "" => None,
            "o" => Some(Marker {
                style: MarkerStyle::Circle,
                size: marker_size,
            }),
            other => {
                return Err(JsError::new(&format!(
                    "unsupported marker style '{other}'; supported styles are: \"o\" (circle), or \"\" for no marker"
                )));
            }
        };

        let series = LineSeries {
            x: x.to_vec(),
            y: y.to_vec(),
            color: [color[0], color[1], color[2], color[3]],
            line,
            marker,
        };
        series.validate().map_err(|e| JsError::new(&e.to_string()))?;

        self.scene.axes[idx].lines.push(series);
        self.revision += 1;
        self.renderer.upload(&self.scene, self.revision);
        Ok(())
    }

    pub fn set_title(&mut self, axes: usize, text: String) -> Result<(), JsError> {
        let idx = axes_index(&self.scene, axes)?;
        self.scene.axes[idx].title = text;
        Ok(())
    }

    pub fn set_xlabel(&mut self, axes: usize, text: String) -> Result<(), JsError> {
        let idx = axes_index(&self.scene, axes)?;
        self.scene.axes[idx].xlabel = text;
        Ok(())
    }

    pub fn set_ylabel(&mut self, axes: usize, text: String) -> Result<(), JsError> {
        let idx = axes_index(&self.scene, axes)?;
        self.scene.axes[idx].ylabel = text;
        Ok(())
    }

    pub fn set_background(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.scene.background = [r, g, b, a];
    }

    /// Fits the view to the current data and re-uploads.
    ///
    /// The re-upload matters for lines specifically: dash phase is baked
    /// into the instance buffer at upload time as a *screen-space* arc
    /// length (see `Renderer::upload`), computed from the axes' current
    /// view. `add_line`/`clear_lines` upload immediately, before the
    /// frontend calls `autoscale` to fit the view to the new data, so that
    /// baked arc length is wrong -- distances measured against the old
    /// (often default, unrelated-to-the-data) view -- and dash/gap spacing
    /// comes out wildly off. Re-uploading here, after the view is fitted,
    /// bakes the arc length against the view that will actually be
    /// rendered with. Scatter points have no such view-dependent baked
    /// state, so this only matters for lines, but re-uploading unconditionally
    /// is simplest and cheap enough not to bother skipping it otherwise.
    pub fn autoscale(&mut self, axes: usize) -> Result<(), JsError> {
        let idx = axes_index(&self.scene, axes)?;
        self.scene.axes[idx].autoscale(0.05);
        self.renderer.upload(&self.scene, self.revision);
        Ok(())
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(width, height);
    }

    pub fn pointer_down(&mut self, px: f32, py: f32) {
        self.interaction.pointer_down(px, py, &self.scene);
    }

    /// Returns `true` when the view moved and the caller should redraw.
    pub fn pointer_move(&mut self, px: f32, py: f32) -> bool {
        self.interaction.pointer_move(px, py, &mut self.scene)
    }

    /// Returns `true` if the gesture was a click rather than a drag.
    pub fn pointer_up(&mut self) -> bool {
        self.interaction.pointer_up()
    }

    pub fn wheel(&mut self, px: f32, py: f32, delta: f32) {
        self.interaction.wheel(px, py, delta, &mut self.scene);
    }

    /// Hit test, returning `[axes, series, index, x, y, distance_px]` or `undefined`.
    pub fn pick(&self, px: f32, py: f32) -> Option<Box<[f64]>> {
        self.interaction.pick(px, py, &self.scene).map(|h| {
            Box::new([
                h.axes as f64,
                h.series as f64,
                h.index as f64,
                h.x as f64,
                h.y as f64,
                h.distance_px as f64,
            ]) as Box<[f64]>
        })
    }

    /// The containing axes and data coordinates under a pixel, as
    /// `[axes, x, y]`, or `undefined` if the pixel is outside every axes'
    /// plot area. The axes index is included (not just `[x, y]`) so a
    /// caller can identify which axes was clicked even when the click
    /// misses every point -- `pick` alone cannot answer that, since it
    /// returns `undefined` whenever no point is within its hit radius.
    pub fn data_at(&self, px: f32, py: f32) -> Option<Box<[f64]>> {
        let idx = self.scene.axes_at(px, py)?;
        let axes = &self.scene.axes[idx];
        let (lx, ly) = (px - axes.rect.x, py - axes.rect.y);
        let (x, y) = axes.view.screen_to_data(lx, ly, axes.rect.viewport());
        Some(Box::new([idx as f64, x as f64, y as f64]))
    }

    /// One axes' current view as `[x_min, x_max, y_min, y_max]`.
    pub fn view(&self, axes: usize) -> Result<Box<[f64]>, JsError> {
        let idx = axes_index(&self.scene, axes)?;
        let v = self.scene.axes[idx].view;
        Ok(Box::new([v.x_min as f64, v.x_max as f64, v.y_min as f64, v.y_max as f64]))
    }

    pub fn set_view(
        &mut self,
        axes: usize,
        x_min: f32,
        x_max: f32,
        y_min: f32,
        y_max: f32,
    ) -> Result<(), JsError> {
        let idx = axes_index(&self.scene, axes)?;
        self.scene.axes[idx].view = View2d::new(x_min, x_max, y_min, y_max);
        Ok(())
    }

    pub fn point_count(&self) -> u32 {
        self.scene.point_count() as u32
    }

    pub fn draw(&mut self) -> Result<(), JsError> {
        self.scene.layout(self.renderer.viewport());
        // Layout can change axes rects (canvas resize, e.g. the very first
        // resize as a notebook's CSS sizing settles after `attach`), and
        // dash phase is baked as *screen-space* arc length -- a function of
        // the rect, same as zoom. Rebaking here, unconditionally, covers
        // every case that can invalidate it (resize, zoom, pan does not
        // since it's a pure translation but rebaking anyway is cheap) in
        // one place, since `draw` itself only runs on discrete UI events,
        // not a continuous per-frame loop -- see `Renderer::rebake_dash_phase`.
        self.renderer.rebake_dash_phase(&self.scene);
        self.renderer
            .draw(&self.scene)
            .map_err(|e| JsError::new(&e.to_string()))
    }
}
