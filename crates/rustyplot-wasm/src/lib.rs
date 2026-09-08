//! Browser frontend: a thin JavaScript-facing shell around the shared core.
//!
//! JavaScript owns only the DOM event plumbing; every transform, hit test and
//! draw call happens here, so notebook interaction never round-trips to Python.

#![cfg(target_arch = "wasm32")]

use rustyplot_core::{Backend, Interaction, Scene, ScatterSeries, Viewport};
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

#[wasm_bindgen]
impl Plot {
    /// Create a plot bound to a canvas. Async because adapter selection is async on the web.
    ///
    /// `use_webgpu` must reflect what the page has already established is available:
    /// attaching binds the canvas to one context type permanently, so there is no
    /// second chance to try the other backend.
    pub async fn attach(canvas: HtmlCanvasElement, use_webgpu: bool) -> Result<Plot, JsError> {
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

        let viewport = Viewport::new(width as f32, height as f32);
        Ok(Self {
            renderer,
            scene: Scene::new(),
            interaction: Interaction::new(viewport),
            revision: 0,
        })
    }

    /// Replace the scatter data. `color` is a flat RGBA array of length `4 * n`.
    pub fn set_scatter(
        &mut self,
        x: &[f32],
        y: &[f32],
        size: &[f32],
        color: &[f32],
    ) -> Result<(), JsError> {
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

        self.scene.scatter = vec![series];
        self.revision += 1;
        self.renderer.upload(&self.scene.scatter, self.revision);
        Ok(())
    }

    pub fn set_background(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.scene.background = [r, g, b, a];
    }

    pub fn autoscale(&mut self) {
        self.scene.autoscale(0.05);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(width, height);
        self.interaction.viewport = self.renderer.viewport();
    }

    pub fn pointer_down(&mut self, px: f32, py: f32) {
        self.interaction.pointer_down(px, py);
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

    /// Hit test, returning `[series, index, x, y, distance_px]` or `undefined`.
    pub fn pick(&self, px: f32, py: f32) -> Option<Box<[f64]>> {
        self.interaction.pick(px, py, &self.scene).map(|h| {
            Box::new([
                h.series as f64,
                h.index as f64,
                h.x as f64,
                h.y as f64,
                h.distance_px as f64,
            ]) as Box<[f64]>
        })
    }

    /// Data coordinates under a pixel, as `[x, y]`.
    pub fn data_at(&self, px: f32, py: f32) -> Box<[f64]> {
        let (x, y) = self
            .scene
            .view
            .screen_to_data(px, py, self.interaction.viewport);
        Box::new([x as f64, y as f64])
    }

    /// Current view as `[x_min, x_max, y_min, y_max]`.
    pub fn view(&self) -> Box<[f64]> {
        let v = &self.scene.view;
        Box::new([
            v.x_min as f64,
            v.x_max as f64,
            v.y_min as f64,
            v.y_max as f64,
        ])
    }

    pub fn set_view(&mut self, x_min: f32, x_max: f32, y_min: f32, y_max: f32) {
        self.scene.view = rustyplot_core::View2d::new(x_min, x_max, y_min, y_max);
    }

    pub fn point_count(&self) -> u32 {
        self.scene.point_count() as u32
    }

    pub fn draw(&mut self) -> Result<(), JsError> {
        self.renderer
            .draw(&self.scene)
            .map_err(|e| JsError::new(&e.to_string()))
    }
}
