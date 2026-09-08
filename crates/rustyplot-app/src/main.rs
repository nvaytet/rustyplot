//! Native demo window. Shares every line of plotting logic with the browser build,
//! and exists mainly so shaders and interaction can be iterated without a notebook.

use std::sync::Arc;

use rustyplot_core::{Backend, Interaction, Scene, ScatterSeries, Viewport};
use rustyplot_render::Renderer;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// Deterministic xorshift, so runs are comparable and we avoid a `rand` dependency.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1 << 24) as f32
    }
}

fn demo_series(n: usize) -> ScatterSeries {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut series = ScatterSeries {
        x: Vec::with_capacity(n),
        y: Vec::with_capacity(n),
        size: Vec::with_capacity(n),
        color: Vec::with_capacity(n),
    };
    for _ in 0..n {
        // Two correlated gaussian-ish blobs, via sums of uniforms.
        let u = (0..4).map(|_| rng.next_f32()).sum::<f32>() / 4.0;
        let v = (0..4).map(|_| rng.next_f32()).sum::<f32>() / 4.0;
        let x = (u - 0.5) * 4.0;
        let y = (v - 0.5) * 4.0 + 0.6 * x;
        series.x.push(x);
        series.y.push(y);
        series.size.push(3.0 + 5.0 * rng.next_f32());
        series
            .color
            .push([0.15 + 0.5 * u, 0.35, 0.85 - 0.4 * v, 0.55]);
    }
    series
}

struct State {
    window: Arc<Window>,
    renderer: Renderer,
    scene: Scene,
    interaction: Interaction,
    cursor: (f32, f32),
}

#[derive(Default)]
struct App {
    state: Option<State>,
    points: usize,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("rustyplot")
            .with_inner_size(winit::dpi::LogicalSize::new(1000.0, 750.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let size = window.inner_size();

        let instance = rustyplot_render::create_instance(
            wgpu::Backends::from_env().unwrap_or_else(wgpu::Backends::all),
        );
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let mut renderer = pollster::block_on(Renderer::new(
            &instance,
            surface,
            size.width,
            size.height,
        ))
        .expect("create renderer");

        let mut scene = Scene::new();
        let started = std::time::Instant::now();
        scene.primary_mut().scatter.push(demo_series(self.points));
        log::info!(
            "generated {} points in {:?}",
            self.points,
            started.elapsed()
        );
        scene.autoscale(0.05);
        scene.layout(Viewport::new(size.width as f32, size.height as f32));

        let started = std::time::Instant::now();
        renderer.upload(&scene, 1);
        log::info!("uploaded to GPU in {:?}", started.elapsed());

        let interaction = Interaction::new();
        self.state = Some(State {
            window,
            renderer,
            scene,
            interaction,
            cursor: (0.0, 0.0),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                state.renderer.resize(size.width, size.height);
                state.window.request_redraw();
            }

            WindowEvent::CursorMoved { position, .. } => {
                state.cursor = (position.x as f32, position.y as f32);
                if state
                    .interaction
                    .pointer_move(state.cursor.0, state.cursor.1, &mut state.scene)
                {
                    state.window.request_redraw();
                }
            }

            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state: element_state,
                ..
            } => match element_state {
                ElementState::Pressed => state.interaction.pointer_down(
                    state.cursor.0,
                    state.cursor.1,
                    &state.scene,
                ),
                ElementState::Released => {
                    if state.interaction.pointer_up()
                        && let Some(hit) =
                            state
                                .interaction
                                .pick(state.cursor.0, state.cursor.1, &state.scene)
                    {
                        println!("picked point {} at ({}, {})", hit.index, hit.x, hit.y);
                    }
                }
            },

            WindowEvent::MouseWheel { delta, .. } => {
                // Trackpads report pixels, wheels report lines; normalise to browser-ish units.
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => -y * 60.0,
                    MouseScrollDelta::PixelDelta(p) => -p.y as f32,
                };
                state
                    .interaction
                    .wheel(state.cursor.0, state.cursor.1, dy, &mut state.scene);
                state.window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                let started = std::time::Instant::now();
                state.scene.layout(state.renderer.viewport());
                if let Err(e) = state.renderer.draw(&state.scene) {
                    log::error!("draw failed: {e}");
                }
                log::debug!("frame in {:?}", started.elapsed());
            }

            _ => {}
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let points = std::env::var("RUSTYPLOT_POINTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000_000);

    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App {
        state: None,
        points,
    };
    event_loop.run_app(&mut app).expect("run app");
}
