//! wgpu implementation of [`rustyplot_core::Backend`].
//!
//! Runs unchanged on native (Vulkan/Metal/DX12) and in the browser (WebGPU, or
//! WebGL2 where WebGPU is unavailable).

use bytemuck::{Pod, Zeroable};
use rustyplot_core::{Backend, Scene, ScatterSeries, View2d, Viewport};
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub enum RenderError {
    NoAdapter,
    NoDevice(wgpu::RequestDeviceError),
    SurfaceUnavailable,
    CreateSurface(wgpu::CreateSurfaceError),
}

impl core::fmt::Display for RenderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoAdapter => write!(
                f,
                "no suitable GPU adapter found (on the web this means neither WebGPU nor WebGL2 is available)"
            ),
            Self::NoDevice(e) => write!(f, "could not create a GPU device: {e}"),
            Self::SurfaceUnavailable => {
                write!(f, "could not acquire a surface texture, even after reconfiguring")
            }
            Self::CreateSurface(e) => write!(f, "could not create a surface: {e}"),
        }
    }
}

impl core::error::Error for RenderError {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    scale: [f32; 2],
    offset: [f32; 2],
    viewport: [f32; 2],
    _pad: [f32; 2],
}

impl Uniforms {
    fn new(view: &View2d, vp: Viewport) -> Self {
        Self {
            scale: view.scale(),
            offset: view.offset(),
            viewport: [vp.width, vp.height],
            _pad: [0.0; 2],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PointInstance {
    center: [f32; 2],
    size: f32,
    color: [f32; 4],
}

/// GPU renderer bound to a single surface.
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    instance_buffer: Option<wgpu::Buffer>,
    instance_capacity: usize,
    instance_count: u32,
    /// Bumped whenever scene data changes, to avoid re-uploading on pan/zoom.
    uploaded_revision: Option<u64>,
}

/// Create an instance restricted to `backends`.
///
/// Restricting matters on the web: probing a backend binds the canvas to that
/// context type for good, so a failed WebGPU probe would block the WebGL2
/// fallback. Callers decide up front which one to use.
pub fn create_instance(backends: wgpu::Backends) -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    })
}

impl Renderer {
    pub async fn new(
        instance: &wgpu::Instance,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> Result<Self, RenderError> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|_| RenderError::NoAdapter)?;

        log::info!("rustyplot adapter: {:?}", adapter.get_info());

        // WebGL2 supports far less than WebGPU, so ask only for what the shader needs.
        let limits = if cfg!(target_arch = "wasm32") {
            wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits())
        } else {
            wgpu::Limits::default()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("rustyplot device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(RenderError::NoDevice)?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::default(),
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scatter shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("scatter.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&Uniforms::new(&View2d::default(), Viewport::default())),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("uniform layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scatter pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scatter pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<PointInstance>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2,
                        1 => Float32,
                        2 => Float32x4,
                    ],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            uniform_buffer,
            bind_group,
            instance_buffer: None,
            instance_capacity: 0,
            instance_count: 0,
            uploaded_revision: None,
        })
    }

    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    pub fn viewport(&self) -> Viewport {
        Viewport::new(self.config.width as f32, self.config.height as f32)
    }

    /// Upload point data. Call only when the data changes, not on every frame.
    pub fn upload(&mut self, series: &[ScatterSeries], revision: u64) {
        let total: usize = series.iter().map(ScatterSeries::len).sum();
        let mut instances: Vec<PointInstance> = Vec::with_capacity(total);
        for s in series {
            for i in 0..s.len() {
                instances.push(PointInstance {
                    center: [s.x[i], s.y[i]],
                    size: s.size[i],
                    color: s.color[i],
                });
            }
        }
        self.instance_count = instances.len() as u32;
        self.uploaded_revision = Some(revision);

        if instances.is_empty() {
            return;
        }

        let bytes: &[u8] = bytemuck::cast_slice(&instances);
        match &self.instance_buffer {
            Some(buffer) if self.instance_capacity >= instances.len() => {
                self.queue.write_buffer(buffer, 0, bytes);
            }
            _ => {
                self.instance_buffer = Some(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("scatter instances"),
                        contents: bytes,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    },
                ));
                self.instance_capacity = instances.len();
            }
        }
    }

    pub fn uploaded_revision(&self) -> Option<u64> {
        self.uploaded_revision
    }

    /// Acquire a frame, treating a suboptimal one as good enough to draw into.
    fn acquire(&self) -> Option<wgpu::SurfaceTexture> {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => Some(frame),
            _ => None,
        }
    }

    fn render(&mut self, scene: &Scene) -> Result<(), RenderError> {
        let frame = match self.acquire() {
            Some(frame) => frame,
            None => {
                self.surface.configure(&self.device, &self.config);
                self.acquire().ok_or(RenderError::SurfaceUnavailable)?
            }
        };

        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&Uniforms::new(&scene.view, self.viewport())),
        );

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        {
            let bg = scene.background;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scatter pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: bg[0] as f64,
                            g: bg[1] as f64,
                            b: bg[2] as f64,
                            a: bg[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            if let Some(buffer) = &self.instance_buffer
                && self.instance_count > 0
            {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..4, 0..self.instance_count);
            }
        }

        self.queue.submit(Some(encoder.finish()));
        self.queue.present(frame);
        Ok(())
    }
}

impl Backend for Renderer {
    type Error = RenderError;

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if width == self.config.width && height == self.config.height {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    fn draw(&mut self, scene: &Scene) -> Result<(), Self::Error> {
        self.render(scene)
    }
}
