//! wgpu implementation of [`rustyplot_core::Backend`].
//!
//! Runs unchanged on native (Vulkan/Metal/DX12) and in the browser (WebGPU, or
//! WebGL2 where WebGPU is unavailable).

mod chrome;
mod text;

use bytemuck::{Pod, Zeroable};
use rustyplot_core::{Backend, Rect, Scene, ScatterSeries, View2d, Viewport};
use wgpu::util::DeviceExt;

use chrome::ChromeInstance;
use text::TextPipeline;

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
struct ChromeUniforms {
    viewport: [f32; 2],
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RotateInstance {
    /// Destination (already-rotated) rect in canvas-absolute pixels.
    rect: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PointInstance {
    center: [f32; 2],
    size: f32,
    color: [f32; 4],
}

/// Per-axes GPU state for the scatter pipeline.
///
/// Each axes gets its own uniform buffer (rather than sharing one and
/// rewriting it between draws) because `queue.write_buffer` calls made before
/// a single `queue.submit()` all land before any of that submission's draw
/// calls execute, not interleaved with them; a shared buffer would end up
/// holding only the last axes' transform for every draw in the frame.
struct AxesGpu {
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    instance_buffer: Option<wgpu::Buffer>,
    instance_count: u32,
}

impl AxesGpu {
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> Self {
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("axes uniforms"),
            contents: bytemuck::bytes_of(&Uniforms::new(&View2d::default(), Viewport::default())),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("axes uniform bind group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        Self {
            uniform_buffer,
            bind_group,
            instance_buffer: None,
            instance_count: 0,
        }
    }
}

/// GPU renderer bound to a single surface.
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    scatter_pipeline: wgpu::RenderPipeline,
    scatter_bind_group_layout: wgpu::BindGroupLayout,
    axes_gpu: Vec<AxesGpu>,

    chrome_pipeline: wgpu::RenderPipeline,
    chrome_uniform_buffer: wgpu::Buffer,
    chrome_bind_group: wgpu::BindGroup,
    chrome_instance_buffer: Option<wgpu::Buffer>,
    chrome_instance_capacity: usize,
    chrome_instance_count: u32,

    rotate_pipeline: wgpu::RenderPipeline,
    rotate_texture_bind_group_layout: wgpu::BindGroupLayout,
    rotate_sampler: wgpu::Sampler,

    text: TextPipeline,

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

fn uniform_bind_group_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
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

        let (scatter_pipeline, scatter_bind_group_layout) =
            Self::create_scatter_pipeline(&device, format);
        let (chrome_pipeline, chrome_bind_group_layout) =
            Self::create_chrome_pipeline(&device, format);

        let chrome_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chrome uniforms"),
            contents: bytemuck::bytes_of(&ChromeUniforms {
                viewport: [width as f32, height as f32],
                _pad: [0.0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chrome_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chrome uniform bind group"),
            layout: &chrome_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: chrome_uniform_buffer.as_entire_binding(),
            }],
        });

        let (rotate_pipeline, rotate_texture_bind_group_layout) =
            Self::create_rotate_pipeline(&device, format, &chrome_bind_group_layout);
        let rotate_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ylabel sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let text = TextPipeline::new(&device, &queue, format);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            scatter_pipeline,
            scatter_bind_group_layout,
            axes_gpu: Vec::new(),
            chrome_pipeline,
            chrome_uniform_buffer,
            chrome_bind_group,
            chrome_instance_buffer: None,
            chrome_instance_capacity: 0,
            chrome_instance_count: 0,
            rotate_pipeline,
            rotate_texture_bind_group_layout,
            rotate_sampler,
            text,
            uploaded_revision: None,
        })
    }

    fn create_scatter_pipeline(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scatter shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("scatter.wgsl").into()),
        });

        let bind_group_layout = uniform_bind_group_layout(device, "scatter uniform layout");

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

        (pipeline, bind_group_layout)
    }

    fn create_chrome_pipeline(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("chrome shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("chrome.wgsl").into()),
        });

        let bind_group_layout = uniform_bind_group_layout(device, "chrome uniform layout");

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("chrome pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("chrome pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<ChromeInstance>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x4,
                        1 => Float32x4,
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

        (pipeline, bind_group_layout)
    }

    /// Rotated y-label pipeline. Shares its group(0) (uniform) layout with
    /// the chrome pipeline (`chrome_bind_group_layout`) since `RotateUniforms`
    /// and `ChromeUniforms` are structurally identical, so `chrome_bind_group`
    /// can be bound directly here too; only the texture+sampler group(1) is new.
    fn create_rotate_pipeline(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        uniform_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rotate shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("rotate.wgsl").into()),
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ylabel texture layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rotate pipeline layout"),
            bind_group_layouts: &[Some(uniform_bind_group_layout), Some(&texture_bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rotate pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<RotateInstance>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x4],
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

        (pipeline, texture_bind_group_layout)
    }

    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    pub fn viewport(&self) -> Viewport {
        Viewport::new(self.config.width as f32, self.config.height as f32)
    }

    /// Upload point data for every axes. Call only when the data changes, not
    /// on every frame (pan/zoom only change the view, not the point buffers).
    pub fn upload(&mut self, scene: &Scene, revision: u64) {
        while self.axes_gpu.len() < scene.axes.len() {
            self.axes_gpu
                .push(AxesGpu::new(&self.device, &self.scatter_bind_group_layout));
        }
        self.axes_gpu.truncate(scene.axes.len());

        for (axes, gpu) in scene.axes.iter().zip(self.axes_gpu.iter_mut()) {
            let total: usize = axes.scatter.iter().map(ScatterSeries::len).sum();
            let mut instances: Vec<PointInstance> = Vec::with_capacity(total);
            for s in &axes.scatter {
                for i in 0..s.len() {
                    instances.push(PointInstance {
                        center: [s.x[i], s.y[i]],
                        size: s.size[i],
                        color: s.color[i],
                    });
                }
            }
            gpu.instance_count = instances.len() as u32;
            gpu.instance_buffer = if instances.is_empty() {
                None
            } else {
                Some(
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("scatter instances"),
                            contents: bytemuck::cast_slice(&instances),
                            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        }),
                )
            };
        }

        self.uploaded_revision = Some(revision);
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

    /// Clamp a pixel rect to the surface bounds and round to integers, for
    /// `set_scissor_rect`, which rejects anything hanging outside the target.
    fn scissor_rect(&self, rect: Rect) -> (u32, u32, u32, u32) {
        let cw = self.config.width as f32;
        let ch = self.config.height as f32;
        let x = rect.x.clamp(0.0, cw);
        let y = rect.y.clamp(0.0, ch);
        let w = (rect.x + rect.width).clamp(0.0, cw) - x;
        let h = (rect.y + rect.height).clamp(0.0, ch) - y;
        (
            x.round() as u32,
            y.round() as u32,
            w.round().max(0.0) as u32,
            h.round().max(0.0) as u32,
        )
    }

    fn render(&mut self, scene: &Scene) -> Result<(), RenderError> {
        let frame = match self.acquire() {
            Some(frame) => frame,
            None => {
                self.surface.configure(&self.device, &self.config);
                self.acquire().ok_or(RenderError::SurfaceUnavailable)?
            }
        };

        while self.axes_gpu.len() < scene.axes.len() {
            self.axes_gpu
                .push(AxesGpu::new(&self.device, &self.scatter_bind_group_layout));
        }

        // Per-axes scatter transforms: see `AxesGpu`'s doc comment for why
        // each axes needs its own uniform buffer rather than a shared one.
        for (axes, gpu) in scene.axes.iter().zip(self.axes_gpu.iter()) {
            self.queue.write_buffer(
                &gpu.uniform_buffer,
                0,
                bytemuck::bytes_of(&Uniforms::new(&axes.view, axes.rect.viewport())),
            );
        }

        // Chrome (spines, ticks) for every axes, drawn with one instanced
        // buffer and one draw call: geometry is already in canvas-absolute
        // pixel coordinates, so a single "whole canvas" viewport covers it.
        let mut chrome_instances: Vec<ChromeInstance> = Vec::new();
        for axes in &scene.axes {
            chrome_instances.extend(chrome::build_chrome(axes));
        }
        self.chrome_instance_count = chrome_instances.len() as u32;
        if !chrome_instances.is_empty() {
            let bytes: &[u8] = bytemuck::cast_slice(&chrome_instances);
            match &self.chrome_instance_buffer {
                Some(buffer) if self.chrome_instance_capacity >= chrome_instances.len() => {
                    self.queue.write_buffer(buffer, 0, bytes);
                }
                _ => {
                    self.chrome_instance_buffer = Some(self.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("chrome instances"),
                            contents: bytes,
                            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        },
                    ));
                    self.chrome_instance_capacity = chrome_instances.len();
                }
            }
        }
        self.queue.write_buffer(
            &self.chrome_uniform_buffer,
            0,
            bytemuck::bytes_of(&ChromeUniforms {
                viewport: [self.config.width as f32, self.config.height as f32],
                _pad: [0.0; 2],
            }),
        );

        // Text labels: shaped fresh every frame. Tick positions and their
        // labels change continuously during pan/zoom, and the label count
        // is tiny, so there is no revision-gated caching here.
        let mut text_buffers: Vec<(glyphon::Buffer, f32, f32, f32)> = Vec::new();
        for axes in &scene.axes {
            self.text.queue_axes(axes, &mut text_buffers);
        }

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Rotated y-axis labels: each is shaped and rendered into its own
        // offscreen texture. `ylabel_viewport`/`ylabel_renderer` are shared
        // across axes (see the note on `ylabel_atlas` in `text.rs`), so each
        // axes' `prepare`+render pass is recorded into its *own* command
        // encoder and submitted immediately -- before the next axes'
        // `prepare` call overwrites the same buffers. Deferring all of these
        // to the single end-of-frame `submit()` (as the main and chrome
        // passes are) would reproduce exactly the ordering hazard that gave
        // `AxesGpu` its own uniform buffer: every `queue.write_buffer` call
        // made before a submission lands before any of that submission's
        // render passes execute, so a second axes' `prepare()` would corrupt
        // the first axes' pass before it ever ran.
        struct YLabelDraw {
            bind_group: wgpu::BindGroup,
            instance_buffer: wgpu::Buffer,
        }
        let mut ylabel_draws: Vec<YLabelDraw> = Vec::new();
        let mut ylabel_textures: Vec<wgpu::Texture> = Vec::new();
        for axes in &scene.axes {
            if axes.ylabel.is_empty() {
                continue;
            }
            let (buffer, w, h) = self.text.shape_ylabel(&axes.ylabel);
            let mut ylabel_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("ylabel"),
                    });
            let (texture, texture_view) = self.text.render_ylabel(
                &self.device,
                &self.queue,
                &mut ylabel_encoder,
                &buffer,
                w,
                h,
                self.config.format,
            );
            self.queue.submit(Some(ylabel_encoder.finish()));
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ylabel texture bind group"),
                layout: &self.rotate_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.rotate_sampler),
                    },
                ],
            });
            // Rotating 90° swaps width and height for the destination rect.
            let dest_width = h as f32;
            let dest_height = w as f32;
            let x = self.text.ylabel_x(axes, dest_width);
            let y = axes.rect.y + axes.rect.height / 2.0 - dest_height / 2.0;
            let instance_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ylabel rotate instance"),
                contents: bytemuck::bytes_of(&RotateInstance {
                    rect: [x, y, dest_width, dest_height],
                }),
                usage: wgpu::BufferUsages::VERTEX,
            });
            ylabel_draws.push(YLabelDraw {
                bind_group,
                instance_buffer,
            });
            ylabel_textures.push(texture);
        }
        let _ylabel_textures = ylabel_textures;

        // The main frame pass is recorded into its own encoder, submitted
        // once at the end of `render` -- separately from the y-label
        // passes above, which are already complete (and their textures
        // fully written) by the time this pass samples them.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        {
            let bg = scene.background;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame pass"),
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

            // Scatter data, clipped to each axes' own plot area so points
            // outside the current view never bleed into a neighbouring axes
            // or its margins.
            pass.set_pipeline(&self.scatter_pipeline);
            for (axes, gpu) in scene.axes.iter().zip(self.axes_gpu.iter()) {
                let Some(buffer) = &gpu.instance_buffer else {
                    continue;
                };
                if gpu.instance_count == 0 {
                    continue;
                }
                let (x, y, w, h) = self.scissor_rect(axes.rect);
                if w == 0 || h == 0 {
                    continue;
                }
                pass.set_viewport(
                    axes.rect.x,
                    axes.rect.y,
                    axes.rect.width,
                    axes.rect.height,
                    0.0,
                    1.0,
                );
                pass.set_scissor_rect(x, y, w, h);
                pass.set_bind_group(0, &gpu.bind_group, &[]);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..4, 0..gpu.instance_count);
            }

            // Chrome and text both use canvas-absolute coordinates, so the
            // viewport/scissor set per axes above must be undone first.
            pass.set_viewport(
                0.0,
                0.0,
                self.config.width as f32,
                self.config.height as f32,
                0.0,
                1.0,
            );
            pass.set_scissor_rect(0, 0, self.config.width, self.config.height);

            if let Some(buffer) = &self.chrome_instance_buffer
                && self.chrome_instance_count > 0
            {
                pass.set_pipeline(&self.chrome_pipeline);
                pass.set_bind_group(0, &self.chrome_bind_group, &[]);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..4, 0..self.chrome_instance_count);
            }

            if !ylabel_draws.is_empty() {
                pass.set_pipeline(&self.rotate_pipeline);
                // Same uniform layout as chrome (see `create_rotate_pipeline`).
                pass.set_bind_group(0, &self.chrome_bind_group, &[]);
                for draw in &ylabel_draws {
                    pass.set_bind_group(1, &draw.bind_group, &[]);
                    pass.set_vertex_buffer(0, draw.instance_buffer.slice(..));
                    pass.draw(0..4, 0..1);
                }
            }

            self.text.render(
                &self.device,
                &self.queue,
                self.config.width,
                self.config.height,
                &text_buffers,
                &mut pass,
            );
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
