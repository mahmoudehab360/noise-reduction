use crate::params::Params;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub const MAXSLOTS: usize = 7; // 3 frames on each side + the current one

const KERNELS: [&str; 8] = [
    "k_cur",
    "k_down4",
    "k_me_coarse",
    "k_me_refine",
    "k_temporal",
    "k_nlm",
    "k_chroma",
    "k_final",
];

/// Byte offset of `row0` inside `U` (22nd 4-byte field).
const ROW0_OFFSET: u64 = 21 * 4;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct U {
    w: u32,
    h: u32,
    nslots: u32,
    cur: u32,
    w4: u32,
    h4: u32,
    cbw: u32,
    cbh: u32,
    fbw: u32,
    fbh: u32,
    range4: i32,
    refine_r: i32,
    spatial_r: i32,
    patch_r: i32,
    stride: i32,
    chroma_r: i32,
    cstride: i32,
    me_on: u32,
    spatial_on: u32,
    frame_idx: u32,
    grain_size: u32,
    row0: u32,
    sig_t_l: f32,
    sig_t_c: f32,
    mot_scale: f32,
    t_blend: f32,
    sig_s_l: f32,
    sig_s_c: f32,
    s_blend: f32,
    detail: f32,
    sharpen: f32,
    grain: f32,
}

fn build_uniforms(w: u32, h: u32, ns: u32, cur: u32, p: &Params, frame_idx: u32) -> U {
    let w4 = (w + 3) / 4;
    let h4 = (h + 3) / 4;
    let (patch_r, stride): (i32, i32) = match p.s_mode {
        0 => (1, 2),
        1 => (1, 1),
        _ => (2, 1),
    };
    let spatial_r: i32 = match p.s_radius {
        0 => 3,
        1 => 5,
        _ => 7,
    };
    let chroma_r: i32 = match p.s_radius {
        0 => 4,
        1 => 7,
        _ => 10,
    };
    let k = 0.0012f32; // threshold unit -> noise sigma (0..1 range)
    U {
        w,
        h,
        nslots: ns,
        cur,
        w4,
        h4,
        cbw: (w4 + 7) / 8,
        cbh: (h4 + 7) / 8,
        fbw: (w + 15) / 16,
        fbh: (h + 15) / 16,
        range4: match p.t_range {
            0 => 2,
            1 => 4,
            _ => 8,
        },
        refine_r: if p.t_me == 1 { 2 } else { 3 },
        spatial_r,
        patch_r,
        stride,
        chroma_r,
        cstride: if chroma_r > 7 { 2 } else { 1 },
        me_on: (p.t_me != 0) as u32,
        spatial_on: p.s_on as u32,
        frame_idx,
        grain_size: p.grain_size.clamp(1, 4),
        row0: 0,
        sig_t_l: p.t_luma * k,
        sig_t_c: p.t_chroma * k,
        mot_scale: 0.002 + p.t_motion * 0.0008,
        t_blend: p.t_blend / 100.0,
        sig_s_l: p.s_luma * k,
        sig_s_c: p.s_chroma * k,
        s_blend: p.s_blend / 100.0,
        detail: p.detail / 100.0,
        sharpen: p.sharpen / 100.0 * 1.5,
        grain: p.grain / 100.0 * 0.05,
    }
}

// ---------------------------------------------------------------------------
// Adapter discovery: prefer the NVIDIA card
// ---------------------------------------------------------------------------

pub fn adapter_label(a: &wgpu::Adapter) -> String {
    let i = a.get_info();
    format!("{} ({:?})", i.name, i.backend)
}

fn score(i: &wgpu::AdapterInfo) -> i32 {
    let mut s = 0;
    if i.vendor == 0x10DE {
        s += 100; // NVIDIA
    }
    match i.device_type {
        wgpu::DeviceType::DiscreteGpu => s += 50,
        wgpu::DeviceType::IntegratedGpu => s += 10,
        wgpu::DeviceType::VirtualGpu => s += 5,
        _ => {}
    }
    if i.backend == wgpu::Backend::Vulkan {
        s += 5;
    }
    s
}

pub fn list_adapters() -> (wgpu::Instance, Vec<wgpu::Adapter>) {
    let backends = wgpu::Backends::VULKAN | wgpu::Backends::DX12;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        ..Default::default()
    });
    let mut v: Vec<wgpu::Adapter> = instance.enumerate_adapters(backends).into_iter().collect();
    v.retain(|a| a.get_info().device_type != wgpu::DeviceType::Cpu);
    v.sort_by_key(|a| -score(&a.get_info()));
    (instance, v)
}

// ---------------------------------------------------------------------------
// Denoiser
// ---------------------------------------------------------------------------

struct Res {
    w: u32,
    h: u32,
    frames: wgpu::Buffer,
    out_px: wgpu::Buffer,
    staging: wgpu::Buffer,
    bg: Vec<wgpu::BindGroup>,
    // kept alive for the bind groups
    _keep: Vec<wgpu::Buffer>,
}

pub struct Denoiser {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipes: Vec<wgpu::ComputePipeline>,
    uni: wgpu::Buffer,
    res: Option<Res>,
    err: Arc<Mutex<Option<String>>>,
}

fn make_bg(device: &wgpu::Device, pipe: &wgpu::ComputePipeline, entries: &[(u32, &wgpu::Buffer)]) -> wgpu::BindGroup {
    let layout = pipe.get_bind_group_layout(0);
    let es: Vec<wgpu::BindGroupEntry> = entries
        .iter()
        .map(|(b, buf)| wgpu::BindGroupEntry {
            binding: *b,
            resource: buf.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &es,
    })
}

impl Denoiser {
    pub fn new(adapter: &wgpu::Adapter) -> Result<Self, String> {
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("lowlight-dn"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
            },
            None,
        ))
        .map_err(|e| format!("GPU initialisation failed: {e}"))?;

        let err: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        {
            let e2 = err.clone();
            device.on_uncaptured_error(Box::new(move |e| {
                *e2.lock().unwrap() = Some(e.to_string());
            }));
        }

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lowlight-dn"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });
        let pipes: Vec<wgpu::ComputePipeline> = KERNELS
            .iter()
            .map(|k| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(*k),
                    layout: None,
                    module: &module,
                    entry_point: *k,
                })
            })
            .collect();

        let uni = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<U>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        if let Some(e) = err.lock().unwrap().take() {
            return Err(format!("Shader / pipeline error: {e}"));
        }
        Ok(Self {
            device,
            queue,
            pipes,
            uni,
            res: None,
            err,
        })
    }

    fn ensure(&mut self, w: u32, h: u32) {
        if let Some(r) = &self.res {
            if r.w == w && r.h == h {
                return;
            }
        }
        self.res = None; // free the old buffers first
        let device = &self.device;
        let n = w as u64 * h as u64;
        let w4 = (w + 3) / 4;
        let h4 = (h + 3) / 4;
        let n4 = w4 as u64 * h4 as u64;
        let cb = (((w4 + 7) / 8) as u64) * (((h4 + 7) / 8) as u64);
        let fb = (((w + 15) / 16) as u64) * (((h + 15) / 16) as u64);
        let k = MAXSLOTS as u64;

        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage,
                mapped_at_creation: false,
            })
        };
        let st = wgpu::BufferUsages::STORAGE;
        let frames = mk("frames", k * n * 4, st | wgpu::BufferUsages::COPY_DST);
        let cur_ycc = mk("cur_ycc", n * 16, st);
        let luma4 = mk("luma4", k * n4 * 4, st);
        let mvc = mk("mvc", k * cb * 2 * 4, st);
        let mv = mk("mv", k * fb * 2 * 4, st);
        let sad = mk("sad", k * fb * 4, st);
        let ycc_t = mk("ycc_t", n * 16, st);
        let yb_t = mk("yb_t", n * 4, st);
        let yb_s = mk("yb_s", n * 4, st);
        let ycc_s = mk("ycc_s", n * 16, st);
        let out_px = mk("out_px", n * 4, st | wgpu::BufferUsages::COPY_SRC);
        let staging = mk(
            "staging",
            n * 4,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let u = &self.uni;
        let p = &self.pipes;
        let bg = vec![
            make_bg(device, &p[0], &[(0, u), (1, &frames), (2, &cur_ycc)]),
            make_bg(device, &p[1], &[(0, u), (1, &frames), (3, &luma4)]),
            make_bg(device, &p[2], &[(0, u), (3, &luma4), (4, &mvc)]),
            make_bg(device, &p[3], &[(0, u), (1, &frames), (4, &mvc), (5, &mv), (6, &sad)]),
            make_bg(
                device,
                &p[4],
                &[(0, u), (1, &frames), (2, &cur_ycc), (5, &mv), (6, &sad), (7, &ycc_t), (8, &yb_t)],
            ),
            make_bg(device, &p[5], &[(0, u), (8, &yb_t), (9, &yb_s)]),
            make_bg(device, &p[6], &[(0, u), (7, &ycc_t), (9, &yb_s), (10, &ycc_s)]),
            make_bg(
                device,
                &p[7],
                &[(0, u), (7, &ycc_t), (8, &yb_t), (9, &yb_s), (10, &ycc_s), (11, &out_px)],
            ),
        ];

        self.res = Some(Res {
            w,
            h,
            frames,
            out_px,
            staging,
            bg,
            _keep: vec![cur_ycc, luma4, mvc, mv, sad, ycc_t, yb_t, yb_s, ycc_s],
        });
    }

    fn dispatch(&self, enc: &mut wgpu::CommandEncoder, k: usize, x: u32, y: u32, z: u32) {
        let r = self.res.as_ref().unwrap();
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipes[k]);
        pass.set_bind_group(0, &r.bg[k], &[]);
        pass.dispatch_workgroups(x, y, z);
    }

    /// Denoise `frames[cur]` using the other frames of the window as temporal neighbours.
    /// `frames` must be consecutive frames (RGBA8). Result (RGBA8) is written to `out`.
    /// Returns GPU time in milliseconds.
    pub fn process(
        &mut self,
        frames: &[&[u8]],
        cur: usize,
        w: u32,
        h: u32,
        p: &Params,
        frame_idx: u32,
        out: &mut Vec<u8>,
    ) -> Result<f32, String> {
        if frames.is_empty() || frames.len() > MAXSLOTS || cur >= frames.len() {
            return Err("bad frame window".into());
        }
        self.ensure(w, h);
        let n = w as usize * h as usize;
        {
            let r = self.res.as_ref().unwrap();
            for (s, f) in frames.iter().enumerate() {
                if f.len() != n * 4 {
                    return Err("frame size mismatch".into());
                }
                self.queue.write_buffer(&r.frames, (s * n * 4) as u64, f);
            }
        }
        let ns = frames.len() as u32;
        let u = build_uniforms(w, h, ns, cur as u32, p, frame_idx);
        self.queue.write_buffer(&self.uni, 0, bytemuck::bytes_of(&u));

        let t0 = Instant::now();
        let gx = (w + 15) / 16;
        let gy = (h + 15) / 16;
        let w4 = (w + 3) / 4;
        let h4 = (h + 3) / 4;
        let cbw = (w4 + 7) / 8;
        let cbh = (h4 + 7) / 8;
        let fbw = (w + 15) / 16;
        let fbh = (h + 15) / 16;

        // 1) prepare + motion estimation + temporal
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.dispatch(&mut enc, 0, gx, gy, 1);
        if ns > 1 {
            if p.t_me != 0 {
                self.dispatch(&mut enc, 1, (w4 + 15) / 16, (h4 + 15) / 16, ns);
                self.dispatch(&mut enc, 2, (cbw + 7) / 8, (cbh + 7) / 8, ns);
            }
            self.dispatch(&mut enc, 3, (fbw + 7) / 8, (fbh + 7) / 8, ns);
        }
        self.dispatch(&mut enc, 4, gx, gy, 1);
        self.queue.submit(Some(enc.finish()));

        // 2) spatial luma (non-local means) in strips so a single GPU job never runs long
        //    enough to trigger the Windows GPU watchdog (TDR).
        let strip = ((300_000 / w).max(16) / 16) * 16;
        let mut row = 0u32;
        while row < h {
            let rows = strip.min(h - row);
            self.queue
                .write_buffer(&self.uni, ROW0_OFFSET, bytemuck::bytes_of(&row));
            let mut e = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            self.dispatch(&mut e, 5, gx, (rows + 15) / 16, 1);
            self.queue.submit(Some(e.finish()));
            row += strip;
        }

        // 3) chroma + final + readback
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.dispatch(&mut enc, 6, gx, gy, 1);
        self.dispatch(&mut enc, 7, gx, gy, 1);
        {
            let r = self.res.as_ref().unwrap();
            enc.copy_buffer_to_buffer(&r.out_px, 0, &r.staging, 0, (n * 4) as u64);
        }
        self.queue.submit(Some(enc.finish()));

        let r = self.res.as_ref().unwrap();
        let slice = r.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(format!("GPU readback failed: {e}")),
            Err(e) => return Err(e.to_string()),
        }
        {
            let data = slice.get_mapped_range();
            out.clear();
            out.extend_from_slice(&data);
        }
        r.staging.unmap();

        if let Some(e) = self.err.lock().unwrap().take() {
            return Err(format!("GPU error: {e}"));
        }
        Ok(t0.elapsed().as_secs_f32() * 1000.0)
    }
}
