use crate::gpu::{self, Denoiser};
use crate::media::{self, EncodeOpts, Encoder, FrameReader, VideoInfo};
use crate::noise;
use crate::params::Params;
use eframe::egui;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

pub struct ExportJob {
    pub out: PathBuf,
    pub start: u32,
    pub end: u32,
    pub params: Params,
    pub opts: EncodeOpts,
}

pub enum Cmd {
    Open(PathBuf),
    SetGpu(usize),
    Preview { frame: u32, params: Params },
    Play { start: u32, end: u32, params: Params, gen: u32 },
    SetParams(Params),
    Stop,
    Noise,
    Export(ExportJob),
}

pub enum Msg {
    Gpus(Vec<String>, usize),
    Info(VideoInfo),
    Frame { idx: u32, w: u32, h: u32, orig: Vec<u8>, dn: Vec<u8>, ms: f32 },
    Progress { done: u32, total: u32, fps: f32, eta: f32 },
    Noise(f32, f32),
    PlayEnded(u32),
    ExportDone(Option<PathBuf>),
    Status(String),
    Error(String),
}

/// extra frames kept behind the playhead so the temporal window can grow without re-decoding
const HIST: u32 = 3;

pub struct Worker {
    rx: Receiver<Cmd>,
    tx: Sender<Msg>,
    ctx: egui::Context,
    ffmpeg: PathBuf,
    ffprobe: PathBuf,
    instance: Option<wgpu::Instance>,
    adapters: Vec<wgpu::Adapter>,
    den: Option<Denoiser>,
    info: Option<VideoInfo>,
    pending: Option<Cmd>,
    last_orig: Vec<u8>,
    out_buf: Vec<u8>,
}

impl Worker {
    pub fn new(rx: Receiver<Cmd>, tx: Sender<Msg>, ctx: egui::Context, ffmpeg: PathBuf, ffprobe: PathBuf) -> Self {
        Self {
            rx,
            tx,
            ctx,
            ffmpeg,
            ffprobe,
            instance: None,
            adapters: Vec::new(),
            den: None,
            info: None,
            pending: None,
            last_orig: Vec::new(),
            out_buf: Vec::new(),
        }
    }

    fn send(&self, m: Msg) {
        let _ = self.tx.send(m);
        self.ctx.request_repaint();
    }

    fn init_gpu(&mut self, i: usize) {
        match Denoiser::new(&self.adapters[i]) {
            Ok(d) => {
                self.den = Some(d);
                let l = gpu::adapter_label(&self.adapters[i]);
                self.send(Msg::Status(format!("GPU ready: {l}")));
            }
            Err(e) => {
                self.den = None;
                self.send(Msg::Error(e));
            }
        }
    }

    pub fn run(mut self) {
        let (inst, adapters) = gpu::list_adapters();
        let names: Vec<String> = adapters.iter().map(gpu::adapter_label).collect();
        self.instance = Some(inst);
        self.adapters = adapters;
        if self.adapters.is_empty() {
            self.send(Msg::Error(
                "No Vulkan / DirectX 12 GPU was found. Install or update the NVIDIA driver.".into(),
            ));
        } else {
            self.init_gpu(0);
        }
        self.send(Msg::Gpus(names, 0));

        loop {
            let cmd = match self.pending.take() {
                Some(c) => c,
                None => match self.rx.recv() {
                    Ok(c) => c,
                    Err(_) => break,
                },
            };
            let cmd = self.coalesce(cmd);
            match cmd {
                Cmd::Open(path) => match media::probe(&self.ffprobe, &path) {
                    Ok(info) => {
                        self.info = Some(info.clone());
                        self.last_orig.clear();
                        self.send(Msg::Info(info));
                    }
                    Err(e) => self.send(Msg::Error(e)),
                },
                Cmd::SetGpu(i) => {
                    if i < self.adapters.len() {
                        self.den = None;
                        self.init_gpu(i);
                    }
                }
                Cmd::Preview { frame, params } => self.preview(frame, &params),
                Cmd::Play { start, end, params, gen } => {
                    self.play(start, end, params, gen);
                }
                Cmd::Export(job) => self.export(job),
                Cmd::Noise => {
                    if let Some(info) = &self.info {
                        if self.last_orig.len() == info.frame_bytes() {
                            let (a, b) = noise::estimate(
                                &self.last_orig,
                                info.width as usize,
                                info.height as usize,
                            );
                            self.send(Msg::Noise(a, b));
                        }
                    }
                }
                Cmd::SetParams(_) | Cmd::Stop => {}
            }
        }
    }

    /// While the user drags a slider or scrubs, only the newest preview request matters.
    fn coalesce(&mut self, mut cmd: Cmd) -> Cmd {
        if matches!(cmd, Cmd::Preview { .. }) {
            while let Ok(next) = self.rx.try_recv() {
                if matches!(next, Cmd::Preview { .. }) {
                    cmd = next;
                } else {
                    self.pending = Some(next);
                    break;
                }
            }
        }
        cmd
    }

    /// Denoise frame `i` using whatever neighbours of `items` fall inside the temporal window.
    /// The result is left in `self.out_buf`.
    fn run_window(&mut self, items: &[(u32, &[u8])], i: u32, p: &Params) -> Result<f32, String> {
        let (w, h) = match &self.info {
            Some(x) => (x.width, x.height),
            None => return Err("no video loaded".into()),
        };
        let n = p.t_frames;
        let sel: Vec<&(u32, &[u8])> = items
            .iter()
            .filter(|e| e.0 + n >= i && e.0 <= i + n)
            .collect();
        let cur = sel
            .iter()
            .position(|e| e.0 == i)
            .ok_or_else(|| "current frame is missing from the window".to_string())?;
        let slices: Vec<&[u8]> = sel.iter().map(|e| e.1).collect();
        let den = self
            .den
            .as_mut()
            .ok_or_else(|| "GPU is not ready".to_string())?;
        den.process(&slices, cur, w, h, p, i, &mut self.out_buf)
    }

    fn emit(&mut self, items: &[(u32, &[u8])], i: u32, ms: f32, keep: bool) {
        let (w, h) = match &self.info {
            Some(x) => (x.width, x.height),
            None => return,
        };
        let orig = items
            .iter()
            .find(|e| e.0 == i)
            .map(|e| e.1.to_vec())
            .unwrap_or_default();
        if keep {
            self.last_orig = orig.clone();
        }
        let dn = self.out_buf.clone();
        self.send(Msg::Frame { idx: i, w, h, orig, dn, ms });
    }

    fn preview(&mut self, frame: u32, p: &Params) {
        let info = match self.info.clone() {
            Some(i) => i,
            None => return,
        };
        if self.den.is_none() {
            self.send(Msg::Error("GPU is not ready.".into()));
            return;
        }
        let last = info.frames.saturating_sub(1);
        let frame = frame.min(last);
        let lo = frame.saturating_sub(p.t_frames);
        let hi = (frame + p.t_frames).min(last);
        let mut rd = match FrameReader::spawn(&self.ffmpeg, &info, lo, hi - lo + 1) {
            Ok(r) => r,
            Err(e) => {
                self.send(Msg::Error(e));
                return;
            }
        };
        let mut win: Vec<(u32, Vec<u8>)> = Vec::new();
        loop {
            match rd.read() {
                Ok(Some(x)) => win.push(x),
                Ok(None) => break,
                Err(e) => {
                    self.send(Msg::Error(e));
                    return;
                }
            }
        }
        if !win.iter().any(|e| e.0 == frame) {
            self.send(Msg::Error(format!(
                "Could not decode frame {frame}. Details: {}",
                media::log_path("decode").display()
            )));
            return;
        }
        let items: Vec<(u32, &[u8])> = win.iter().map(|e| (e.0, e.1.as_slice())).collect();
        match self.run_window(&items, frame, p) {
            Ok(ms) => self.emit(&items, frame, ms, true),
            Err(e) => self.send(Msg::Error(e)),
        }
    }

    fn play(&mut self, start: u32, end: u32, mut p: Params, gen: u32) {
        let info = match self.info.clone() {
            Some(i) => i,
            None => {
                self.send(Msg::PlayEnded(gen));
                return;
            }
        };
        if self.den.is_none() {
            self.send(Msg::Error("GPU is not ready.".into()));
            self.send(Msg::PlayEnded(gen));
            return;
        }
        let last = info.frames.saturating_sub(1);
        let end = end.min(last);
        let start = start.min(end);
        let first = start.saturating_sub(HIST);
        let stop = (end + HIST).min(last);
        let mut rd = match FrameReader::spawn(&self.ffmpeg, &info, first, stop - first + 1) {
            Ok(r) => r,
            Err(e) => {
                self.send(Msg::Error(e));
                self.send(Msg::PlayEnded(gen));
                return;
            }
        };
        let mut win: VecDeque<(u32, Vec<u8>)> = VecDeque::new();
        let fps = info.fps();
        let t0 = Instant::now();
        let mut done = 0u32;

        'outer: for i in start..=end {
            loop {
                match self.rx.try_recv() {
                    Ok(Cmd::SetParams(np)) => p = np,
                    Ok(Cmd::Stop) => break 'outer,
                    Ok(other) => {
                        self.pending = Some(other);
                        break 'outer;
                    }
                    Err(_) => break,
                }
            }
            let need = (i + p.t_frames).min(last);
            while win.back().map_or(true, |e| e.0 < need) {
                match rd.read() {
                    Ok(Some(x)) => win.push_back(x),
                    Ok(None) => break,
                    Err(e) => {
                        self.send(Msg::Error(e));
                        break 'outer;
                    }
                }
            }
            while win.front().map_or(false, |e| e.0 + HIST < i) {
                win.pop_front();
            }
            let items: Vec<(u32, &[u8])> = win.iter().map(|e| (e.0, e.1.as_slice())).collect();
            match self.run_window(&items, i, &p) {
                Ok(ms) => self.emit(&items, i, ms, false),
                Err(e) => {
                    self.send(Msg::Error(e));
                    break 'outer;
                }
            }
            done += 1;
            // never run faster than real time
            let target = Duration::from_secs_f64(done as f64 / fps);
            let el = t0.elapsed();
            if el < target {
                std::thread::sleep(target - el);
            }
        }
        self.send(Msg::PlayEnded(gen));
    }

    fn export(&mut self, job: ExportJob) {
        let info = match self.info.clone() {
            Some(i) => i,
            None => return,
        };
        if self.den.is_none() {
            self.send(Msg::Error("GPU is not ready.".into()));
            return;
        }
        let last = info.frames.saturating_sub(1);
        let end = job.end.min(last);
        let start = job.start.min(end);
        let total = end - start + 1;
        let p = job.params.clone();
        let first = start.saturating_sub(HIST);
        let stop = (end + HIST).min(last);

        let mut rd = match FrameReader::spawn(&self.ffmpeg, &info, first, stop - first + 1) {
            Ok(r) => r,
            Err(e) => {
                self.send(Msg::Error(e));
                return;
            }
        };
        let mut enc = match Encoder::spawn(&self.ffmpeg, &info, &job.out, start, total, &job.opts) {
            Ok(e) => e,
            Err(e) => {
                self.send(Msg::Error(e));
                return;
            }
        };
        let mut win: VecDeque<(u32, Vec<u8>)> = VecDeque::new();
        let t0 = Instant::now();
        let mut cancelled = false;

        for i in start..=end {
            let mut stop_req = false;
            loop {
                match self.rx.try_recv() {
                    Ok(Cmd::Stop) => {
                        stop_req = true;
                        break;
                    }
                    Ok(_) => {}
                    Err(_) => break,
                }
            }
            if stop_req {
                cancelled = true;
                break;
            }
            let need = (i + p.t_frames).min(last);
            while win.back().map_or(true, |e| e.0 < need) {
                match rd.read() {
                    Ok(Some(x)) => win.push_back(x),
                    Ok(None) => break,
                    Err(e) => {
                        self.send(Msg::Error(e));
                        return;
                    }
                }
            }
            while win.front().map_or(false, |e| e.0 + HIST < i) {
                win.pop_front();
            }
            let items: Vec<(u32, &[u8])> = win.iter().map(|e| (e.0, e.1.as_slice())).collect();
            let ms = match self.run_window(&items, i, &p) {
                Ok(ms) => ms,
                Err(e) => {
                    self.send(Msg::Error(e));
                    return;
                }
            };
            if let Err(e) = enc.write(&self.out_buf) {
                self.send(Msg::Error(e));
                return;
            }
            let done = i - start + 1;
            let el = t0.elapsed().as_secs_f32().max(0.001);
            let fps = done as f32 / el;
            let eta = (total - done) as f32 / fps.max(0.001);
            self.send(Msg::Progress { done, total, fps, eta });
            if done % 10 == 1 || i == end {
                self.emit(&items, i, ms, false);
            }
        }

        if cancelled {
            enc.abort();
            self.send(Msg::ExportDone(None));
            return;
        }
        match enc.finish() {
            Ok(()) => self.send(Msg::ExportDone(Some(job.out.clone()))),
            Err(e) => self.send(Msg::Error(e)),
        }
    }
}
