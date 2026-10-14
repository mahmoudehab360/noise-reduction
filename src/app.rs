use crate::media::{EncodeOpts, VideoInfo};
use crate::params::Params;
use crate::worker::{Cmd, ExportJob, Msg, Worker};
use eframe::egui::{self, pos2, vec2, Color32, Rect};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};

#[derive(Clone, Copy, PartialEq)]
enum View {
    Denoised,
    Original,
    Split,
    Diff,
}

#[derive(Clone, Copy, PartialEq)]
enum Zoom {
    Fit,
    Pct(f32),
}

struct ExportProg {
    done: u32,
    total: u32,
    fps: f32,
    eta: f32,
}

pub struct App {
    tx: Sender<Cmd>,
    rx: Receiver<Msg>,
    info: Option<VideoInfo>,
    params: Params,
    requested: Option<(u32, Params)>,
    play_params: Params,
    play_gen: u32,
    frame: u32,
    playing: bool,
    looping: bool,
    in_pt: Option<u32>,
    out_pt: Option<u32>,
    tex_o: Option<egui::TextureHandle>,
    tex_d: Option<egui::TextureHandle>,
    tex_x: Option<egui::TextureHandle>,
    img_size: [usize; 2],
    raw_o: Vec<u8>,
    raw_d: Vec<u8>,
    diff_dirty: bool,
    view: View,
    zoom: Zoom,
    pan: egui::Vec2,
    split: f32,
    gpus: Vec<String>,
    gpu_sel: usize,
    status: String,
    error: Option<String>,
    last_ms: f32,
    export: Option<ExportProg>,
    codec: u32,
    quality: u32,
    noise_note: String,
}

fn find_tool(name: &str) -> PathBuf {
    let file = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join(&file);
            if p.exists() {
                return p;
            }
        }
    }
    PathBuf::from(file)
}

fn set_tex(
    ctx: &egui::Context,
    slot: &mut Option<egui::TextureHandle>,
    name: &str,
    size: [usize; 2],
    rgba: &[u8],
) {
    let img = egui::ColorImage::from_rgba_unmultiplied(size, rgba);
    if slot.is_some() {
        slot.as_mut().unwrap().set(img, egui::TextureOptions::LINEAR);
    } else {
        *slot = Some(ctx.load_texture(name, img, egui::TextureOptions::LINEAR));
    }
}

fn tc(frame: u32, fps: f64) -> String {
    let f = fps.round().max(1.0) as u32;
    let secs = frame / f;
    format!("{:02}:{:02}.{:02}", secs / 60, secs % 60, frame % f)
}

fn combo(ui: &mut egui::Ui, id: &str, label: &str, val: &mut u32, opts: &[&str]) {
    ui.horizontal(|ui| {
        ui.label(label);
        egui::ComboBox::from_id_source(id)
            .selected_text(opts[(*val as usize).min(opts.len() - 1)])
            .show_ui(ui, |ui| {
                for (i, o) in opts.iter().enumerate() {
                    ui.selectable_value(val, i as u32, *o);
                }
            });
    });
}

fn pct_slider(ui: &mut egui::Ui, v: &mut f32, label: &str) -> egui::Response {
    ui.add(egui::Slider::new(v, 0.0..=100.0).text(label).fixed_decimals(0))
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let (tx, wrx) = channel::<Cmd>();
        let (wtx, rx) = channel::<Msg>();
        let worker = Worker::new(wrx, wtx, cc.egui_ctx.clone(), find_tool("ffmpeg"), find_tool("ffprobe"));
        std::thread::Builder::new()
            .name("worker".into())
            .stack_size(16 << 20)
            .spawn(move || worker.run())
            .expect("could not start the worker thread");
        Self {
            tx,
            rx,
            info: None,
            params: Params::default(),
            requested: None,
            play_params: Params::default(),
            play_gen: 0,
            frame: 0,
            playing: false,
            looping: true,
            in_pt: None,
            out_pt: None,
            tex_o: None,
            tex_d: None,
            tex_x: None,
            img_size: [0, 0],
            raw_o: Vec::new(),
            raw_d: Vec::new(),
            diff_dirty: false,
            view: View::Denoised,
            zoom: Zoom::Fit,
            pan: egui::Vec2::ZERO,
            split: 0.5,
            gpus: Vec::new(),
            gpu_sel: 0,
            status: "Starting…".into(),
            error: None,
            last_ms: 0.0,
            export: None,
            codec: 0,
            quality: 16,
            noise_note: String::new(),
        }
    }

    // ------------------------------------------------------------ messages

    fn pump(&mut self, ctx: &egui::Context) {
        loop {
            match self.rx.try_recv() {
                Ok(m) => self.on_msg(ctx, m),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if self.error.is_none() {
                        self.error = Some("The background worker stopped unexpectedly.".into());
                    }
                    break;
                }
            }
        }
    }

    fn on_msg(&mut self, ctx: &egui::Context, m: Msg) {
        match m {
            Msg::Gpus(names, sel) => {
                self.gpus = names;
                self.gpu_sel = sel;
            }
            Msg::Info(info) => {
                self.frame = 0;
                self.in_pt = None;
                self.out_pt = None;
                self.playing = false;
                self.requested = None;
                self.tex_o = None;
                self.tex_d = None;
                self.tex_x = None;
                self.zoom = Zoom::Fit;
                self.pan = egui::Vec2::ZERO;
                self.error = None;
                self.noise_note.clear();
                self.status = format!(
                    "{}x{}  {:.3} fps  {} frames",
                    info.width,
                    info.height,
                    info.fps(),
                    info.frames
                );
                self.info = Some(info);
            }
            Msg::Frame { idx, w, h, orig, dn, ms } => {
                let live = self.playing || self.export.is_some();
                if live || idx == self.frame {
                    self.img_size = [w as usize, h as usize];
                    set_tex(ctx, &mut self.tex_o, "orig", self.img_size, &orig);
                    set_tex(ctx, &mut self.tex_d, "dn", self.img_size, &dn);
                    self.raw_o = orig;
                    self.raw_d = dn;
                    self.diff_dirty = true;
                    self.last_ms = ms;
                    if self.playing {
                        self.frame = idx;
                    }
                }
            }
            Msg::Progress { done, total, fps, eta } => {
                self.export = Some(ExportProg { done, total, fps, eta });
            }
            Msg::Noise(sy, sc) => {
                self.params.t_luma = sy;
                self.params.t_chroma = sc;
                self.params.s_luma = (sy * 0.8).round().max(1.0);
                self.params.s_chroma = sc;
                self.noise_note = format!("Estimated noise: luma {:.0}, chroma {:.0}", sy, sc);
            }
            Msg::PlayEnded(g) => {
                if g == self.play_gen && self.playing {
                    if self.looping {
                        self.frame = self.in_pt.unwrap_or(0);
                        self.start_play();
                    } else {
                        self.playing = false;
                        self.requested = None;
                    }
                }
            }
            Msg::ExportDone(path) => {
                self.export = None;
                self.requested = None;
                self.status = match path {
                    Some(p) => format!("Saved: {}", p.display()),
                    None => "Export cancelled.".into(),
                };
            }
            Msg::Status(s) => self.status = s,
            Msg::Error(e) => {
                self.error = Some(e);
                self.playing = false;
                self.export = None;
            }
        }
    }

    // ------------------------------------------------------------ actions

    fn open(&mut self, p: PathBuf) {
        self.stop_play();
        let _ = self.tx.send(Cmd::Open(p));
        self.status = "Opening…".into();
    }

    fn start_play(&mut self) {
        let last = match &self.info {
            Some(i) => i.frames.saturating_sub(1),
            None => return,
        };
        let a = self.in_pt.unwrap_or(0).min(last);
        let b = self.out_pt.unwrap_or(last).min(last);
        let mut s = self.frame;
        if s < a || s >= b {
            s = a;
        }
        self.play_gen = self.play_gen.wrapping_add(1);
        self.playing = true;
        self.play_params = self.params.clone();
        let _ = self.tx.send(Cmd::Play {
            start: s,
            end: b.max(s),
            params: self.params.clone(),
            gen: self.play_gen,
        });
    }

    fn stop_play(&mut self) {
        if self.playing {
            self.playing = false;
            let _ = self.tx.send(Cmd::Stop);
            self.requested = None;
        }
    }

    fn auto_request(&mut self) {
        if self.info.is_none() || self.export.is_some() {
            return;
        }
        if self.playing {
            if self.params != self.play_params {
                self.play_params = self.params.clone();
                let _ = self.tx.send(Cmd::SetParams(self.params.clone()));
            }
            return;
        }
        let want = (self.frame, self.params.clone());
        if self.requested.as_ref() != Some(&want) {
            let _ = self.tx.send(Cmd::Preview {
                frame: want.0,
                params: want.1.clone(),
            });
            self.requested = Some(want);
        }
    }

    fn handle_input(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if self.export.is_none() {
            if let Some(p) = dropped.into_iter().next() {
                self.open(p);
            }
        }
        if self.export.is_some() {
            return;
        }
        let last = match &self.info {
            Some(i) => i.frames.saturating_sub(1),
            None => return,
        };
        let (space, left, right, ki, ko) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::Space),
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
                i.key_pressed(egui::Key::I),
                i.key_pressed(egui::Key::O),
            )
        });
        if space {
            if self.playing {
                self.stop_play();
            } else {
                self.start_play();
            }
        }
        if left {
            self.stop_play();
            self.frame = self.frame.saturating_sub(1);
        }
        if right {
            self.stop_play();
            self.frame = (self.frame + 1).min(last);
        }
        if ki {
            self.in_pt = Some(self.frame);
        }
        if ko {
            self.out_pt = Some(self.frame);
        }
    }

    fn ensure_diff(&mut self, ctx: &egui::Context) {
        if self.view != View::Diff || !self.diff_dirty {
            return;
        }
        let n = self.img_size[0] * self.img_size[1];
        if n == 0 || self.raw_o.len() != n * 4 || self.raw_d.len() != n * 4 {
            return;
        }
        let mut d = vec![255u8; n * 4];
        for k in 0..n {
            for c in 0..3 {
                let v = 128 + 4 * (self.raw_o[k * 4 + c] as i32 - self.raw_d[k * 4 + c] as i32);
                d[k * 4 + c] = v.clamp(0, 255) as u8;
            }
        }
        set_tex(ctx, &mut self.tex_x, "diff", self.img_size, &d);
        self.diff_dirty = false;
    }

    // ------------------------------------------------------------ panels

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button("Open video…").clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("Video", &["mp4", "mov", "mkv", "avi", "mxf", "webm", "m4v", "mts", "m2ts"])
                    .pick_file()
                {
                    self.open(p);
                }
            }
            if let Some(info) = &self.info {
                ui.label(
                    info.path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                );
            }
            ui.separator();
            ui.label("GPU:");
            let mut sel = self.gpu_sel;
            egui::ComboBox::from_id_source("gpu")
                .width(260.0)
                .selected_text(self.gpus.get(sel).cloned().unwrap_or_else(|| "…".to_string()))
                .show_ui(ui, |ui| {
                    for (i, n) in self.gpus.iter().enumerate() {
                        ui.selectable_value(&mut sel, i, n.as_str());
                    }
                });
            if sel != self.gpu_sel {
                self.gpu_sel = sel;
                self.stop_play();
                let _ = self.tx.send(Cmd::SetGpu(sel));
                self.requested = None;
            }
            ui.separator();
            ui.weak(self.status.clone());
        });
        let mut dismiss = false;
        if let Some(e) = &self.error {
            ui.horizontal(|ui| {
                ui.colored_label(Color32::LIGHT_RED, e.as_str());
                if ui.small_button("dismiss").clicked() {
                    dismiss = true;
                }
            });
        }
        if dismiss {
            self.error = None;
        }
        ui.add_space(2.0);
    }

    fn bottom(&mut self, ui: &mut egui::Ui, busy: bool) {
        let info = match self.info.clone() {
            Some(i) => i,
            None => {
                ui.add_space(8.0);
                ui.label("Open a video with the button above, or drag a file onto the window.");
                return;
            }
        };
        let last = info.frames.saturating_sub(1);
        ui.add_space(4.0);
        ui.add_enabled_ui(!busy, |ui| {
            ui.horizontal(|ui| {
                if ui.button("|<").on_hover_text("Go to start / In point").clicked() {
                    self.stop_play();
                    self.frame = self.in_pt.unwrap_or(0);
                }
                if ui.button("<").on_hover_text("Previous frame (Left arrow)").clicked() {
                    self.stop_play();
                    self.frame = self.frame.saturating_sub(1);
                }
                let lbl = if self.playing { "Pause" } else { "Play" };
                if ui
                    .add(egui::Button::new(lbl).min_size(vec2(64.0, 0.0)))
                    .on_hover_text("Plays the In-Out range with the filters applied (Space)")
                    .clicked()
                {
                    if self.playing {
                        self.stop_play();
                    } else {
                        self.start_play();
                    }
                }
                if ui.button(">").on_hover_text("Next frame (Right arrow)").clicked() {
                    self.stop_play();
                    self.frame = (self.frame + 1).min(last);
                }
                if ui.button(">|").on_hover_text("Go to end / Out point").clicked() {
                    self.stop_play();
                    self.frame = self.out_pt.unwrap_or(last);
                }
                ui.checkbox(&mut self.looping, "Loop");
                ui.separator();
                if ui.button("Set In (I)").clicked() {
                    self.in_pt = Some(self.frame);
                    if self.out_pt.map_or(false, |o| o < self.frame) {
                        self.out_pt = None;
                    }
                }
                if ui.button("Set Out (O)").clicked() {
                    self.out_pt = Some(self.frame);
                    if self.in_pt.map_or(false, |a| a > self.frame) {
                        self.in_pt = None;
                    }
                }
                if ui.button("Clear In/Out").clicked() {
                    self.in_pt = None;
                    self.out_pt = None;
                }
                ui.separator();
                ui.monospace(format!(
                    "{}   frame {}/{}",
                    tc(self.frame, info.fps()),
                    self.frame + 1,
                    info.frames
                ));
            });
            self.timeline(ui, &info);
        });
    }

    fn timeline(&mut self, ui: &mut egui::Ui, info: &VideoInfo) {
        let total = info.frames.max(1);
        let (resp, painter) = ui.allocate_painter(vec2(ui.available_width(), 56.0), egui::Sense::click_and_drag());
        let rect = resp.rect;
        painter.rect_filled(rect, 3.0, Color32::from_gray(28));
        let tf = total as f32;
        let x_of = |f: u32| rect.left() + rect.width() * (f as f32 / tf);

        let a = self.in_pt.unwrap_or(0).min(total - 1);
        let b = self.out_pt.unwrap_or(total - 1).min(total - 1);
        painter.rect_filled(
            Rect::from_min_max(pos2(x_of(a), rect.top()), pos2(x_of(b + 1), rect.bottom())),
            0.0,
            Color32::from_rgb(38, 66, 104),
        );

        let fps = info.fps();
        let secs_total = total as f64 / fps;
        let steps = [1u32, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600];
        let px_per_sec = rect.width() as f64 / secs_total.max(0.001);
        let step = steps
            .iter()
            .copied()
            .find(|s| px_per_sec * (*s as f64) >= 70.0)
            .unwrap_or(3600);
        let mut s = 0u32;
        while (s as f64) <= secs_total {
            let f = ((s as f64) * fps) as u32;
            let x = x_of(f.min(total));
            painter.line_segment(
                [pos2(x, rect.bottom() - 14.0), pos2(x, rect.bottom())],
                egui::Stroke::new(1.0, Color32::GRAY),
            );
            painter.text(
                pos2(x + 3.0, rect.bottom() - 14.0),
                egui::Align2::LEFT_BOTTOM,
                format!("{}:{:02}", s / 60, s % 60),
                egui::FontId::proportional(10.0),
                Color32::GRAY,
            );
            s += step;
        }

        let px = x_of(self.frame.min(total));
        let orange = Color32::from_rgb(255, 160, 40);
        painter.line_segment(
            [pos2(px, rect.top()), pos2(px, rect.bottom())],
            egui::Stroke::new(2.0, orange),
        );
        painter.rect_filled(Rect::from_center_size(pos2(px, rect.top() + 5.0), vec2(9.0, 10.0)), 2.0, orange);

        if resp.clicked() || resp.dragged() {
            if let Some(pos) = resp.interact_pointer_pos() {
                let f = (((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0) * tf) as u32;
                self.stop_play();
                self.frame = f.min(total - 1);
            }
        }
    }

    fn side(&mut self, ui: &mut egui::Ui, busy: bool) {
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            let has_video = self.info.is_some();
            ui.add_enabled_ui(!busy && has_video, |ui| {
                egui::CollapsingHeader::new("Noise profile")
                    .default_open(true)
                    .show(ui, |ui| {
                        if ui.button("Auto-detect noise level (current frame)").clicked() {
                            let _ = self.tx.send(Cmd::Noise);
                        }
                        if !self.noise_note.is_empty() {
                            ui.small(self.noise_note.clone());
                        }
                        ui.small("Thresholds are on a 0-100 scale. Start from the detected values, then trade smoothing against detail.");
                    });

                egui::CollapsingHeader::new("Temporal NR")
                    .default_open(true)
                    .show(ui, |ui| {
                        let p = &mut self.params;
                        ui.add(egui::Slider::new(&mut p.t_frames, 0..=3).text("Frames (each side)"))
                            .on_hover_text("How many frames before AND after are averaged in. 0 turns Temporal NR off.");
                        combo(ui, "t_me", "Motion estimation", &mut p.t_me, &["Off (static only)", "Faster", "Better"]);
                        combo(ui, "t_range", "Motion range", &mut p.t_range, &["Small", "Medium", "Large"]);
                        pct_slider(ui, &mut p.t_luma, "Luma threshold");
                        pct_slider(ui, &mut p.t_chroma, "Chroma threshold");
                        pct_slider(ui, &mut p.t_motion, "Motion threshold")
                            .on_hover_text("Higher = more tolerant to motion (stronger blending, risk of ghosting).");
                        pct_slider(ui, &mut p.t_blend, "Blend");
                    });

                egui::CollapsingHeader::new("Spatial NR")
                    .default_open(true)
                    .show(ui, |ui| {
                        let p = &mut self.params;
                        ui.checkbox(&mut p.s_on, "Enable Spatial NR");
                        combo(ui, "s_mode", "Mode", &mut p.s_mode, &["Faster", "Better", "Enhanced"]);
                        combo(ui, "s_radius", "Radius", &mut p.s_radius, &["Small", "Medium", "Large"]);
                        ui.small("Enhanced + Large is the slowest setting. Use Faster + Small for smooth playback.");
                        pct_slider(ui, &mut p.s_luma, "Luma threshold");
                        pct_slider(ui, &mut p.s_chroma, "Chroma threshold");
                        pct_slider(ui, &mut p.s_blend, "Blend");
                    });

                egui::CollapsingHeader::new("Finishing")
                    .default_open(false)
                    .show(ui, |ui| {
                        let p = &mut self.params;
                        pct_slider(ui, &mut p.detail, "Detail recovery")
                            .on_hover_text("Gives back part of the fine detail that Spatial NR removed.");
                        pct_slider(ui, &mut p.sharpen, "Sharpen");
                        pct_slider(ui, &mut p.grain, "Film grain");
                        ui.add(egui::Slider::new(&mut p.grain_size, 1..=4).text("Grain size"));
                    });

                if ui.button("Reset all settings").clicked() {
                    self.params = Params::default();
                }

                egui::CollapsingHeader::new("Preview")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.radio_value(&mut self.view, View::Denoised, "Denoised");
                            ui.radio_value(&mut self.view, View::Original, "Original");
                        });
                        ui.horizontal(|ui| {
                            ui.radio_value(&mut self.view, View::Split, "Split");
                            ui.radio_value(&mut self.view, View::Diff, "Removed noise");
                        });
                        if self.view == View::Diff {
                            self.diff_dirty = true;
                        }
                        if self.view == View::Split {
                            ui.add(egui::Slider::new(&mut self.split, 0.0..=1.0).text("Split position"));
                        }
                        ui.horizontal(|ui| {
                            ui.label("Zoom:");
                            if ui.button("Fit").clicked() {
                                self.zoom = Zoom::Fit;
                                self.pan = egui::Vec2::ZERO;
                            }
                            if ui.button("100%").clicked() {
                                self.zoom = Zoom::Pct(1.0);
                                self.pan = egui::Vec2::ZERO;
                            }
                            if ui.button("200%").clicked() {
                                self.zoom = Zoom::Pct(2.0);
                                self.pan = egui::Vec2::ZERO;
                            }
                        });
                        ui.small("Drag the image to pan when zoomed. Judge noise at 100%.");
                    });
            });

            ui.separator();
            ui.heading("Export");
            if let Some(ex) = &self.export {
                let frac = ex.done as f32 / ex.total.max(1) as f32;
                ui.add(egui::ProgressBar::new(frac).show_percentage());
                ui.label(format!(
                    "{}/{} frames   {:.1} fps   ~{:.0}s left",
                    ex.done, ex.total, ex.fps, ex.eta
                ));
                if ui.button("Cancel").clicked() {
                    let _ = self.tx.send(Cmd::Stop);
                }
            } else if has_video {
                combo(ui, "codec", "Format", &mut self.codec, &["H.264 (MP4)", "H.265 (MP4)", "ProRes 422 HQ (MOV)"]);
                if self.codec < 2 {
                    ui.add(egui::Slider::new(&mut self.quality, 10..=28).text("CRF (lower = better)"));
                }
                ui.small("Exports the In-Out range (or the whole clip) with the audio.");
                if ui.button("Export…").clicked() {
                    self.do_export();
                }
            }
        });
    }

    fn do_export(&mut self) {
        let (stem, last) = match &self.info {
            Some(i) => (
                i.path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "video".to_string()),
                i.frames.saturating_sub(1),
            ),
            None => return,
        };
        let ext = if self.codec == 2 { "mov" } else { "mp4" };
        let picked = rfd::FileDialog::new()
            .set_file_name(format!("{stem}_dn.{ext}"))
            .add_filter("Video", &[ext])
            .save_file();
        if let Some(out) = picked {
            self.stop_play();
            let start = self.in_pt.unwrap_or(0).min(last);
            let end = self.out_pt.unwrap_or(last).min(last).max(start);
            let _ = self.tx.send(Cmd::Export(ExportJob {
                out,
                start,
                end,
                params: self.params.clone(),
                opts: EncodeOpts {
                    codec: self.codec,
                    quality: self.quality,
                },
            }));
            self.export = Some(ExportProg {
                done: 0,
                total: end - start + 1,
                fps: 0.0,
                eta: 0.0,
            });
            self.error = None;
        }
    }

    fn center(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        self.ensure_diff(ctx);
        let (resp, painter) = ui.allocate_painter(ui.available_size(), egui::Sense::drag());
        let rect = resp.rect;
        painter.rect_filled(rect, 0.0, Color32::from_gray(16));
        let (to, td) = (self.tex_o.clone(), self.tex_d.clone());
        let (Some(to), Some(td)) = (to, td) else {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                if self.info.is_some() { "Processing the first frame…" } else { "Open a video to start" },
                egui::FontId::proportional(18.0),
                Color32::GRAY,
            );
            return;
        };
        let [iw, ih] = self.img_size;
        if iw == 0 || ih == 0 {
            return;
        }
        let (iw, ih) = (iw as f32, ih as f32);
        let scale = match self.zoom {
            Zoom::Fit => (rect.width() / iw).min(rect.height() / ih),
            Zoom::Pct(z) => z / ctx.pixels_per_point(),
        };
        if matches!(self.zoom, Zoom::Fit) {
            self.pan = egui::Vec2::ZERO;
        } else if resp.dragged() {
            self.pan += resp.drag_delta();
        }
        let img_rect = Rect::from_center_size(rect.center() + self.pan, vec2(iw * scale, ih * scale));
        let full = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
        let white = Color32::WHITE;

        let label = match self.view {
            View::Denoised => {
                painter.image(td.id(), img_rect, full, white);
                "DENOISED"
            }
            View::Original => {
                painter.image(to.id(), img_rect, full, white);
                "ORIGINAL"
            }
            View::Split => {
                let sx = img_rect.left() + img_rect.width() * self.split;
                let l = Rect::from_min_max(img_rect.min, pos2(sx, img_rect.max.y));
                let r = Rect::from_min_max(pos2(sx, img_rect.min.y), img_rect.max);
                painter.image(
                    to.id(),
                    l,
                    Rect::from_min_max(pos2(0.0, 0.0), pos2(self.split, 1.0)),
                    white,
                );
                painter.image(
                    td.id(),
                    r,
                    Rect::from_min_max(pos2(self.split, 0.0), pos2(1.0, 1.0)),
                    white,
                );
                painter.line_segment(
                    [pos2(sx, img_rect.min.y), pos2(sx, img_rect.max.y)],
                    egui::Stroke::new(1.5, white),
                );
                "ORIGINAL | DENOISED"
            }
            View::Diff => {
                if let Some(tx) = &self.tex_x {
                    painter.image(tx.id(), img_rect, full, white);
                }
                "REMOVED NOISE (x4, gray = unchanged)"
            }
        };
        let soft = Color32::from_rgba_unmultiplied(255, 255, 255, 200);
        painter.text(
            rect.left_top() + vec2(10.0, 8.0),
            egui::Align2::LEFT_TOP,
            label,
            egui::FontId::proportional(13.0),
            soft,
        );
        painter.text(
            rect.left_bottom() + vec2(10.0, -8.0),
            egui::Align2::LEFT_BOTTOM,
            format!("{}x{}    {:.0} ms/frame (GPU + readback)", iw as u32, ih as u32, self.last_ms),
            egui::FontId::proportional(12.0),
            soft,
        );
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.pump(ctx);
        self.handle_input(ctx);
        let busy = self.export.is_some();
        egui::TopBottomPanel::top("top").show(ctx, |ui| self.top_bar(ui));
        egui::TopBottomPanel::bottom("bottom")
            .min_height(110.0)
            .show(ctx, |ui| self.bottom(ui, busy));
        egui::SidePanel::right("right")
            .default_width(350.0)
            .show(ctx, |ui| self.side(ui, busy));
        egui::CentralPanel::default().show(ctx, |ui| self.center(ui, ctx));
        self.auto_request();
    }
}
