use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

/// Build a Command that does not flash a console window on Windows.
pub fn command(exe: &Path) -> Command {
    let mut c = Command::new(exe);
    #[cfg(windows)]
    c.creation_flags(0x0800_0000);
    c
}

pub fn log_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("lowlight-dn-{name}.log"))
}

fn log_stdio(name: &str) -> Stdio {
    match File::create(log_path(name)) {
        Ok(f) => Stdio::from(f),
        Err(_) => Stdio::null(),
    }
}

#[derive(Clone, Debug)]
pub struct VideoInfo {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub frames: u32,
    pub matrix: String, // bt709 | bt601 | bt2020
    pub range: String,  // tv | pc
}

impl VideoInfo {
    pub fn fps(&self) -> f64 {
        self.fps_num as f64 / self.fps_den.max(1) as f64
    }
    pub fn frame_bytes(&self) -> usize {
        self.width as usize * self.height as usize * 4
    }
}

fn parse_frac(s: &str) -> Option<(u32, u32)> {
    let mut it = s.split('/');
    let a: u32 = it.next()?.trim().parse().ok()?;
    let b: u32 = it.next().unwrap_or("1").trim().parse().ok()?;
    if a == 0 || b == 0 {
        None
    } else {
        Some((a, b))
    }
}

pub fn probe(ffprobe: &Path, path: &Path) -> Result<VideoInfo, String> {
    let out = command(ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,color_space,color_range,pix_fmt:stream_tags=rotate:stream_side_data=rotation:format=duration",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .map_err(|e| format!("Could not run ffprobe ({e}). Put ffprobe.exe next to the program."))?;
    if !out.status.success() {
        return Err(format!("ffprobe failed: {}", String::from_utf8_lossy(&out.stderr)));
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).map_err(|e| e.to_string())?;
    let s = v["streams"]
        .get(0)
        .ok_or_else(|| "No video stream found in this file.".to_string())?;

    let mut width = s["width"].as_u64().unwrap_or(0) as u32;
    let mut height = s["height"].as_u64().unwrap_or(0) as u32;
    if width == 0 || height == 0 {
        return Err("Could not read the video size.".into());
    }

    // Phone footage is often stored rotated; ffmpeg auto-rotates, so swap the size.
    let mut rot: i64 = 0;
    if let Some(arr) = s["side_data_list"].as_array() {
        for d in arr {
            if let Some(r) = d["rotation"].as_i64() {
                rot = r;
            } else if let Some(r) = d["rotation"].as_f64() {
                rot = r as i64;
            }
        }
    }
    if let Some(r) = s["tags"]["rotate"].as_str().and_then(|x| x.parse::<i64>().ok()) {
        rot = r;
    }
    if rot.rem_euclid(180) == 90 {
        std::mem::swap(&mut width, &mut height);
    }

    let (fps_num, fps_den) = s["avg_frame_rate"]
        .as_str()
        .and_then(parse_frac)
        .or_else(|| s["r_frame_rate"].as_str().and_then(parse_frac))
        .unwrap_or((30, 1));

    let dur = s["duration"]
        .as_str()
        .and_then(|x| x.parse::<f64>().ok())
        .or_else(|| v["format"]["duration"].as_str().and_then(|x| x.parse::<f64>().ok()))
        .unwrap_or(0.0);
    let frames = s["nb_frames"]
        .as_str()
        .and_then(|x| x.parse::<u32>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| (dur * fps_num as f64 / fps_den as f64).round() as u32);
    if frames == 0 {
        return Err("Could not determine the number of frames.".into());
    }

    let cs = s["color_space"].as_str().unwrap_or("");
    let matrix = match cs {
        "bt709" => "bt709",
        "smpte170m" | "bt470bg" => "bt601",
        "bt2020nc" | "bt2020c" => "bt2020",
        _ => {
            if width.max(height) >= 1280 {
                "bt709"
            } else {
                "bt601"
            }
        }
    }
    .to_string();
    let pix = s["pix_fmt"].as_str().unwrap_or("");
    let range = if s["color_range"].as_str() == Some("pc") || pix.starts_with("yuvj") {
        "pc"
    } else {
        "tv"
    }
    .to_string();

    Ok(VideoInfo {
        path: path.to_path_buf(),
        width,
        height,
        fps_num,
        fps_den,
        frames,
        matrix,
        range,
    })
}

// ---------------------------------------------------------------------------
// Decoder: streams raw RGBA frames out of ffmpeg
// ---------------------------------------------------------------------------

pub struct FrameReader {
    child: Child,
    out: ChildStdout,
    next_idx: u32,
    remaining: u32,
    fsize: usize,
}

impl FrameReader {
    pub fn spawn(ffmpeg: &Path, info: &VideoInfo, start: u32, count: u32) -> Result<Self, String> {
        let vf = format!(
            "scale=in_color_matrix={}:in_range={}:flags=accurate_rnd+full_chroma_int",
            info.matrix, info.range
        );
        let mut cmd = command(ffmpeg);
        cmd.arg("-v").arg("error").arg("-nostdin");
        if start > 0 {
            let t = (start as f64 - 0.5) / info.fps();
            cmd.arg("-ss").arg(format!("{:.6}", t));
        }
        cmd.arg("-i")
            .arg(&info.path)
            .arg("-an")
            .arg("-sn")
            .arg("-fps_mode")
            .arg("passthrough")
            .arg("-vf")
            .arg(&vf)
            .arg("-frames:v")
            .arg(count.to_string())
            .arg("-f")
            .arg("rawvideo")
            .arg("-pix_fmt")
            .arg("rgba")
            .arg("-")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(log_stdio("decode"));
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Could not run ffmpeg ({e}). Put ffmpeg.exe next to the program."))?;
        let out = child
            .stdout
            .take()
            .ok_or_else(|| "ffmpeg gave no output stream".to_string())?;
        Ok(Self {
            child,
            out,
            next_idx: start,
            remaining: count,
            fsize: info.frame_bytes(),
        })
    }

    /// Returns (frame_index, rgba bytes) or None at the end of the stream.
    pub fn read(&mut self) -> Result<Option<(u32, Vec<u8>)>, String> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let mut buf = vec![0u8; self.fsize];
        match self.out.read_exact(&mut buf) {
            Ok(()) => {
                let idx = self.next_idx;
                self.next_idx += 1;
                self.remaining -= 1;
                Ok(Some((idx, buf)))
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                self.remaining = 0;
                Ok(None)
            }
            Err(e) => Err(e.to_string()),
        }
    }
}

impl Drop for FrameReader {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ---------------------------------------------------------------------------
// Encoder: receives raw RGBA frames and writes the final file
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct EncodeOpts {
    pub codec: u32,   // 0 = H.264, 1 = H.265, 2 = ProRes 422 HQ
    pub quality: u32, // CRF (lower = better)
}

pub struct Encoder {
    child: Child,
    stdin: Option<ChildStdin>,
    log: PathBuf,
}

macro_rules! push {
    ($v:expr, $($x:expr),+ $(,)?) => { $( $v.push(String::from($x)); )+ };
}

impl Encoder {
    pub fn spawn(
        ffmpeg: &Path,
        info: &VideoInfo,
        out: &Path,
        start: u32,
        count: u32,
        o: &EncodeOpts,
    ) -> Result<Encoder, String> {
        let fps = info.fps();
        let t_in = format!("{:.6}", start as f64 / fps);
        let dur = format!("{:.6}", count as f64 / fps);
        let size = format!("{}x{}", info.width, info.height);
        let fr = format!("{}/{}", info.fps_num, info.fps_den);
        let pixfmt = if o.codec == 2 { "yuv422p10le" } else { "yuv420p" };
        let vf = format!(
            "scale=out_color_matrix={}:out_range={}:flags=accurate_rnd+full_chroma_int,format={}",
            info.matrix, info.range, pixfmt
        );
        let crf = o.quality.to_string();

        let mut a: Vec<String> = Vec::new();
        push!(a, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", &size, "-framerate", &fr, "-i", "-");
        push!(a, "-ss", &t_in, "-t", &dur, "-i");
        a.push(info.path.to_string_lossy().into_owned());
        push!(a, "-map", "0:v:0", "-map", "1:a:0?", "-vf", &vf);
        match o.codec {
            0 => {
                push!(a, "-c:v", "libx264", "-preset", "medium", "-crf", &crf);
            }
            1 => {
                push!(a, "-c:v", "libx265", "-preset", "medium", "-crf", &crf, "-tag:v", "hvc1");
            }
            _ => {
                push!(a, "-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0");
            }
        }
        if o.codec == 2 {
            push!(a, "-c:a", "pcm_s16le");
        } else {
            push!(a, "-c:a", "aac", "-b:a", "192k");
        }
        match info.matrix.as_str() {
            "bt709" => {
                push!(a, "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709");
            }
            "bt601" => {
                push!(a, "-colorspace", "smpte170m", "-color_primaries", "smpte170m", "-color_trc", "smpte170m");
            }
            _ => {}
        }
        push!(a, "-color_range", if info.range == "pc" { "pc" } else { "tv" });
        push!(a, "-movflags", "+faststart");
        a.push(out.to_string_lossy().into_owned());

        let log = log_path("encode");
        let mut child = command(ffmpeg)
            .args(&a)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(log_stdio("encode"))
            .spawn()
            .map_err(|e| format!("Could not run ffmpeg ({e}). Put ffmpeg.exe next to the program."))?;
        let stdin = child.stdin.take();
        Ok(Encoder { child, stdin, log })
    }

    pub fn write(&mut self, data: &[u8]) -> Result<(), String> {
        match self.stdin.as_mut() {
            Some(s) => s.write_all(data).map_err(|e| {
                format!("The encoder stopped ({e}). Details: {}", self.log.display())
            }),
            None => Err("encoder is closed".to_string()),
        }
    }

    pub fn finish(mut self) -> Result<(), String> {
        drop(self.stdin.take());
        let st = self.child.wait().map_err(|e| e.to_string())?;
        if st.success() {
            Ok(())
        } else {
            Err(format!(
                "ffmpeg finished with an error (code {:?}). Details: {}",
                st.code(),
                self.log.display()
            ))
        }
    }

    pub fn abort(mut self) {
        drop(self.stdin.take());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        drop(self.stdin.take());
        let _ = self.child.kill();
    }
}
