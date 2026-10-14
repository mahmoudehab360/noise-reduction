/// All user-facing filter settings. Thresholds use a 0..100 scale.
#[derive(Clone, PartialEq, Debug)]
pub struct Params {
    // ---- Temporal NR ----
    pub t_frames: u32, // frames on EACH side (0..=3)
    pub t_me: u32,     // 0 = Off (static only), 1 = Faster, 2 = Better
    pub t_range: u32,  // 0 = Small, 1 = Medium, 2 = Large
    pub t_luma: f32,
    pub t_chroma: f32,
    pub t_motion: f32,
    pub t_blend: f32,
    // ---- Spatial NR ----
    pub s_on: bool,
    pub s_mode: u32,   // 0 = Faster, 1 = Better, 2 = Enhanced
    pub s_radius: u32, // 0 = Small, 1 = Medium, 2 = Large
    pub s_luma: f32,
    pub s_chroma: f32,
    pub s_blend: f32,
    // ---- Extras ----
    pub detail: f32,  // detail recovery 0..100
    pub sharpen: f32, // 0..100
    pub grain: f32,   // 0..100
    pub grain_size: u32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            t_frames: 2,
            t_me: 2,
            t_range: 1,
            t_luma: 30.0,
            t_chroma: 40.0,
            t_motion: 40.0,
            t_blend: 100.0,
            s_on: true,
            s_mode: 1,
            s_radius: 1,
            s_luma: 25.0,
            s_chroma: 40.0,
            s_blend: 100.0,
            detail: 0.0,
            sharpen: 0.0,
            grain: 0.0,
            grain_size: 1,
        }
    }
}
