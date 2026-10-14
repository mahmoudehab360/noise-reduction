/// Estimates the noise level of an RGBA frame (Immerkaer's method on flat blocks).
/// Returns (luma, chroma) already converted to the 0..100 threshold scale used by the UI.
pub fn estimate(rgba: &[u8], w: usize, h: usize) -> (f32, f32) {
    if w < 40 || h < 40 || rgba.len() < w * h * 4 {
        return (25.0, 35.0);
    }
    let n = w * h;
    let mut y = vec![0f32; n];
    let mut cb = vec![0f32; n];
    let mut cr = vec![0f32; n];
    for i in 0..n {
        let r = rgba[i * 4] as f32 / 255.0;
        let g = rgba[i * 4 + 1] as f32 / 255.0;
        let b = rgba[i * 4 + 2] as f32 / 255.0;
        let l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        y[i] = l;
        cb[i] = (b - l) / 1.8556;
        cr[i] = (r - l) / 1.5748;
    }

    let bs = 32usize;
    let mut sy: Vec<f32> = Vec::new();
    let mut sc: Vec<f32> = Vec::new();
    let mut by = 0usize;
    while (by + 1) * bs <= h {
        let mut bx = 0usize;
        while (bx + 1) * bs <= w {
            let x0 = bx * bs;
            let y0 = by * bs;
            let mut ml = 0.0f32;
            for yy in y0..y0 + bs {
                for xx in x0..x0 + bs {
                    ml += y[yy * w + xx];
                }
            }
            ml /= (bs * bs) as f32;
            // skip crushed blacks / clipped whites: their noise is clipped
            if ml > 0.02 && ml < 0.98 {
                sy.push(lap_sigma(&y, w, x0, y0, bs));
                sc.push(0.5 * (lap_sigma(&cb, w, x0, y0, bs) + lap_sigma(&cr, w, x0, y0, bs)));
            }
            bx += 1;
        }
        by += 1;
    }
    let k = 0.0012f32; // threshold unit
    let a = (pick(sy) / k).round().clamp(1.0, 100.0);
    let b = (pick(sc) / k).round().clamp(1.0, 100.0);
    (a, b)
}

fn lap_sigma(p: &[f32], w: usize, x0: usize, y0: usize, bs: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut cnt = 0u32;
    for yy in (y0 + 1)..(y0 + bs - 1) {
        for xx in (x0 + 1)..(x0 + bs - 1) {
            let i = yy * w + xx;
            let v = p[i - w - 1] - 2.0 * p[i - w] + p[i - w + 1] - 2.0 * p[i - 1] + 4.0 * p[i]
                - 2.0 * p[i + 1]
                + p[i + w - 1]
                - 2.0 * p[i + w]
                + p[i + w + 1];
            sum += v.abs();
            cnt += 1;
        }
    }
    (sum / cnt.max(1) as f32) * 0.2089 // sqrt(pi/2) / 6
}

fn pick(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.03;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let k = (v.len() / 5).max(1); // flattest 20% of the blocks
    v[k / 2] * 1.15
}
