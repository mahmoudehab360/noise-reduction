// LowLight DN - compute kernels
// Pipeline per output frame:
//   k_cur -> k_down4 -> k_me_coarse -> k_me_refine -> k_temporal
//   -> k_nlm (in strips) -> k_chroma -> k_final

struct P {
  w: u32, h: u32, nslots: u32, cur: u32,
  w4: u32, h4: u32, cbw: u32, cbh: u32,
  fbw: u32, fbh: u32,
  range4: i32, refine_r: i32, spatial_r: i32, patch_r: i32,
  stride: i32, chroma_r: i32, cstride: i32,
  me_on: u32, spatial_on: u32, frame_idx: u32, grain_size: u32, row0: u32,
  sig_t_l: f32, sig_t_c: f32, mot_scale: f32, t_blend: f32,
  sig_s_l: f32, sig_s_c: f32, s_blend: f32, detail: f32,
  sharpen: f32, grain: f32,
}

@group(0) @binding(0)  var<uniform> u: P;
@group(0) @binding(1)  var<storage, read> frames: array<u32>;
@group(0) @binding(2)  var<storage, read_write> cur_ycc: array<vec4<f32>>;
@group(0) @binding(3)  var<storage, read_write> luma4: array<f32>;
@group(0) @binding(4)  var<storage, read_write> mvc: array<i32>;
@group(0) @binding(5)  var<storage, read_write> mv: array<i32>;
@group(0) @binding(6)  var<storage, read_write> sad: array<f32>;
@group(0) @binding(7)  var<storage, read_write> ycc_t: array<vec4<f32>>;
@group(0) @binding(8)  var<storage, read_write> yb_t: array<f32>;
@group(0) @binding(9)  var<storage, read_write> yb_s: array<f32>;
@group(0) @binding(10) var<storage, read_write> ycc_s: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> out_px: array<u32>;

// ---------------------------------------------------------------- helpers

fn unpack_rgb(v: u32) -> vec3<f32> {
  return vec3<f32>(f32(v & 255u), f32((v >> 8u) & 255u), f32((v >> 16u) & 255u)) / 255.0;
}

fn rgb2ycc(c: vec3<f32>) -> vec3<f32> {
  let y = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
  return vec3<f32>(y, (c.b - y) / 1.8556, (c.r - y) / 1.5748);
}

fn ycc2rgb(c: vec3<f32>) -> vec3<f32> {
  let r = c.x + 1.5748 * c.z;
  let b = c.x + 1.8556 * c.y;
  let g = (c.x - 0.2126 * r - 0.0722 * b) / 0.7152;
  return vec3<f32>(r, g, b);
}

fn px_raw(slot: u32, x: i32, y: i32) -> u32 {
  let cx = clamp(x, 0, i32(u.w) - 1);
  let cy = clamp(y, 0, i32(u.h) - 1);
  return frames[slot * u.w * u.h + u32(cy) * u.w + u32(cx)];
}

fn px_ycc(slot: u32, x: i32, y: i32) -> vec3<f32> {
  return rgb2ycc(unpack_rgb(px_raw(slot, x, y)));
}

fn px_luma(slot: u32, x: i32, y: i32) -> f32 {
  return dot(unpack_rgb(px_raw(slot, x, y)), vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn cur_at(x: i32, y: i32) -> vec3<f32> {
  let cx = clamp(x, 0, i32(u.w) - 1);
  let cy = clamp(y, 0, i32(u.h) - 1);
  return cur_ycc[u32(cy) * u.w + u32(cx)].xyz;
}

fn yt_at(x: i32, y: i32) -> f32 {
  let cx = clamp(x, 0, i32(u.w) - 1);
  let cy = clamp(y, 0, i32(u.h) - 1);
  return yb_t[u32(cy) * u.w + u32(cx)];
}

fn ys_at(x: i32, y: i32) -> f32 {
  let cx = clamp(x, 0, i32(u.w) - 1);
  let cy = clamp(y, 0, i32(u.h) - 1);
  return yb_s[u32(cy) * u.w + u32(cx)];
}

fn hash_u(v: u32) -> u32 {
  var x = v;
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  x = x ^ (x >> 16u);
  return x;
}

fn rnd(a: u32, b: u32, c: u32) -> f32 {
  return f32(hash_u(a ^ hash_u(b ^ hash_u(c)))) / 4294967295.0;
}

// ---------------------------------------------------------------- kernels

@compute @workgroup_size(16, 16, 1)
fn k_cur(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= u.w || g.y >= u.h) { return; }
  let i = g.y * u.w + g.x;
  let c = rgb2ycc(unpack_rgb(frames[u.cur * u.w * u.h + i]));
  cur_ycc[i] = vec4<f32>(c, 1.0);
}

// 4x4 box-downsampled luma of every frame in the window (robust to noise)
@compute @workgroup_size(16, 16, 1)
fn k_down4(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= u.w4 || g.y >= u.h4 || g.z >= u.nslots) { return; }
  var s = 0.0;
  for (var j = 0; j < 4; j = j + 1) {
    for (var i = 0; i < 4; i = i + 1) {
      s = s + px_luma(g.z, i32(g.x) * 4 + i, i32(g.y) * 4 + j);
    }
  }
  luma4[g.z * u.w4 * u.h4 + g.y * u.w4 + g.x] = s / 16.0;
}

// coarse full-search block matching on the 1/4-resolution luma (8x8 blocks = 32x32 px)
@compute @workgroup_size(8, 8, 1)
fn k_me_coarse(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= u.cbw || g.y >= u.cbh || g.z >= u.nslots) { return; }
  let slot = g.z;
  let bi = (slot * u.cbw * u.cbh + g.y * u.cbw + g.x) * 2u;
  if (slot == u.cur) {
    mvc[bi] = 0;
    mvc[bi + 1u] = 0;
    return;
  }
  let n4 = u.w4 * u.h4;
  let x0 = i32(g.x) * 8;
  let y0 = i32(g.y) * 8;
  let r = u.range4;
  var best = 1e30;
  var bx = 0;
  var by = 0;
  for (var dy = -r; dy <= r; dy = dy + 1) {
    for (var dx = -r; dx <= r; dx = dx + 1) {
      var sum = 0.0;
      for (var j = 0; j < 8; j = j + 1) {
        let yy = y0 + j;
        if (yy >= i32(u.h4)) { break; }
        for (var i = 0; i < 8; i = i + 1) {
          let xx = x0 + i;
          if (xx >= i32(u.w4)) { break; }
          let a = luma4[u.cur * n4 + u32(yy) * u.w4 + u32(xx)];
          let sx = clamp(xx + dx, 0, i32(u.w4) - 1);
          let sy = clamp(yy + dy, 0, i32(u.h4) - 1);
          let b = luma4[slot * n4 + u32(sy) * u.w4 + u32(sx)];
          sum = sum + abs(a - b);
        }
      }
      let cost = sum * (1.0 + 0.02 * f32(abs(dx) + abs(dy)));
      if (cost < best) {
        best = cost;
        bx = dx;
        by = dy;
      }
    }
  }
  mvc[bi] = bx * 4;
  mvc[bi + 1u] = by * 4;
}

fn block_sad(slot: u32, x0: i32, y0: i32, mx: i32, my: i32) -> f32 {
  var sum = 0.0;
  var cnt = 0.0;
  for (var j = 0; j < 16; j = j + 2) {
    let yy = y0 + j;
    if (yy >= i32(u.h)) { break; }
    for (var i = 0; i < 16; i = i + 2) {
      let xx = x0 + i;
      if (xx >= i32(u.w)) { break; }
      let a = px_luma(u.cur, xx, yy);
      let b = px_luma(slot, xx + mx, yy + my);
      sum = sum + abs(a - b);
      cnt = cnt + 1.0;
    }
  }
  return sum / max(cnt, 1.0);
}

// refinement on full-res 16x16 blocks around the coarse vector.
// also stores the mean absolute luma error (used as a motion confidence)
@compute @workgroup_size(8, 8, 1)
fn k_me_refine(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= u.fbw || g.y >= u.fbh || g.z >= u.nslots) { return; }
  let slot = g.z;
  let bidx = slot * u.fbw * u.fbh + g.y * u.fbw + g.x;
  let bi = bidx * 2u;
  let x0 = i32(g.x) * 16;
  let y0 = i32(g.y) * 16;
  if (slot == u.cur) {
    mv[bi] = 0;
    mv[bi + 1u] = 0;
    sad[bidx] = 0.0;
    return;
  }
  if (u.me_on == 0u) {
    mv[bi] = 0;
    mv[bi + 1u] = 0;
    sad[bidx] = block_sad(slot, x0, y0, 0, 0);
    return;
  }
  let cx = min(g.x / 2u, u.cbw - 1u);
  let cy = min(g.y / 2u, u.cbh - 1u);
  let ci = (slot * u.cbw * u.cbh + cy * u.cbw + cx) * 2u;
  let ix = mvc[ci];
  let iy = mvc[ci + 1u];
  let r = u.refine_r;
  var bsad = block_sad(slot, x0, y0, 0, 0);
  var best = bsad;
  var bx = 0;
  var by = 0;
  for (var dy = -r; dy <= r; dy = dy + 1) {
    for (var dx = -r; dx <= r; dx = dx + 1) {
      let mx = ix + dx;
      let my = iy + dy;
      let sd = block_sad(slot, x0, y0, mx, my);
      let cost = sd * (1.0 + 0.004 * f32(abs(mx) + abs(my)));
      if (cost < best) {
        best = cost;
        bsad = sd;
        bx = mx;
        by = my;
      }
    }
  }
  mv[bi] = bx;
  mv[bi + 1u] = by;
  sad[bidx] = bsad;
}

fn mv_at(base: u32, bx: i32, by: i32) -> vec2<f32> {
  let b = (base + u32(by) * u.fbw + u32(bx)) * 2u;
  return vec2<f32>(f32(mv[b]), f32(mv[b + 1u]));
}

// motion-compensated temporal denoise
@compute @workgroup_size(16, 16, 1)
fn k_temporal(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= u.w || g.y >= u.h) { return; }
  let x = i32(g.x);
  let y = i32(g.y);
  let i = g.y * u.w + g.x;
  let c = cur_ycc[i].xyz;
  var acc = c;
  var wsum = 1.0;
  let sl = u.sig_t_l;
  let sc = u.sig_t_c;
  if (u.nslots > 1u && (sl > 0.0 || sc > 0.0)) {
    let fx = (f32(x) + 0.5) / 16.0 - 0.5;
    let fy = (f32(y) + 0.5) / 16.0 - 0.5;
    let flx = floor(fx);
    let fly = floor(fy);
    let tx = fx - flx;
    let ty = fy - fly;
    let maxbx = i32(u.fbw) - 1;
    let maxby = i32(u.fbh) - 1;
    let xa = clamp(i32(flx), 0, maxbx);
    let xb = clamp(i32(flx) + 1, 0, maxbx);
    let ya = clamp(i32(fly), 0, maxby);
    let yb = clamp(i32(fly) + 1, 0, maxby);
    let xn = clamp(x / 16, 0, maxbx);
    let yn = clamp(y / 16, 0, maxby);
    for (var s = 0u; s < u.nslots; s = s + 1u) {
      if (s == u.cur) { continue; }
      let base = s * u.fbw * u.fbh;
      let m = mix(
        mix(mv_at(base, xa, ya), mv_at(base, xb, ya), tx),
        mix(mv_at(base, xa, yb), mv_at(base, xb, yb), tx),
        ty);
      let mx = i32(round(m.x));
      let my = i32(round(m.y));
      let sdv = sad[base + u32(yn) * u.fbw + u32(xn)];
      let conf = exp(-max(sdv - 1.7 * sl, 0.0) / u.mot_scale);
      var ssl = 0.0;
      var ssc = 0.0;
      for (var j = -1; j <= 1; j = j + 1) {
        for (var k = -1; k <= 1; k = k + 1) {
          let a = cur_at(x + k, y + j);
          let b = px_ycc(s, x + k + mx, y + j + my);
          let d = b - a;
          ssl = ssl + d.x * d.x;
          ssc = ssc + 0.5 * (d.y * d.y + d.z * d.z);
        }
      }
      ssl = ssl / 9.0;
      ssc = ssc / 9.0;
      let el = max(ssl - 2.0 * sl * sl, 0.0);
      let ec = max(ssc - 2.0 * sc * sc, 0.0);
      let dist = f32(abs(i32(s) - i32(u.cur)));
      let w = exp(-el / (2.83 * sl * sl + 1e-9) - ec / (2.83 * sc * sc + 1e-9)) * conf * pow(0.8, dist - 1.0);
      acc = acc + w * px_ycc(s, x + mx, y + my);
      wsum = wsum + w;
    }
  }
  var t = acc / wsum;
  t = mix(c, t, u.t_blend);
  ycc_t[i] = vec4<f32>(t, 1.0);
  yb_t[i] = t.x;
}

// spatial luma denoise: non-local means (patch-based). Dispatched in row strips.
@compute @workgroup_size(16, 16, 1)
fn k_nlm(@builtin(global_invocation_id) g: vec3<u32>) {
  let gy = g.y + u.row0;
  if (g.x >= u.w || gy >= u.h) { return; }
  let i = gy * u.w + g.x;
  if (u.spatial_on == 0u || u.sig_s_l <= 0.0) {
    yb_s[i] = yb_t[i];
    return;
  }
  let x = i32(g.x);
  let y = i32(gy);
  let sg = u.sig_s_l;
  let pr = u.patch_r;
  let npatch = f32((2 * pr + 1) * (2 * pr + 1));
  let cc = 6.0 * sg * sg * sqrt(2.0 / npatch);
  var acc = yb_t[i];
  var wsum = 1.0;
  let r = u.spatial_r;
  for (var dy = -r; dy <= r; dy = dy + u.stride) {
    for (var dx = -r; dx <= r; dx = dx + u.stride) {
      if (dx == 0 && dy == 0) { continue; }
      var d = 0.0;
      for (var j = -pr; j <= pr; j = j + 1) {
        for (var k = -pr; k <= pr; k = k + 1) {
          let a = yt_at(x + k, y + j);
          let b = yt_at(x + dx + k, y + dy + j);
          d = d + (a - b) * (a - b);
        }
      }
      d = d / npatch;
      let w = exp(-max(d - 2.0 * sg * sg, 0.0) / cc);
      acc = acc + w * yt_at(x + dx, y + dy);
      wsum = wsum + w;
    }
  }
  yb_s[i] = acc / wsum;
}

// spatial chroma denoise: wide bilateral guided by the denoised luma
@compute @workgroup_size(16, 16, 1)
fn k_chroma(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= u.w || g.y >= u.h) { return; }
  let x = i32(g.x);
  let y = i32(g.y);
  let i = g.y * u.w + g.x;
  let c = ycc_t[i].xyz;
  let ys = yb_s[i];
  var res = c.yz;
  let sc = u.sig_s_c;
  if (u.spatial_on != 0u && sc > 0.0) {
    let r = u.chroma_r;
    let ysig = max(2.0 * u.sig_s_l, 0.02);
    let rs = f32(r) * 0.5 + 0.5;
    var acc = c.yz;
    var wsum = 1.0;
    for (var dy = -r; dy <= r; dy = dy + u.cstride) {
      for (var dx = -r; dx <= r; dx = dx + u.cstride) {
        if (dx == 0 && dy == 0) { continue; }
        let cx = clamp(x + dx, 0, i32(u.w) - 1);
        let cy = clamp(y + dy, 0, i32(u.h) - 1);
        let qi = u32(cy) * u.w + u32(cx);
        let q = ycc_t[qi];
        let dY = yb_s[qi] - ys;
        let dC = q.yz - c.yz;
        let wsp = exp(-f32(dx * dx + dy * dy) / (2.0 * rs * rs));
        let wy = exp(-(dY * dY) / (2.0 * ysig * ysig));
        let wc = exp(-max(dot(dC, dC) - 4.0 * sc * sc, 0.0) / (6.0 * sc * sc));
        let w = wsp * wy * wc;
        acc = acc + w * q.yz;
        wsum = wsum + w;
      }
    }
    res = acc / wsum;
  }
  ycc_s[i] = vec4<f32>(ys, res, 1.0);
}

// blend, detail recovery, sharpen, grain, dither, pack to RGBA8
@compute @workgroup_size(16, 16, 1)
fn k_final(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= u.w || g.y >= u.h) { return; }
  let x = i32(g.x);
  let y = i32(g.y);
  let i = g.y * u.w + g.x;
  let c0 = ycc_t[i].xyz;
  let s1 = ycc_s[i].xyz;
  var o = mix(c0, s1, u.s_blend);

  // detail recovery: give back part of what spatial NR removed where it looks like real detail
  let rem = yb_t[i] - yb_s[i];
  let sg = max(u.sig_s_l, 0.0001);
  o.x = o.x + u.detail * rem * smoothstep(1.2 * sg, 3.0 * sg, abs(rem));

  if (u.sharpen > 0.0) {
    let cc = ys_at(x, y);
    let bl = (ys_at(x - 1, y - 1) + 2.0 * ys_at(x, y - 1) + ys_at(x + 1, y - 1)
            + 2.0 * ys_at(x - 1, y) + 4.0 * cc + 2.0 * ys_at(x + 1, y)
            + ys_at(x - 1, y + 1) + 2.0 * ys_at(x, y + 1) + ys_at(x + 1, y + 1)) / 16.0;
    o.x = o.x + u.sharpen * (cc - bl);
  }

  if (u.grain > 0.0) {
    let gs = max(u.grain_size, 1u);
    let gx = g.x / gs;
    let gy = g.y / gs;
    let n = rnd(gx * 7919u + gy * 104729u, u.frame_idx, 1u) + rnd(gx * 15485863u + gy, u.frame_idx, 2u) - 1.0;
    o.x = o.x + u.grain * n;
  }

  var rgb = ycc2rgb(o);
  // tiny triangular dither: avoids banding in smooth, heavily denoised shadows
  let dth = (rnd(g.x + g.y * u.w, u.frame_idx, 3u) + rnd(g.y * 31u + g.x, u.frame_idx, 4u) - 1.0) * (0.7 / 255.0);
  rgb = clamp(rgb + vec3<f32>(dth), vec3<f32>(0.0), vec3<f32>(1.0));
  let q = vec3<u32>(rgb * 255.0 + vec3<f32>(0.5));
  out_px[i] = q.x | (q.y << 8u) | (q.z << 16u) | (255u << 24u);
}
