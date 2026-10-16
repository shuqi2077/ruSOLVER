use super::complex_qr_kernel::combine_norm;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda]
fn block() -> usize { RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize }

#[ruda]
fn pair(n: usize, round: usize, index: usize) -> (usize, usize) {
    let slots = n + n % 2;
    let p = (round + index) % (slots - 1); let mut q = slots - 1;
    if index != 0 { q = (round + slots - 1 - index) % (slots - 1); }
    (p, q)
}

#[ruda]
fn bad<F: Float>(x: F) -> bool { x != x || x.abs() == F::cast_from(f32::INFINITY) }

#[ruda]
fn maximum<F: Float>(x: F) -> F {
    let lane = UNIT_POS as usize; let mut work = SharedMemory::<F>::new(256usize);
    work[lane] = x; sync_ruda(); let mut width = 128usize;
    while width > 0 {
        if lane < width { work[lane] = F::max(work[lane], work[lane + width]); }
        sync_ruda(); width /= 2;
    }
    let result = work[0usize]; sync_ruda(); result
}

#[ruda]
fn sum<F: Float>(x: F) -> F {
    let lane = UNIT_POS as usize; let mut work = SharedMemory::<F>::new(256usize);
    work[lane] = x; sync_ruda(); let mut width = 128usize;
    while width > 0 {
        if lane < width { work[lane] = work[lane] + work[lane + width]; }
        sync_ruda(); width /= 2;
    }
    let result = work[0usize]; sync_ruda(); result
}

#[ruda(launch)]
pub(super) fn prepare<F: Float>(
    ar: &Tensor<F>, ai: &Tensor<F>, real: &mut Array<F>, imag: &mut Array<F>,
    info: &mut Array<i32>, sweeps: &mut Array<i32>, rows: u32, cols: u32,
    #[comptime] hermitian: bool, #[comptime] upper: bool, #[comptime] adjoint: bool,
) {
    let batch = block(); if batch >= info.len() { terminate!(); }
    let lane = UNIT_POS as usize; let m = rows as usize; let n = cols as usize;
    let mut i = lane; let mut invalid = F::new(0.0);
    while i < m * n {
        let row = i / n; let col = i % n; let mut sr = row; let mut sc = col; let mut conjugate = false;
        if comptime!(hermitian) {
            if upper && row > col || !upper && row < col { sr = col; sc = row; conjugate = true; }
        } else if comptime!(adjoint) { sr = col; sc = row; conjugate = true; }
        let r = ar[batch * ar.stride(0) + sr * ar.stride(1) + sc * ar.stride(2)];
        let mut im = F::new(0.0);
        if !hermitian || row != col { im = ai[batch * ai.stride(0) + sr * ai.stride(1) + sc * ai.stride(2)]; }
        if conjugate { im = -im; }
        real[batch * m * n + i] = r; imag[batch * m * n + i] = im;
        if bad::<F>(r) || bad::<F>(im) { invalid = F::new(1.0); } i += 256;
    }
    let invalid = maximum::<F>(invalid);
    if lane == 0 {
        let mut code = 1i32; if m == 0 || n == 0 { code = 0; } if invalid != F::new(0.0) { code = -1; }
        info[batch] = code; sweeps[batch] = 0;
    }
}

#[ruda(launch)]
pub(super) fn eigen_angles<F: Float>(
    real: &Array<F>, imag: &Array<F>, angles: &mut Array<F>, info: &Array<i32>, order: u32, round: u32,
) {
    let i = block() * 256usize + UNIT_POS as usize; let n = order as usize; let pairs = (n + n % 2) / 2;
    if i >= info.len() * pairs { terminate!(); }
    let batch = i / pairs; let (p, q) = pair(n, round as usize, i % pairs);
    let mut c = F::new(1.0); let mut sr = F::new(0.0); let mut si = F::new(0.0);
    if info[batch] == 1 && p < n && q < n {
        let base = batch * n * n; let a = real[base + p * n + p]; let b = real[base + q * n + q];
        let gr = real[base + p * n + q]; let gi = imag[base + p * n + q];
        let scale = F::max(F::max(a.abs(), b.abs()), F::max(gr.abs(), gi.abs()));
        if scale > F::new(0.0) {
            let r = gr / scale; let im = gi / scale; let g = r.hypot(im);
            if g > F::new(0.0) {
                let delta = b / scale - a / scale; let root = delta.hypot(F::new(2.0) * g);
                let mut t = F::new(1.0);
                if delta > F::new(0.0) { t = F::new(2.0) * g / (delta + root); }
                else if delta < F::new(0.0) { t = F::new(2.0) * g / (delta - root); }
                c = F::new(1.0) / (F::new(1.0) + t * t).sqrt();
                sr = t * c * (r / g); si = t * c * (im / g);
            }
        }
    }
    angles[3 * i] = c; angles[3 * i + 1] = sr; angles[3 * i + 2] = si;
}

#[ruda(launch)]
pub(super) fn eigen_rotate<F: Float>(
    real: &mut Array<F>, imag: &mut Array<F>, vr: &mut Array<F>, vi: &mut Array<F>,
    angles: &Array<F>, info: &Array<i32>, order: u32, round: u32, #[comptime] left: bool, #[comptime] vectors: bool,
) {
    let i = block() * 256usize + UNIT_POS as usize; let n = order as usize; let pairs = (n + n % 2) / 2;
    if i >= info.len() * pairs * n { terminate!(); }
    let batch = i / (pairs * n); if info[batch] != 1 { terminate!(); }
    let t = i / n % pairs; let row = i % n; let (p, q) = pair(n, round as usize, t);
    if p >= n || q >= n { terminate!(); }
    let offset = (batch * pairs + t) * 3; let c = angles[offset]; let sr = angles[offset + 1]; let si = angles[offset + 2];
    let base = batch * n * n; let mut x = base + row * n + p; let mut y = base + row * n + q;
    if comptime!(left) { x = base + p * n + row; y = base + q * n + row; }
    let xr = real[x]; let xi = imag[x]; let yr = real[y]; let yi = imag[y];
    if comptime!(left) {
        real[x] = c * xr - sr * yr + si * yi; imag[x] = c * xi - sr * yi - si * yr;
        real[y] = sr * xr + si * xi + c * yr; imag[y] = sr * xi - si * xr + c * yi;
    } else {
        real[x] = c * xr - sr * yr - si * yi; imag[x] = c * xi - sr * yi + si * yr;
        real[y] = sr * xr - si * xi + c * yr; imag[y] = sr * xi + si * xr + c * yi;
        if row == p { imag[x] = F::new(0.0); real[y] = F::new(0.0); imag[y] = F::new(0.0); }
        if row == q { imag[y] = F::new(0.0); real[x] = F::new(0.0); imag[x] = F::new(0.0); }
        if comptime!(vectors) {
            let xr = vr[x]; let xi = vi[x]; let yr = vr[y]; let yi = vi[y];
            vr[x] = c * xr - sr * yr - si * yi; vi[x] = c * xi - sr * yi + si * yr;
            vr[y] = sr * xr - si * xi + c * yr; vi[y] = sr * xi + si * xr + c * yi;
        }
    }
}

#[ruda(launch)]
pub(super) fn eigen_check<F: Float + RudaElement>(
    real: &Array<F>, imag: &Array<F>, info: &mut Array<i32>, sweeps: &mut Array<i32>,
    order: u32, sweep: u32, atol: F, rtol: F,
) {
    let batch = block(); if batch >= info.len() { terminate!(); } if info[batch] != 1 { terminate!(); }
    let lane = UNIT_POS as usize; let n = order as usize; let base = batch * n * n;
    let mut i = lane; let mut scale = F::new(0.0); let mut invalid = F::new(0.0);
    while i < n * n {
        let r = real[base + i]; let im = imag[base + i];
        scale = F::max(scale, F::max(r.abs(), im.abs())); if bad::<F>(r) || bad::<F>(im) { invalid = F::new(1.0); } i += 256;
    }
    let scale = maximum::<F>(scale); let invalid = maximum::<F>(invalid);
    let mut off = F::new(0.0); let mut all = F::new(0.0);
    if scale > F::new(0.0) && invalid == F::new(0.0) {
        i = lane;
        while i < n * n {
            let value = (real[base + i] / scale).hypot(imag[base + i] / scale);
            all = F::max(all, value); if i / n != i % n { off = F::max(off, value); } i += 256;
        }
    }
    let off = maximum::<F>(off); let all = maximum::<F>(all);
    if lane == 0 {
        let mut code = 1i32;
        if invalid != F::new(0.0) { code = -3; }
        else if scale == F::new(0.0) || off <= atol / scale + rtol * all { code = 0; }
        info[batch] = code; sweeps[batch] = sweep as i32;
    }
}

#[ruda(launch)]
pub(super) fn svd_pair<F: Float + RudaElement>(
    real: &mut Array<F>, imag: &mut Array<F>, vr: &mut Array<F>, vi: &mut Array<F>,
    flags: &mut Array<i32>, info: &Array<i32>, rows: u32, cols: u32, round: u32,
    atol: F, rtol: F, #[comptime] apply: bool, #[comptime] vectors: bool,
) {
    let work = block(); let m = rows as usize; let n = cols as usize; let pairs = (n + n % 2) / 2;
    if work >= info.len() * pairs { terminate!(); }
    let lane = UNIT_POS as usize; let batch = work / pairs; let rounds = n + n % 2 - 1;
    let flag = (batch * rounds + round as usize) * pairs + work % pairs;
    let (pp, qq) = pair(n, round as usize, work % pairs); let mut p = pp; let mut q = qq;
    if info[batch] != 1 || p >= n || q >= n {
        if !apply && lane == 0 { flags[flag] = 0; } terminate!();
    }
    let base = batch * m * n; let mut sp = F::new(0.0); let mut sq = F::new(0.0); let mut invalid = F::new(0.0);
    let mut row = lane;
    while row < m {
        let pr = real[base + row * n + p]; let pi = imag[base + row * n + p];
        let qr = real[base + row * n + q]; let qi = imag[base + row * n + q];
        sp = F::max(sp, F::max(pr.abs(), pi.abs())); sq = F::max(sq, F::max(qr.abs(), qi.abs()));
        if bad::<F>(pr) || bad::<F>(pi) || bad::<F>(qr) || bad::<F>(qi) { invalid = F::new(1.0); } row += 256;
    }
    let mut sp = maximum::<F>(sp); let mut sq = maximum::<F>(sq); let invalid = maximum::<F>(invalid);
    if invalid != F::new(0.0) || sp == F::new(0.0) || sq == F::new(0.0) {
        if !apply && lane == 0 { let mut code = 0; if invalid != F::new(0.0) { code = -3; } flags[flag] = code; }
        terminate!();
    }
    let mut a = F::new(0.0); let mut b = F::new(0.0); let mut gr = F::new(0.0); let mut gi = F::new(0.0); row = lane;
    while row < m {
        let pr = real[base + row * n + p] / sp; let pi = imag[base + row * n + p] / sp;
        let qr = real[base + row * n + q] / sq; let qi = imag[base + row * n + q] / sq;
        a += pr * pr + pi * pi; b += qr * qr + qi * qi; gr += pr * qr + pi * qi; gi += pr * qi - pi * qr;
        row += 256;
    }
    let mut a = sum::<F>(a).sqrt(); let mut b = sum::<F>(b).sqrt(); let gr = sum::<F>(gr); let mut gi = sum::<F>(gi);
    let g = gr.hypot(gi); let rho = g / a / b;
    let bound = rtol + atol / sp / sq / a / b;
    if !apply {
        if lane == 0 { let mut code = 0; if rho > bound { code = 1; } if bad::<F>(rho) { code = -3; } flags[flag] = code; }
        terminate!();
    }
    if rho <= bound || bad::<F>(rho) { terminate!(); }
    let scale = F::max(sp, sq);
    if sp / scale * a < sq / scale * b {
        p = qq; q = pp; let temporary = sp; sp = sq; sq = temporary; let temporary = a; a = b; b = temporary; gi = -gi;
    }
    let ratio = sq / sp * (b / a); let delta = F::new(1.0) - ratio * ratio;
    let t_over_ratio = -F::new(2.0) * rho / (delta + delta.hypot(F::new(2.0) * rho * ratio));
    let t = t_over_ratio * ratio; let c = F::new(1.0) / (F::new(1.0) + t * t).sqrt();
    let sr = gr / g; let si = gi / g; let sine = t * c;
    let small_into_large = sine * (sq / sp); let large_into_small = t_over_ratio * c * (b / a);
    row = lane;
    while row < m {
        let x = base + row * n + p; let y = base + row * n + q;
        let xr = real[x] / sp; let xi = imag[x] / sp; let yr = real[y] / sq; let yi = imag[y] / sq;
        real[x] = sp * (c * xr - small_into_large * (sr * yr + si * yi));
        imag[x] = sp * (c * xi - small_into_large * (sr * yi - si * yr));
        real[y] = sq * (large_into_small * (sr * xr - si * xi) + c * yr);
        imag[y] = sq * (large_into_small * (sr * xi + si * xr) + c * yi); row += 256;
    }
    if comptime!(vectors) {
        row = lane;
        while row < n {
            let x = (batch * n + row) * n + p; let y = (batch * n + row) * n + q;
            let xr = vr[x]; let xi = vi[x]; let yr = vr[y]; let yi = vi[y];
            vr[x] = c * xr - sine * (sr * yr + si * yi); vi[x] = c * xi - sine * (sr * yi - si * yr);
            vr[y] = sine * (sr * xr - si * xi) + c * yr; vi[y] = sine * (sr * xi + si * xr) + c * yi; row += 256;
        }
    }
}

#[ruda(launch)]
pub(super) fn svd_check<F: Float>(
    real: &Array<F>, imag: &Array<F>, flags: &Array<i32>, info: &mut Array<i32>, sweeps: &mut Array<i32>,
    rows: u32, cols: u32, sweep: u32,
) {
    let batch = block(); if batch >= info.len() { terminate!(); } if info[batch] != 1 { terminate!(); }
    let lane = UNIT_POS as usize; let m = rows as usize; let n = cols as usize;
    let count = (n + n % 2 - 1) * ((n + n % 2) / 2); let mut i = lane; let mut invalid = F::new(0.0); let mut active = F::new(0.0);
    while i < count { let code = flags[batch * count + i]; if code < 0 { invalid = F::new(1.0); } if code > 0 { active = F::new(1.0); } i += 256; }
    i = lane;
    while i < m * n { if bad::<F>(real[batch * m * n + i]) || bad::<F>(imag[batch * m * n + i]) { invalid = F::new(1.0); } i += 256; }
    let invalid = maximum::<F>(invalid); let active = maximum::<F>(active);
    if lane == 0 { let mut code = 0; if active != F::new(0.0) { code = 1; } if invalid != F::new(0.0) { code = -3; }
        info[batch] = code; sweeps[batch] = sweep as i32; }
}

#[ruda(launch)]
pub(super) fn seal(info: &mut Array<i32>) {
    let i = block() * 256usize + UNIT_POS as usize; if i < info.len() && info[i] == 1 { info[i] = -4; }
}

#[ruda(launch)]
pub(super) fn eigen_values<F: Float>(real: &Array<F>, values: &mut Array<F>, info: &Array<i32>, order: u32) {
    let i = block() * 256usize + UNIT_POS as usize; let n = order as usize;
    if i < values.len() { let batch = i / n; let j = i % n; let mut value = F::new(0.0);
        if info[batch] == 0 { value = real[(batch * n + j) * n + j]; } values[i] = value; }
}

#[ruda(launch)]
pub(super) fn svd_values<F: Float>(real: &mut Array<F>, imag: &mut Array<F>, values: &mut Array<F>, info: &Array<i32>, rows: u32, cols: u32) {
    let work = block(); if work >= values.len() { terminate!(); }
    let lane = UNIT_POS as usize; let m = rows as usize; let n = cols as usize; let batch = work / n; let col = work % n;
    let mut scale = F::new(0.0); let mut norm = F::new(0.0); let mut row = lane;
    if info[batch] == 0 {
        while row < m {
            let p = (batch * m + row) * n + col;
            let (s, q) = combine_norm::<F>(scale, norm, real[p].abs(), F::new(1.0));
            let (s, q) = combine_norm::<F>(s, q, imag[p].abs(), F::new(1.0)); scale = s; norm = q; row += 256;
        }
    }
    let mut scales = SharedMemory::<F>::new(256usize); let mut norms = SharedMemory::<F>::new(256usize);
    scales[lane] = scale; norms[lane] = norm; sync_ruda(); let mut width = 128usize;
    while width > 0 {
        if lane < width { let (s, q) = combine_norm::<F>(scales[lane], norms[lane], scales[lane + width], norms[lane + width]); scales[lane] = s; norms[lane] = q; }
        sync_ruda(); width /= 2;
    }
    scale = scales[0usize]; norm = norms[0usize].sqrt(); if lane == 0 { values[work] = scale * norm; }
    row = lane;
    while row < m {
        let p = (batch * m + row) * n + col; let mut r = F::new(0.0); let mut i = F::new(0.0);
        if scale != F::new(0.0) { r = real[p] / scale / norm; i = imag[p] / scale / norm; }
        real[p] = r; imag[p] = i; row += 256;
    }
}

#[ruda(launch)]
pub(super) fn sort_values<F: Float>(values: &Array<F>, sorted: &mut Array<F>, permutation: &mut Array<u32>, order: u32, #[comptime] descending: bool) {
    let i = block() * 256usize + UNIT_POS as usize; if i >= values.len() { terminate!(); }
    let n = order as usize; let batch = i / n; let col = i % n; let value = values[i]; let mut rank = 0usize;
    for j in 0..n {
        let other = values[batch * n + j]; let mut before = other < value;
        if comptime!(descending) { before = other > value; }
        if before || other == value && j < col { rank += 1; }
    }
    sorted[batch * n + rank] = value; permutation[batch * n + rank] = col as u32;
}

#[ruda(launch)]
pub(super) fn gather<F: Float>(ar: &Array<F>, ai: &Array<F>, real: &mut Array<F>, imag: &mut Array<F>, permutation: &Array<u32>, info: &Array<i32>, rows: u32, cols: u32) {
    let i = block() * 256usize + UNIT_POS as usize; if i >= real.len() { terminate!(); }
    let m = rows as usize; let n = cols as usize; let batch = i / (m * n); let row = i / n % m; let col = i % n;
    let mut r = F::new(0.0); let mut im = F::new(0.0);
    if info[batch] == 0 { let p = (batch * m + row) * n + permutation[batch * n + col] as usize; r = ar[p]; im = ai[p]; }
    real[i] = r; imag[i] = im;
}

#[ruda(launch)]
pub(super) fn complete<F: Float>(ar: &Array<F>, ai: &Array<F>, real: &mut Array<F>, imag: &mut Array<F>, values: &Array<F>, info: &Array<i32>, rows: u32, cols: u32, output_cols: u32) {
    let i = block() * 256usize + UNIT_POS as usize; if i >= real.len() { terminate!(); }
    let m = rows as usize; let n = cols as usize; let width = output_cols as usize; let batch = i / (m * width); let row = i / width % m; let col = i % width;
    if info[batch] != 0 { real[i] = F::new(0.0); imag[i] = F::new(0.0); }
    else if col < n && values[batch * n + col] != F::new(0.0) { let p = (batch * m + row) * n + col; real[i] = ar[p]; imag[i] = ai[p]; }
}
