// Adapts LAPACK 3.12.1 ZGEBAL scaling, ZLARFG, ZLAHQR and ZTREVC conventions.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley. All rights reserved.
// Copyright (c) 2006-2023 The University of Colorado Denver. All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer listed in this license
//   in the documentation and/or other materials provided with the distribution.
// - Neither the name of the copyright holders nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
// The copyright holders provide no reassurances that the source code provided
// does not infringe any patent, copyright, or any other intellectual property
// rights of third parties. The copyright holders disclaim any liability to any
// recipient for claims brought against recipient by any third party for
// infringement of that parties intellectual property rights.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
use super::{complex_kernel::divide, complex_qr_kernel::combine_norm};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda]
fn block() -> usize { RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize }

#[ruda]
fn bad<F: Float>(v: F) -> bool { v != v || v.abs() == F::cast_from(f32::INFINITY) }

#[ruda]
fn constants<F: Float>(#[comptime] wide: bool) -> (F, F) {
    let mut eps = F::cast_from(f32::EPSILON); let mut tiny = F::cast_from(f32::MIN_POSITIVE);
    if comptime!(wide) { eps = F::cast_from(f64::EPSILON); tiny = F::cast_from(f64::MIN_POSITIVE); }
    (eps, tiny)
}

#[ruda]
fn negative<F: Float>(v: F, #[comptime] wide: bool) -> bool {
    let mut result = false;
    if comptime!(wide) { result = (u64::reinterpret(v) >> 63u32) != 0u64; }
    else { result = (u32::reinterpret(v) >> 31u32) != 0u32; }
    result
}

#[ruda]
fn magnitude<F: Float>(r: F, i: F) -> F {
    let s = r.abs().max(i.abs()); let mut value = F::new(0.0);
    if s != F::new(0.0) { let x = r / s; let y = i / s; value = s * (x * x + y * y).sqrt(); }
    value
}

#[ruda]
fn maximum<F: Float>(value: F) -> F {
    let lane = UNIT_POS as usize; let mut values = SharedMemory::<F>::new(256);
    values[lane] = value; sync_ruda(); let mut width = 128usize;
    while width > 0 {
        if lane < width { values[lane] = values[lane].max(values[lane + width]); }
        sync_ruda(); width /= 2;
    }
    let out = values[0]; sync_ruda(); out
}

#[ruda]
fn norm<F: Float>(real: &Array<F>, imag: &Array<F>, start: usize, length: usize, stride: usize) -> (F, F) {
    let lane = UNIT_POS as usize; let mut i = lane;
    let mut s = F::new(0.0); let mut q = F::new(0.0);
    while i < length {
        let p = start + i * stride;
        let (s1, q1) = combine_norm(s, q, real[p].abs(), F::new(1.0));
        let (s2, q2) = combine_norm(s1, q1, imag[p].abs(), F::new(1.0));
        s = s2; q = q2; i += 256;
    }
    let mut scales = SharedMemory::<F>::new(256); let mut sums = SharedMemory::<F>::new(256);
    scales[lane] = s; sums[lane] = q; sync_ruda(); let mut width = 128usize;
    while width > 0 {
        if lane < width {
            let (s1, q1) = combine_norm(scales[lane], sums[lane], scales[lane + width], sums[lane + width]);
            scales[lane] = s1; sums[lane] = q1;
        }
        sync_ruda(); width /= 2;
    }
    s = scales[0]; q = sums[0]; sync_ruda(); (s, q)
}

#[ruda]
fn complex_sqrt<F: Float>(r: F, i: F, #[comptime] wide: bool) -> (F, F) {
    let s = r.abs().max(i.abs()); let mut x = F::new(0.0); let mut y = F::new(0.0);
    if s != F::new(0.0) {
        let a = r / s; let b = i / s;
        let t = s.sqrt() * ((magnitude(a, b) + a.abs()) * F::new(0.5)).sqrt();
        if r >= F::new(0.0) { x = t; y = (i / t) * F::new(0.5); }
        else { x = (i.abs() / t) * F::new(0.5); y = t; if negative(i, wide) { y = -y; } }
    }
    (x, y)
}

#[ruda]
fn beta<F: Float>(ar: F, ai: F, tail: F, #[comptime] wide: bool) -> F {
    let (s, q) = combine_norm(ar.abs(), F::new(1.0), ai.abs(), F::new(1.0));
    let (s, q) = combine_norm(s, q, tail, F::new(1.0));
    let mut value = -s * q.sqrt(); if negative(ar, wide) { value = -value; } value
}

#[ruda]
fn reflector2<F: Float>(ar: F, ai: F, tail: F, #[comptime] wide: bool) -> (F, F, F, F, F) {
    let mut a = ar; let mut b = ai; let mut x = tail;
    let mut value = ar; let mut tr = F::new(0.0); let mut ti = F::new(0.0);
    let mut vr = tail; let mut vi = F::new(0.0);
    if tail != F::new(0.0) || ai != F::new(0.0) {
        let (eps, tiny) = constants::<F>(wide); let safmin = tiny / (eps * F::new(0.5));
        value = beta(a, b, x.abs(), wide); let mut count = 0u32;
        while value.abs() < safmin && count < 20u32 {
            a /= safmin; b /= safmin; x /= safmin; value /= safmin; count += 1;
        }
        if count != 0 { value = beta(a, b, x.abs(), wide); }
        tr = (value - a) / value; ti = -b / value;
        let (r, i) = divide(x, F::new(0.0), a - value, b, wide); vr = r; vi = i;
        for _ in 0..count { value *= safmin; }
    }
    (value, tr, ti, vr, vi)
}

#[ruda(launch)]
pub(super) fn prepare<F: Float>(
    ar: &Tensor<F>, ai: &Tensor<F>, hr: &mut Array<F>, hi: &mut Array<F>, zr: &mut Array<F>, zi: &mut Array<F>,
    factors: &mut Array<F>, scaling: &mut Array<F>, info: &mut Array<i32>, order: u32,
    #[comptime] vectors: bool, #[comptime] wide: bool,
) {
    let batch = block(); if batch >= info.len() { terminate!(); }
    let lane = UNIT_POS as usize; let n = order as usize; let base = batch * n * n;
    let mut p = lane; let mut largest = F::new(0.0); let mut invalid = F::new(0.0);
    while p < n * n {
        let row = p / n; let col = p % n;
        let r = ar[batch * ar.stride(0) + row * ar.stride(1) + col * ar.stride(2)];
        let i = ai[batch * ai.stride(0) + row * ai.stride(1) + col * ai.stride(2)];
        hr[base + p] = r; hi[base + p] = i;
        if bad(r) || bad(i) { invalid = F::new(1.0); }
        largest = largest.max(r.abs()).max(i.abs());
        if comptime!(vectors) {
            let mut v = F::new(0.0); if row == col { v = F::new(1.0); }
            zr[base + p] = v; zi[base + p] = F::new(0.0);
        }
        p += 256;
    }
    let invalid = maximum(invalid); let largest = maximum(largest);
    let (eps, tiny) = constants::<F>(wide); let small = tiny.sqrt() / eps; let big = F::new(1.0) / small;
    let mut target = largest;
    if largest > F::new(0.0) && largest < small { target = small; }
    if largest > big { target = big; }
    if invalid == F::new(0.0) && target != largest {
        p = lane;
        let factor = target / largest;
        while p < n * n { hr[base + p] *= factor; hi[base + p] *= factor; p += 256; }
    }
    p = lane; while p < n { factors[batch * n + p] = F::new(1.0); p += 256; }
    if lane == 0 {
        info[batch] = 0; if invalid != F::new(0.0) { info[batch] = -1; }
        scaling[batch * 2] = F::new(1.0); scaling[batch * 2 + 1] = F::new(1.0);
        if target != largest && invalid == F::new(0.0) { scaling[batch * 2] = largest; scaling[batch * 2 + 1] = target; }
    }
}

#[ruda(launch)]
pub(super) fn balance<F: Float>(
    hr: &mut Array<F>, hi: &mut Array<F>, factors: &mut Array<F>, info: &mut Array<i32>, order: u32, #[comptime] wide: bool,
) {
    let batch = block(); if batch >= info.len() || info[batch] != 0 { terminate!(); }
    let lane = UNIT_POS as usize; let n = order as usize; let base = batch * n * n;
    let (eps, tiny) = constants::<F>(wide); let min1 = tiny / eps; let max1 = F::new(1.0) / min1;
    let min2 = min1 * F::new(2.0); let max2 = F::new(1.0) / min2;
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..n {
            let (cs, cq) = norm(hr, hi, base + i, n, n);
            let (rs, rq) = norm(hr, hi, base + i * n, n, 1usize);
            let mut c = cs * cq.sqrt(); let mut r = rs * rq.sqrt();
            let mut ca = F::new(0.0); let mut ra = F::new(0.0); let mut j = lane;
            while j < n {
                ca = ca.max(magnitude(hr[base + j * n + i], hi[base + j * n + i]));
                ra = ra.max(magnitude(hr[base + i * n + j], hi[base + i * n + j])); j += 256;
            }
            ca = maximum(ca); ra = maximum(ra);
            if bad(c + r + ca + ra) { if lane == 0 { info[batch] = -3; } terminate!(); }
            let mut f = F::new(1.0); let original = c + r; let mut g = r * F::new(0.5);
            if c != F::new(0.0) && r != F::new(0.0) {
                while c < g && f.max(c).max(ca) < max2 && r.min(g).min(ra) > min2 {
                    f *= F::new(2.0); c *= F::new(2.0); ca *= F::new(2.0);
                    r *= F::new(0.5); g *= F::new(0.5); ra *= F::new(0.5);
                }
                g = c * F::new(0.5);
                while g >= r && r.max(ra) < max2 && f.min(c).min(g).min(ca) > min2 {
                    f *= F::new(0.5); c *= F::new(0.5); g *= F::new(0.5); ca *= F::new(0.5);
                    r *= F::new(2.0); ra *= F::new(2.0);
                }
                let old = factors[batch * n + i];
                let mut take = c + r < F::cast_from(0.95f64) * original;
                if f < F::new(1.0) && old < F::new(1.0) && f * old <= min1 { take = false; }
                if f > F::new(1.0) && old > F::new(1.0) && old >= max1 / f { take = false; }
                sync_ruda();
                if take {
                    changed = true; if lane == 0 { factors[batch * n + i] = old * f; }
                    j = lane;
                    while j < n {
                        if j != i {
                            let row = base + i * n + j; let col = base + j * n + i;
                            hr[row] /= f; hi[row] /= f; hr[col] *= f; hi[col] *= f;
                        }
                        j += 256;
                    }
                }
                sync_ruda();
            }
        }
    }
}

#[ruda(launch)]
pub(super) fn hessenberg<F: Float>(
    hr: &mut Array<F>, hi: &mut Array<F>, zr: &mut Array<F>, zi: &mut Array<F>,
    vr: &mut Array<F>, vi: &mut Array<F>, info: &mut Array<i32>, order: u32,
    #[comptime] vectors: bool, #[comptime] wide: bool,
) {
    let batch = block(); if batch >= info.len() || info[batch] != 0 { terminate!(); }
    let lane = UNIT_POS as usize; let n = order as usize; let base = batch * n * n; let vbase = batch * n;
    let (eps, tiny) = constants::<F>(wide); let safmin = tiny / (eps * F::new(0.5));
    for j in 0..n - 1 {
        let first = j + 1; let length = n - first; let mut row = lane;
        while row < length { vr[vbase + row] = hr[base + (first + row) * n + j]; vi[vbase + row] = hi[base + (first + row) * n + j]; row += 256; }
        sync_ruda();
        let mut ar = vr[vbase]; let mut ai = vi[vbase]; let (s, q) = norm(vr, vi, vbase + 1, length - 1, 1usize);
        let mut tail = s * q.sqrt(); let mut b = ar; let mut tr = F::new(0.0); let mut ti = F::new(0.0);
        if tail != F::new(0.0) || ai != F::new(0.0) {
            b = beta(ar, ai, tail, wide); let mut count = 0u32;
            while b.abs() < safmin && count < 20u32 {
                row = lane + 1; while row < length { vr[vbase + row] /= safmin; vi[vbase + row] /= safmin; row += 256; }
                ar /= safmin; ai /= safmin; b /= safmin; count += 1; sync_ruda();
            }
            if count != 0 { let (s, q) = norm(vr, vi, vbase + 1, length - 1, 1usize); tail = s * q.sqrt(); b = beta(ar, ai, tail, wide); }
            tr = (b - ar) / b; ti = -ai / b;
            row = lane + 1;
            while row < length {
                let (r, i) = divide(vr[vbase + row], vi[vbase + row], ar - b, ai, wide);
                vr[vbase + row] = r; vi[vbase + row] = i; row += 256;
            }
            for _ in 0..count { b *= safmin; }
        }
        sync_ruda();
        if lane == 0 { vr[vbase] = F::new(1.0); vi[vbase] = F::new(0.0); }
        row = lane;
        while row < length {
            let p = base + (first + row) * n + j; hr[p] = F::new(0.0); hi[p] = F::new(0.0);
            if row == 0 { hr[p] = b; }
            row += 256;
        }
        sync_ruda();
        if tr != F::new(0.0) || ti != F::new(0.0) {
            let mut col = first + lane;
            while col < n {
                let mut dr = F::new(0.0); let mut di = F::new(0.0);
                for k in 0..length {
                    let r = hr[base + (first + k) * n + col]; let i = hi[base + (first + k) * n + col];
                    dr += vr[vbase + k] * r + vi[vbase + k] * i; di += vr[vbase + k] * i - vi[vbase + k] * r;
                }
                let wr = tr * dr + ti * di; let wi = tr * di - ti * dr;
                for k in 0..length {
                    let p = base + (first + k) * n + col;
                    hr[p] -= vr[vbase + k] * wr - vi[vbase + k] * wi; hi[p] -= vr[vbase + k] * wi + vi[vbase + k] * wr;
                }
                col += 256;
            }
            sync_ruda(); row = lane;
            while row < n {
                let mut dr = F::new(0.0); let mut di = F::new(0.0);
                for k in 0..length {
                    let r = hr[base + row * n + first + k]; let i = hi[base + row * n + first + k];
                    dr += r * vr[vbase + k] - i * vi[vbase + k]; di += r * vi[vbase + k] + i * vr[vbase + k];
                }
                let wr = tr * dr - ti * di; let wi = tr * di + ti * dr;
                for k in 0..length {
                    let p = base + row * n + first + k;
                    hr[p] -= wr * vr[vbase + k] + wi * vi[vbase + k]; hi[p] -= wi * vr[vbase + k] - wr * vi[vbase + k];
                }
                if comptime!(vectors) {
                    dr = F::new(0.0); di = F::new(0.0);
                    for k in 0..length {
                        let r = zr[base + row * n + first + k]; let i = zi[base + row * n + first + k];
                        dr += r * vr[vbase + k] - i * vi[vbase + k]; di += r * vi[vbase + k] + i * vr[vbase + k];
                    }
                    let wr = tr * dr - ti * di; let wi = tr * di + ti * dr;
                    for k in 0..length {
                        let p = base + row * n + first + k;
                        zr[p] -= wr * vr[vbase + k] + wi * vi[vbase + k]; zi[p] -= wi * vr[vbase + k] - wr * vi[vbase + k];
                    }
                }
                row += 256;
            }
            sync_ruda();
        }
    }
}

#[ruda(launch)]
pub(super) fn schur<F: Float>(
    hr: &mut Array<F>, hi: &mut Array<F>, zr: &mut Array<F>, zi: &mut Array<F>,
    wr: &mut Array<F>, wi: &mut Array<F>, scaling: &Array<F>, info: &mut Array<i32>, order: u32,
    #[comptime] vectors: bool, #[comptime] wide: bool,
) {
    let batch = block(); if batch >= info.len() { terminate!(); }
    let lane = UNIT_POS as usize; let n = order as usize; let base = batch * n * n;
    let (eps, tiny) = constants::<F>(wide); let small = tiny * (F::cast_from(n) / eps);
    let mut state = SharedMemory::<u32>::new(2); let mut scalars = SharedMemory::<F>::new(6);
    let mut end = n; let mut iterations = 0usize; let limit = 30usize * n.max(10usize);
    let eligible = info[batch] == 0; sync_ruda();
    while end > 0 && eligible {
        let i = end - 1;
        if lane == 0 {
            let mut l = i;
            while l > 0 {
                let sub = base + l * n + l - 1; let d0 = base + (l - 1) * n + l - 1; let d1 = base + l * n + l;
                let subnorm = hr[sub].abs() + hi[sub].abs();
                let mut split = subnorm <= small;
                let mut test = hr[d0].abs() + hi[d0].abs() + hr[d1].abs() + hi[d1].abs();
                if test == F::new(0.0) {
                    if l >= 2 { test += hr[base + (l - 1) * n + l - 2].abs(); }
                    if l + 1 < n { test += hr[base + (l + 1) * n + l].abs(); }
                }
                if hr[sub].abs() <= eps * test {
                    let above = base + (l - 1) * n + l; let upper = hr[above].abs() + hi[above].abs();
                    let ab = subnorm.max(upper); let ba = subnorm.min(upper);
                    let d = hr[d1].abs() + hi[d1].abs(); let delta = (hr[d0] - hr[d1]).abs() + (hi[d0] - hi[d1]).abs();
                    let aa = d.max(delta); let bb = d.min(delta); let s = aa + ab;
                    if s == F::new(0.0) || ba * (ab / s) <= small.max(eps * (bb * (aa / s))) { split = true; }
                }
                if split { break; } l -= 1;
            }
            state[0] = l as u32;
            if l > 0 { hr[base + l * n + l - 1] = F::new(0.0); hi[base + l * n + l - 1] = F::new(0.0); }
            if l < i {
                if iterations > limit { info[batch] = end as i32; }
                else {
                    let mut tr = hr[base + i * n + i]; let mut ti = hi[base + i * n + i];
                    if (iterations + 1) % 20usize == 0 {
                        tr += F::new(0.75) * hr[base + i * n + i - 1].abs();
                    } else if (iterations + 1) % 10usize == 0 {
                        tr = hr[base + l * n + l] + F::new(0.75) * hr[base + (l + 1) * n + l].abs(); ti = hi[base + l * n + l];
                    } else {
                        let (pr, pi) = complex_sqrt(hr[base + (i - 1) * n + i], hi[base + (i - 1) * n + i], wide);
                        let (qr, qi) = complex_sqrt(hr[base + i * n + i - 1], hi[base + i * n + i - 1], wide);
                        let ur = pr * qr - pi * qi; let ui = pr * qi + pi * qr;
                        let unorm = ur.abs() + ui.abs();
                        if unorm != F::new(0.0) {
                            let xr = F::new(0.5) * (hr[base + (i - 1) * n + i - 1] - tr);
                            let xi = F::new(0.5) * (hi[base + (i - 1) * n + i - 1] - ti);
                            let sx = xr.abs() + xi.abs(); let s = unorm.max(sx);
                            let a = xr / s; let b = xi / s; let c = ur / s; let d = ui / s;
                            let (yr, yi) = complex_sqrt(a * a - b * b + c * c - d * d, F::new(2.0) * (a * b + c * d), wide);
                            let mut yr = yr * s; let mut yi = yi * s;
                            if sx > F::new(0.0) && (xr / sx) * yr + (xi / sx) * yi < F::new(0.0) { yr = -yr; yi = -yi; }
                            let (r, im) = divide(ur, ui, xr + yr, xi + yi, wide);
                            tr -= ur * r - ui * im; ti -= ur * im + ui * r;
                        }
                    }
                    let mut m = i - 1;
                    loop {
                        let p = base + m * n + m; let ar = hr[p] - tr; let ai = hi[p] - ti;
                        let tail = hr[base + (m + 1) * n + m]; let s = ar.abs() + ai.abs() + tail.abs();
                        let mut vr = F::new(1.0); let mut vi = F::new(0.0); let mut v2 = F::new(0.0);
                        if s != F::new(0.0) { vr = ar / s; vi = ai / s; v2 = tail / s; }
                        scalars[0] = vr; scalars[1] = vi; scalars[2] = v2;
                        let mut start = m == l;
                        if m > l {
                            let h10 = hr[base + m * n + m - 1]; let p2 = base + (m + 1) * n + m + 1;
                            start = h10.abs() * v2.abs() <= eps * ((vr.abs() + vi.abs()) * (hr[p].abs() + hi[p].abs() + hr[p2].abs() + hi[p2].abs()));
                        }
                        if start { break; } m -= 1;
                    }
                    state[1] = m as u32;
                    if bad(tr) || bad(ti) { info[batch] = -3; }
                }
            }
        }
        sync_ruda();
        if info[batch] != 0 { break; }
        let l = state[0] as usize; sync_ruda();
        if l == i { end -= 1; iterations = 0; }
        else {
            iterations += 1; let m = state[1] as usize;
            for k in m..i {
                if lane == 0 {
                    let mut ar = scalars[0]; let mut ai = scalars[1]; let mut tail = scalars[2];
                    if k > m { ar = hr[base + k * n + k - 1]; ai = hi[base + k * n + k - 1]; tail = hr[base + (k + 1) * n + k - 1]; }
                    let (b, tr, ti, vr, vi) = reflector2(ar, ai, tail, wide);
                    scalars[0] = tr; scalars[1] = ti; scalars[2] = vr; scalars[3] = vi; scalars[4] = tr * vr - ti * vi;
                    if k > m {
                        hr[base + k * n + k - 1] = b; hi[base + k * n + k - 1] = F::new(0.0);
                        hr[base + (k + 1) * n + k - 1] = F::new(0.0); hi[base + (k + 1) * n + k - 1] = F::new(0.0);
                    }
                }
                sync_ruda();
                let tr = scalars[0]; let ti = scalars[1]; let vr = scalars[2]; let vi = scalars[3]; let t2 = scalars[4];
                let mut col = k + lane;
                while col < n {
                    let p = base + k * n + col; let q = p + n;
                    let r = tr * hr[p] + ti * hi[p] + t2 * hr[q]; let im = tr * hi[p] - ti * hr[p] + t2 * hi[q];
                    hr[p] -= r; hi[p] -= im; hr[q] -= r * vr - im * vi; hi[q] -= r * vi + im * vr; col += 256;
                }
                sync_ruda(); let mut row = lane;
                while row <= (k + 2).min(i) {
                    let p = base + row * n + k; let q = p + 1;
                    let r = tr * hr[p] - ti * hi[p] + t2 * hr[q]; let im = tr * hi[p] + ti * hr[p] + t2 * hi[q];
                    hr[p] -= r; hi[p] -= im; hr[q] -= r * vr + im * vi; hi[q] -= im * vr - r * vi; row += 256;
                }
                if comptime!(vectors) {
                    row = lane;
                    while row < n {
                        let p = base + row * n + k; let q = p + 1;
                        let r = tr * zr[p] - ti * zi[p] + t2 * zr[q]; let im = tr * zi[p] + ti * zr[p] + t2 * zi[q];
                        zr[p] -= r; zi[p] -= im; zr[q] -= r * vr + im * vi; zi[q] -= im * vr - r * vi; row += 256;
                    }
                }
                sync_ruda();
                if k == m && m > l {
                    let rr = F::new(1.0) - tr; let ii = -ti; let abs = magnitude(rr, ii); let rr = rr / abs; let ii = ii / abs;
                    if lane == 0 {
                        let p = base + (m + 1) * n + m; let r = hr[p]; let im = hi[p];
                        hr[p] = r * rr + im * ii; hi[p] = im * rr - r * ii;
                        if m + 2 <= i {
                            let p = base + (m + 2) * n + m + 1; let r = hr[p]; let im = hi[p];
                            hr[p] = r * rr - im * ii; hi[p] = r * ii + im * rr;
                        }
                    }
                    sync_ruda();
                    for j in m..i + 1 {
                        if j != m + 1 {
                            col = j + 1 + lane;
                            while col < n {
                                let p = base + j * n + col; let r = hr[p]; let im = hi[p];
                                hr[p] = r * rr - im * ii; hi[p] = r * ii + im * rr; col += 256;
                            }
                            row = lane;
                            while row < j {
                                let p = base + row * n + j; let r = hr[p]; let im = hi[p];
                                hr[p] = r * rr + im * ii; hi[p] = im * rr - r * ii; row += 256;
                            }
                            if comptime!(vectors) {
                                row = lane;
                                while row < n {
                                    let p = base + row * n + j; let r = zr[p]; let im = zi[p];
                                    zr[p] = r * rr + im * ii; zi[p] = im * rr - r * ii; row += 256;
                                }
                            }
                            sync_ruda();
                        }
                    }
                }
            }
            if lane == 0 {
                let p = base + i * n + i - 1; let r = hr[p]; let im = hi[p];
                scalars[0] = F::new(1.0); scalars[1] = F::new(0.0);
                if im != F::new(0.0) {
                    let a = magnitude(r, im); scalars[0] = r / a; scalars[1] = im / a; hr[p] = a; hi[p] = F::new(0.0);
                }
            }
            sync_ruda();
            let rr = scalars[0]; let ii = scalars[1]; let mut col = i + 1 + lane;
            while col < n {
                let p = base + i * n + col; let r = hr[p]; let im = hi[p];
                hr[p] = r * rr + im * ii; hi[p] = im * rr - r * ii; col += 256;
            }
            let mut row = lane;
            while row < i {
                let p = base + row * n + i; let r = hr[p]; let im = hi[p];
                hr[p] = r * rr - im * ii; hi[p] = r * ii + im * rr; row += 256;
            }
            if comptime!(vectors) {
                row = lane;
                while row < n {
                    let p = base + row * n + i; let r = zr[p]; let im = zi[p];
                    zr[p] = r * rr - im * ii; zi[p] = r * ii + im * rr; row += 256;
                }
            }
            sync_ruda();
        }
    }
    let mut p = lane; let mut invalid = F::new(0.0);
    while p < n * n { if bad(hr[base + p]) || bad(hi[base + p]) { invalid = F::new(1.0); } p += 256; }
    p = lane; let from = scaling[batch * 2 + 1]; let to = scaling[batch * 2];
    while p < n {
        let r = hr[base + p * n + p]; let im = hi[base + p * n + p];
        wr[batch * n + p] = r * (to / from); wi[batch * n + p] = im * (to / from); p += 256;
    }
    let invalid = maximum(invalid); if lane == 0 && info[batch] == 0 && invalid != F::new(0.0) { info[batch] = -3; }
}

#[ruda]
fn sum<F: Float>(value: F) -> F {
    let lane = UNIT_POS as usize; let mut values = SharedMemory::<F>::new(256);
    values[lane] = value; sync_ruda(); let mut width = 128usize;
    while width > 0 {
        if lane < width { values[lane] += values[lane + width]; }
        sync_ruda(); width /= 2;
    }
    let out = values[0]; sync_ruda(); out
}

#[ruda(launch)]
pub(super) fn triangular_vectors<F: Float>(
    hr: &Array<F>, hi: &Array<F>, xr: &mut Array<F>, xi: &mut Array<F>, info: &Array<i32>, order: u32, #[comptime] wide: bool,
) {
    let work = block(); let n = order as usize; if work >= info.len() * n { terminate!(); }
    let batch = work / n; let col = work % n; let lane = UNIT_POS as usize; let base = batch * n * n;
    let mut row = lane;
    while row < n {
        xr[base + row * n + col] = F::new(0.0); xi[base + row * n + col] = F::new(0.0);
        if row == col { xr[base + row * n + col] = F::new(1.0); } row += 256;
    }
    if info[batch] != 0 { terminate!(); }
    sync_ruda();
    let lambda_r = hr[base + col * n + col]; let lambda_i = hi[base + col * n + col];
    let (eps, tiny) = constants::<F>(wide);
    let smin = (eps * (lambda_r.abs() + lambda_i.abs())).max(tiny * (F::cast_from(n) / eps));
    let mut end = col;
    while end > 0 {
        let j = end - 1; let mut k = j + 1 + lane; let mut rr = F::new(0.0); let mut ri = F::new(0.0);
        while k <= col {
            let p = base + j * n + k; let q = base + k * n + col;
            rr -= hr[p] * xr[q] - hi[p] * xi[q]; ri -= hr[p] * xi[q] + hi[p] * xr[q]; k += 256;
        }
        let rr = sum(rr); let ri = sum(ri);
        let mut dr = hr[base + j * n + j] - lambda_r; let mut di = hi[base + j * n + j] - lambda_i;
        if dr.abs() + di.abs() < smin { dr = smin; di = F::new(0.0); }
        let ds = dr.abs().max(di.abs()); let rs = rr.abs().max(ri.abs());
        let mut scale = F::new(1.0); let mut r = rr; let mut im = ri;
        if rs > ds {
            scale = ds / rs;
            let (a, b) = divide(rr / rs, ri / rs, dr / ds, di / ds, wide); r = a; im = b;
        } else { let (a, b) = divide(rr, ri, dr, di, wide); r = a; im = b; }
        let normalize = r.abs().max(im.abs()).max(F::new(1.0));
        k = j + 1 + lane;
        while k <= col {
            let p = base + k * n + col; xr[p] = (xr[p] * scale) / normalize; xi[p] = (xi[p] * scale) / normalize; k += 256;
        }
        if lane == 0 { xr[base + j * n + col] = r / normalize; xi[base + j * n + col] = im / normalize; }
        sync_ruda(); end -= 1;
    }
}

#[ruda(launch)]
pub(super) fn backtransform<F: Float>(
    zr: &Array<F>, zi: &Array<F>, xr: &Array<F>, xi: &Array<F>, yr: &mut Array<F>, yi: &mut Array<F>,
    factors: &Array<F>, info: &Array<i32>, order: u32,
) {
    let work = block(); let n = order as usize; if work >= info.len() * n { terminate!(); }
    let batch = work / n; let col = work % n; let lane = UNIT_POS as usize; let base = batch * n * n;
    let mut row = lane;
    while row < n {
        let mut r = F::new(0.0); let mut im = F::new(0.0);
        if info[batch] == 0 {
            for k in 0..col + 1 {
                let p = base + row * n + k; let q = base + k * n + col;
                r += zr[p] * xr[q] - zi[p] * xi[q]; im += zr[p] * xi[q] + zi[p] * xr[q];
            }
            r *= factors[batch * n + row]; im *= factors[batch * n + row];
        }
        yr[base + row * n + col] = r; yi[base + row * n + col] = im; row += 256;
    }
    if info[batch] != 0 { terminate!(); }
    sync_ruda();
    let (scale, square_sum) = norm(yr, yi, base + col, n, n); let root = square_sum.sqrt();
    let mut phase = SharedMemory::<F>::new(2); let mut index = SharedMemory::<u32>::new(1);
    if lane == 0 {
        let mut largest = F::new(0.0); let mut choice = 0usize;
        for j in 0..n {
            let p = base + j * n + col; let value = magnitude(yr[p], yi[p]);
            if value > largest { largest = value; choice = j; }
        }
        index[0] = choice as u32; phase[0] = yr[base + choice * n + col] / largest; phase[1] = -yi[base + choice * n + col] / largest;
    }
    sync_ruda();
    let pr = phase[0]; let pi = phase[1]; let selected = index[0] as usize; row = lane;
    while row < n {
        let p = base + row * n + col; let r = (yr[p] / scale) / root; let im = (yi[p] / scale) / root;
        yr[p] = r * pr - im * pi; yi[p] = r * pi + im * pr;
        if row == selected { yi[p] = F::new(0.0); } row += 256;
    }
}

#[ruda(launch)]
pub(super) fn check_vectors<F: Float>(real: &Array<F>, imag: &Array<F>, info: &mut Array<i32>, order: u32) {
    let batch = block(); if batch >= info.len() || info[batch] != 0 { terminate!(); }
    let n = order as usize; let lane = UNIT_POS as usize; let base = batch * n * n;
    let mut p = lane; let mut invalid = F::new(0.0);
    while p < n * n { if bad(real[base + p]) || bad(imag[base + p]) { invalid = F::new(1.0); } p += 256; }
    let invalid = maximum(invalid); if lane == 0 && invalid != F::new(0.0) { info[batch] = -3; }
}
