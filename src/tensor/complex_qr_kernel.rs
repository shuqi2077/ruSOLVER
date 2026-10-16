// Reflector generation adapts LAPACK 3.12.1 CLARFG/ZLARFG.
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
use super::complex_kernel::divide;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda]
pub(super) fn combine_norm<F: Float>(scale: F, sum: F, other: F, other_sum: F) -> (F, F) {
    let mut s = scale; let mut q = sum;
    if scale != scale || other != other { s = scale + other; q = s; }
    else if scale == F::cast_from(f32::INFINITY) || other == F::cast_from(f32::INFINITY) {
        s = F::cast_from(f32::INFINITY); q = F::new(1.0);
    } else if scale < other {
        let ratio = scale / other; s = other; q = other_sum + sum * ratio * ratio;
    } else if scale != F::new(0.0) {
        let ratio = other / scale; q = sum + other_sum * ratio * ratio;
    }
    (s, q)
}

#[ruda]
fn tail_norm<F: Float>(real: &Array<F>, imag: &Array<F>, base: usize, m: usize, n: usize, j: usize) -> F {
    let lane = UNIT_POS as usize;
    let mut scale = F::new(0.0); let mut sum = F::new(0.0); let mut row = j + 1 + lane;
    while row < m {
        let p = base + row * n + j;
        let (sr, qr) = combine_norm::<F>(scale, sum, real[p].abs(), F::new(1.0));
        let (si, qi) = combine_norm::<F>(sr, qr, imag[p].abs(), F::new(1.0));
        scale = si; sum = qi; row += 256;
    }
    let mut scales = SharedMemory::<F>::new(256usize); let mut sums = SharedMemory::<F>::new(256usize);
    scales[lane] = scale; sums[lane] = sum; sync_ruda();
    let mut width = 128usize;
    while width > 0 {
        if lane < width {
            let (s, q) = combine_norm::<F>(scales[lane], sums[lane], scales[lane + width], sums[lane + width]);
            scales[lane] = s; sums[lane] = q;
        }
        sync_ruda(); width /= 2;
    }
    let value = scales[0usize] * sums[0usize].sqrt(); sync_ruda(); value
}

#[ruda]
fn beta<F: Float>(ar: F, ai: F, norm: F, #[comptime] wide: bool) -> F {
    let (s, q) = combine_norm::<F>(ar.abs(), F::new(1.0), ai.abs(), F::new(1.0));
    let (s, q) = combine_norm::<F>(s, q, norm, F::new(1.0));
    let mut negative = false;
    if comptime!(wide) { negative = (u64::reinterpret(ar) >> 63u32) != 0u64; }
    else { negative = (u32::reinterpret(ar) >> 31u32) != 0u32; }
    let mut value = -s * q.sqrt(); if negative { value = -value; } value
}

#[ruda(launch)]
pub(super) fn panel<F: Float>(
    real: &mut Array<F>, imag: &mut Array<F>, tau_real: &mut Array<F>, tau_imag: &mut Array<F>,
    batches: u32, rows: u32, cols: u32, column: u32, #[comptime] wide: bool,
) {
    let batch = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if batch >= batches as usize { terminate!(); }
    let lane = UNIT_POS as usize; let m = rows as usize; let n = cols as usize; let j = column as usize;
    let base = batch * m * n; let diagonal = base + j * n + j; let t = batch * m.min(n) + j;
    let mut ar = real[diagonal]; let mut ai = imag[diagonal];
    let mut norm = tail_norm::<F>(real, imag, base, m, n, j);
    if norm == F::new(0.0) && ai == F::new(0.0) {
        if lane == 0 { tau_real[t] = F::new(0.0); tau_imag[t] = F::new(0.0); }
    } else {
        let mut b = beta::<F>(ar, ai, norm, wide);
        let mut safmin = F::cast_from(f32::MIN_POSITIVE / (f32::EPSILON / 2.0));
        if comptime!(wide) { safmin = F::cast_from(f64::MIN_POSITIVE / (f64::EPSILON / 2.0)); }
        let reciprocal = F::new(1.0) / safmin; let mut count = 0u32;
        if b.abs() < safmin {
            while b.abs() < safmin && count < 20u32 {
                let mut row = j + 1 + lane;
                while row < m {
                    let p = base + row * n + j; real[p] *= reciprocal; imag[p] *= reciprocal; row += 256;
                }
                ar *= reciprocal; ai *= reciprocal; b *= reciprocal; count += 1;
                sync_ruda();
            }
            norm = tail_norm::<F>(real, imag, base, m, n, j);
            b = beta::<F>(ar, ai, norm, wide);
        }
        let tr = (b - ar) / b; let ti = -ai / b;
        let (rr, ri) = divide::<F>(F::new(1.0), F::new(0.0), ar - b, ai, wide);
        let mut row = j + 1 + lane;
        while row < m {
            let p = base + row * n + j; let r = real[p]; let i = imag[p];
            real[p] = r * rr - i * ri; imag[p] = r * ri + i * rr; row += 256;
        }
        if lane == 0 {
            tau_real[t] = tr; tau_imag[t] = ti;
            for _ in 0..count { b *= safmin; }
            real[diagonal] = b; imag[diagonal] = F::new(0.0);
        }
    }
}

#[ruda(launch)]
pub(super) fn apply<F: Float>(
    packed_real: &Array<F>, packed_imag: &Array<F>, tau_real: &Array<F>, tau_imag: &Array<F>,
    real: &mut Array<F>, imag: &mut Array<F>, batches: u32, rows: u32, cols: u32,
    reflectors: u32, output_cols: u32, column: u32, first_col: u32, #[comptime] adjoint: bool,
) {
    let work = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let m = rows as usize; let n = cols as usize; let nrhs = output_cols as usize;
    let width = nrhs - first_col as usize;
    if work >= batches as usize * width { terminate!(); }
    let batch = work / width; let col = work % width + first_col as usize;
    let lane = UNIT_POS as usize; let j = column as usize; let factor = batch * m * n; let out = batch * m * nrhs;
    let t = batch * reflectors as usize + j; let tr = tau_real[t]; let mut ti = tau_imag[t];
    if tr == F::new(0.0) && ti == F::new(0.0) { terminate!(); }
    if comptime!(adjoint) { ti = -ti; }
    let mut dr = F::new(0.0); let mut di = F::new(0.0); let mut row = j + lane;
    while row < m {
        let mut vr = F::new(1.0); let mut vi = F::new(0.0);
        if row != j { vr = packed_real[factor + row * n + j]; vi = packed_imag[factor + row * n + j]; }
        let r = real[out + row * nrhs + col]; let i = imag[out + row * nrhs + col];
        dr += vr * r + vi * i; di += vr * i - vi * r; row += 256;
    }
    let mut reals = SharedMemory::<F>::new(256usize); let mut imags = SharedMemory::<F>::new(256usize);
    reals[lane] = dr; imags[lane] = di; sync_ruda();
    let mut step = 128usize;
    while step > 0 {
        if lane < step { reals[lane] = reals[lane] + reals[lane + step]; imags[lane] = imags[lane] + imags[lane + step]; }
        sync_ruda(); step /= 2;
    }
    let wr = tr * reals[0usize] - ti * imags[0usize]; let wi = tr * imags[0usize] + ti * reals[0usize];
    row = j + lane;
    while row < m {
        let mut vr = F::new(1.0); let mut vi = F::new(0.0);
        if row != j { vr = packed_real[factor + row * n + j]; vi = packed_imag[factor + row * n + j]; }
        let p = out + row * nrhs + col; real[p] -= vr * wr - vi * wi; imag[p] -= vr * wi + vi * wr;
        row += 256;
    }
}

#[ruda(launch)]
pub(super) fn resize<F: Float>(
    ar: &Array<F>, ai: &Array<F>, real: &mut Array<F>, imag: &mut Array<F>,
    rows: u32, cols: u32, output_rows: u32, output_cols: u32, #[comptime] upper: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < real.len() {
        let om = output_rows as usize; let on = output_cols as usize;
        let m = rows as usize; let n = cols as usize;
        let batch = p / (om * on); let row = p / on % om; let col = p % on;
        let mut r = F::new(0.0); let mut i = F::new(0.0);
        let mut take = row < m && col < n; if comptime!(upper) { take = take && row <= col; }
        if take { let q = (batch * m + row) * n + col; r = ar[q]; i = ai[q]; }
        real[p] = r; imag[p] = i;
    }
}

#[ruda(launch)]
pub(super) fn residuals<F: Float>(real: &Array<F>, imag: &Array<F>, out: &mut Array<F>, rows: u32, cols: u32, start: u32) {
    let work = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if work >= out.len() { terminate!(); }
    let lane = UNIT_POS as usize; let m = rows as usize; let nrhs = cols as usize;
    let batch = work / nrhs; let col = work % nrhs; let mut row = start as usize + lane;
    let mut value = F::new(0.0);
    while row < m {
        let p = (batch * m + row) * nrhs + col;
        value += real[p] * real[p] + imag[p] * imag[p]; row += 256;
    }
    let mut sums = SharedMemory::<F>::new(256usize); sums[lane] = value; sync_ruda();
    let mut width = 128usize;
    while width > 0 {
        if lane < width { sums[lane] = sums[lane] + sums[lane + width]; }
        sync_ruda(); width /= 2;
    }
    if lane == 0 { out[work] = sums[0usize]; }
}
