// Lower Bunch-Kaufman pivoting/solves adapt LAPACK 3.12.1
// ZHETF2/ZSYTF2/ZHETRS/ZSYTRS.
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
fn block() -> usize { RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize }

#[ruda]
fn product<F: Float>(ar: F, ai: F, br: F, bi: F) -> (F, F) { (ar * br - ai * bi, ar * bi + ai * br) }

#[ruda]
fn block_solve<F: Float>(
    ar: F, ai: F, dr: F, di: F, er: F, ei: F, ur: F, ui: F, vr: F, vi: F,
    #[comptime] hermitian: bool, #[comptime] row: bool, #[comptime] wide: bool,
) -> (F, F, F, F) {
    let mut lower_i = ei; if comptime!(hermitian && row) { lower_i = -lower_i; }
    let mut upper_i = lower_i; if comptime!(hermitian) { upper_i = -upper_i; }
    let (a, b) = divide::<F>(ar, ai, er, upper_i, wide); let (c, d) = divide::<F>(dr, di, er, lower_i, wide);
    let (x, y) = divide::<F>(ur, ui, er, upper_i, wide); let (z, w) = divide::<F>(vr, vi, er, lower_i, wide);
    let (den_r, den_i) = product::<F>(a, b, c, d); let den_r = den_r - F::new(1.0);
    let (p, q) = product::<F>(c, d, x, y); let (r, s) = product::<F>(a, b, z, w);
    let (xr, xi) = divide::<F>(p - z, q - w, den_r, den_i, wide);
    let (yr, yi) = divide::<F>(r - x, s - y, den_r, den_i, wide); (xr, xi, yr, yi)
}

#[ruda(launch)]
pub(super) fn zero<F: Float>(out: &mut Array<F>) {
    let i = block() * 256usize + UNIT_POS as usize; if i < out.len() { out[i] = F::new(0.0); }
}

#[ruda(launch)]
pub(super) fn prepare<F: Float>(ar: &Tensor<F>, ai: &Tensor<F>, real: &mut Array<F>, imag: &mut Array<F>, state: &mut Array<i32>, info: &mut Array<i32>, order: u32, #[comptime] hermitian: bool) {
    let i = block() * 256usize + UNIT_POS as usize; let n = order as usize;
    if i < state.len() { state[i] = 0; } if i < info.len() { info[i] = 0; }
    if i < real.len() {
        let batch = i / (n * n); let row = i / n % n; let col = i % n; let mut r = F::new(0.0); let mut im = F::new(0.0);
        if row >= col {
            r = ar[batch * ar.stride(0) + row * ar.stride(1) + col * ar.stride(2)];
            if !hermitian || row != col { im = ai[batch * ai.stride(0) + row * ai.stride(1) + col * ai.stride(2)]; }
        }
        real[i] = r; imag[i] = im;
    }
}

#[ruda(launch)]
pub(super) fn panel<F: Float>(real: &mut Array<F>, imag: &mut Array<F>, pivots: &mut Array<i32>, state: &mut Array<i32>, info: &mut Array<i32>, order: u32, #[comptime] hermitian: bool) {
    let batch = block(); if batch >= info.len() { terminate!(); }
    let lane = UNIT_POS as usize; let n = order as usize; let k = state[batch * 4] as usize;
    if k >= n { if lane == 0 { state[batch * 4 + 3] = 0; } terminate!(); }
    let base = batch * n * n; let diagonal = base + k * n + k;
    let mut absakk = real[diagonal].abs(); if comptime!(!hermitian) { absakk += imag[diagonal].abs(); }
    let mut score = F::new(0.0); let mut index = k + 1; let mut row = k + 1 + lane;
    while row < n {
        let value = real[base + row * n + k].abs() + imag[base + row * n + k].abs();
        if value > score || value != value && score == score { score = value; index = row; } row += 256;
    }
    let mut scores = SharedMemory::<F>::new(256usize); let mut indices = SharedMemory::<u32>::new(256usize);
    scores[lane] = score; indices[lane] = index as u32; sync_ruda(); let mut width = 128usize;
    while width > 0 {
        if lane < width {
            let own = scores[lane]; let other = scores[lane + width]; let oi = indices[lane + width]; let ii = indices[lane];
            if other > own || other == own && oi < ii || other != other && (own == own || oi < ii) { scores[lane] = other; indices[lane] = oi; }
        }
        sync_ruda(); width /= 2;
    }
    let colmax = scores[0usize]; let imax = indices[0usize] as usize;
    let active = (absakk != F::new(0.0) || colmax != F::new(0.0)) && absakk == absakk;
    let alpha = (F::new(1.0) + F::new(17.0).sqrt()) / F::new(8.0); let mut kp = k; let mut step = 1usize;
    if active && !(absakk >= alpha * colmax) {
        let mut col = k + lane; let mut rowmax = F::new(0.0);
        while col < n {
            if col != imax {
                let mut p = base + imax * n + col; if col > imax { p = base + col * n + imax; }
                let value = real[p].abs() + imag[p].abs(); if value > rowmax || value != value { rowmax = value; }
            }
            col += 256;
        }
        sync_ruda(); scores[lane] = rowmax; sync_ruda(); width = 128;
        while width > 0 {
            if lane < width { let other = scores[lane + width]; if other > scores[lane] || other != other { scores[lane] = other; } }
            sync_ruda(); width /= 2;
        }
        let rowmax = scores[0usize]; let mut absaimax = real[base + imax * n + imax].abs();
        if comptime!(!hermitian) { absaimax += imag[base + imax * n + imax].abs(); }
        if absakk >= alpha * colmax * (colmax / rowmax) { kp = k; }
        else if absaimax >= alpha * rowmax { kp = imax; }
        else { kp = imax; step = 2; }
    }
    let kk = k + step - 1;
    sync_ruda();
    if active && kp != kk {
        row = kp + 1 + lane;
        while row < n {
            let p = base + row * n + kk; let q = base + row * n + kp;
            let r = real[p]; let i = imag[p]; real[p] = real[q]; imag[p] = imag[q]; real[q] = r; imag[q] = i; row += 256;
        }
        let mut col = kk + 1 + lane;
        while col < kp {
            let p = base + col * n + kk; let q = base + kp * n + col;
            let r = real[p]; let mut i = imag[p]; let mut qi = imag[q]; if comptime!(hermitian) { i = -i; qi = -qi; }
            real[p] = real[q]; imag[p] = qi; real[q] = r; imag[q] = i; col += 256;
        }
        if lane == 0 {
            let p = base + kk * n + kk; let q = base + kp * n + kp;
            let r = real[p]; let i = imag[p]; real[p] = real[q]; imag[p] = imag[q]; real[q] = r; imag[q] = i;
            if comptime!(hermitian) { imag[base + kp * n + kk] = -imag[base + kp * n + kk]; imag[p] = F::new(0.0); imag[q] = F::new(0.0); }
            if step == 2 {
                let p = base + (k + 1) * n + k; let q = base + kp * n + k;
                let r = real[p]; let i = imag[p]; real[p] = real[q]; imag[p] = imag[q]; real[q] = r; imag[q] = i;
            }
        }
    }
    if lane == 0 {
        state[batch * 4 + 1] = kp as i32; state[batch * 4 + 2] = step as i32;
        let mut valid = 0; if active { valid = 1; } state[batch * 4 + 3] = valid;
        if !active && info[batch] == 0 { info[batch] = (k + 1) as i32; }
        if comptime!(hermitian) { imag[diagonal] = F::new(0.0); if step == 2 { imag[base + (k + 1) * n + k + 1] = F::new(0.0); } }
        let mut pivot = (kp + 1) as i32; if step == 2 { pivot = -pivot; pivots[batch * n + k + 1] = pivot; }
        pivots[batch * n + k] = pivot;
    }
}

#[ruda(launch)]
pub(super) fn weights<F: Float>(ar: &Array<F>, ai: &Array<F>, wr: &mut Array<F>, wi: &mut Array<F>, state: &Array<i32>, order: u32, #[comptime] hermitian: bool, #[comptime] wide: bool) {
    let p = block() * 256usize + UNIT_POS as usize; let n = order as usize;
    if p >= wr.len() / 2 { terminate!(); } let batch = p / n; let row = p % n;
    let k = state[batch * 4] as usize; let step = state[batch * 4 + 2] as usize;
    if state[batch * 4 + 3] == 0 || row < k + step { terminate!(); }
    let base = batch * n * n; let d = base + k * n + k; let u = base + row * n + k;
    if step == 1 {
        let (r, i) = divide::<F>(ar[u], ai[u], ar[d], ai[d], wide); wr[2 * p] = r; wi[2 * p] = i;
    } else {
        let d2 = base + (k + 1) * n + k + 1; let e = base + (k + 1) * n + k;
        let (r, i, s, j) = block_solve::<F>(ar[d], ai[d], ar[d2], ai[d2], ar[e], ai[e], ar[u], ai[u], ar[u + 1], ai[u + 1], hermitian, true, wide);
        wr[2 * p] = r; wi[2 * p] = i; wr[2 * p + 1] = s; wi[2 * p + 1] = j;
    }
}

#[ruda(launch)]
pub(super) fn trailing<F: Float>(real: &mut Array<F>, imag: &mut Array<F>, wr: &Array<F>, wi: &Array<F>, state: &Array<i32>, order: u32, #[comptime] hermitian: bool) {
    let p = block() * 256usize + UNIT_POS as usize; let n = order as usize; if p >= real.len() { terminate!(); }
    let batch = p / (n * n); let row = p / n % n; let col = p % n; let k = state[batch * 4] as usize; let step = state[batch * 4 + 2] as usize;
    if state[batch * 4 + 3] == 0 || col < k + step || row < col { terminate!(); }
    let base = batch * n * n;
    for offset in 0..step {
        let a = base + row * n + k + offset; let w = (batch * n + col) * 2 + offset;
        let mut ii = wi[w]; if comptime!(hermitian) { ii = -ii; }
        let (r, i) = product::<F>(real[a], imag[a], wr[w], ii); real[p] -= r; imag[p] -= i;
    }
    if hermitian && row == col { imag[p] = F::new(0.0); }
}

#[ruda(launch)]
pub(super) fn advance<F: Float>(real: &mut Array<F>, imag: &mut Array<F>, wr: &Array<F>, wi: &Array<F>, state: &mut Array<i32>, order: u32) {
    let batch = block(); if batch >= state.len() / 4 { terminate!(); }
    let lane = UNIT_POS as usize; let n = order as usize; let k = state[batch * 4] as usize; if k >= n { terminate!(); }
    let step = state[batch * 4 + 2] as usize;
    if state[batch * 4 + 3] != 0 {
        let mut row = k + step + lane;
        while row < n {
            for offset in 0..step {
                let p = (batch * n + row) * n + k + offset; let q = (batch * n + row) * 2 + offset;
                real[p] = wr[q]; imag[p] = wi[q];
            }
            row += 256;
        }
    }
    sync_ruda(); if lane == 0 { state[batch * 4] = (k + step) as i32; }
}

#[ruda]
fn swap<F: Float>(real: &mut Array<F>, imag: &mut Array<F>, x: usize, y: usize) {
    let r = real[x]; let i = imag[x]; real[x] = real[y]; imag[x] = imag[y]; real[y] = r; imag[y] = i;
}

#[ruda(launch)]
pub(super) fn solve<F: Float>(ar: &Array<F>, ai: &Array<F>, pivots: &Array<i32>, real: &mut Array<F>, imag: &mut Array<F>, info: &mut Array<i32>, order: u32, rhs: u32, #[comptime] hermitian: bool, #[comptime] wide: bool) {
    let work = block() * 256usize + UNIT_POS as usize; let n = order as usize; let nr = rhs as usize;
    if work >= info.len() * nr { terminate!(); } let batch = work / nr; let col = work % nr; let matrix = batch * n * n; let base = batch * n * nr;
    let mut k = 0usize; let mut valid = true;
    while k < n {
        let pivot = pivots[batch * n + k];
        if pivot == 0 || pivot > order as i32 || pivot < -(order as i32) { valid = false; k = n; }
        else if pivot < 0 {
            if k + 1 >= n { valid = false; k = n; }
            else if pivots[batch * n + k + 1] != pivot { valid = false; k = n; }
            else { k += 2; }
        } else { k += 1; }
    }
    if !valid { if col == 0 { info[batch] = -2; } terminate!(); }
    k = 0;
    while k < n {
        let pivot = pivots[batch * n + k]; let mut step = 1usize; let mut kp = pivot;
        if pivot < 0 { step = 2; kp = -pivot; } let kp = kp as usize - 1;
        swap(real, imag, base + (k + step - 1) * nr + col, base + kp * nr + col);
        for row in k + step..n {
            let p = base + row * nr + col;
            for offset in 0..step {
                let a = matrix + row * n + k + offset; let b = base + (k + offset) * nr + col;
                let (r, i) = product::<F>(ar[a], ai[a], real[b], imag[b]); real[p] -= r; imag[p] -= i;
            }
        }
        let d = matrix + k * n + k; let b = base + k * nr + col;
        if step == 1 {
            let mut di = ai[d]; if comptime!(hermitian) { di = F::new(0.0); }
            let (r, i) = divide::<F>(real[b], imag[b], ar[d], di, wide); real[b] = r; imag[b] = i;
        } else {
            let d2 = matrix + (k + 1) * n + k + 1; let e = matrix + (k + 1) * n + k;
            let (r, i, s, j) = block_solve::<F>(ar[d], ai[d], ar[d2], ai[d2], ar[e], ai[e], real[b], imag[b], real[b + nr], imag[b + nr], hermitian, false, wide);
            real[b] = r; imag[b] = i; real[b + nr] = s; imag[b + nr] = j;
        }
        k += step;
    }
    k = n;
    while k > 0 {
        let last = k - 1; let pivot = pivots[batch * n + last]; let mut step = 1usize; let mut kp = pivot;
        if pivot < 0 { step = 2; kp = -pivot; } let kp = kp as usize - 1; let first = k - step;
        for offset in 0..step {
            let column = first + offset; let p = base + column * nr + col;
            let mut r = real[p]; let mut i = imag[p];
            for row in k..n {
                let a = matrix + row * n + column; let b = base + row * nr + col; let mut ii = ai[a]; if comptime!(hermitian) { ii = -ii; }
                let (pr, pi) = product::<F>(ar[a], ii, real[b], imag[b]); r -= pr; i -= pi;
            }
            real[p] = r; imag[p] = i;
        }
        swap(real, imag, base + last * nr + col, base + kp * nr + col); k -= step;
    }
}
