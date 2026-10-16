// Complex division adapts LAPACK 3.12.1 SLADIV/DLADIV.
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
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda]
fn position() -> usize {
    (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize
}

#[ruda]
fn ladiv2<F: Float>(a: F, b: F, c: F, d: F, ratio: F, reciprocal: F) -> F {
    let mut value = F::new(0.0);
    if ratio != F::new(0.0) {
        let product = b * ratio;
        if product != F::new(0.0) { value = (a + product) * reciprocal; }
        else { value = a * reciprocal + (b * reciprocal) * ratio; }
    } else { value = (a + d * (b / c)) * reciprocal; }
    value
}

#[ruda]
pub(super) fn divide<F: Float>(a: F, b: F, c: F, d: F, #[comptime] wide: bool) -> (F, F) {
    let mut overflow = F::cast_from(f32::MAX); let mut minimum = F::cast_from(f32::MIN_POSITIVE);
    let mut epsilon = F::cast_from(f32::EPSILON / 2.0);
    if comptime!(wide) {
        overflow = F::cast_from(f64::MAX); minimum = F::cast_from(f64::MIN_POSITIVE);
        epsilon = F::cast_from(f64::EPSILON / 2.0);
    }
    let half = F::new(0.5); let two = F::new(2.0); let boost = two / (epsilon * epsilon);
    let ab = F::max(a.abs(), b.abs()); let cd = F::max(c.abs(), d.abs());
    let mut aa = a; let mut bb = b; let mut cc = c; let mut dd = d; let mut scale = F::new(1.0);
    if ab >= half * overflow { aa *= half; bb *= half; scale *= two; }
    if cd >= half * overflow { cc *= half; dd *= half; scale *= half; }
    if ab <= minimum * two / epsilon { aa *= boost; bb *= boost; scale /= boost; }
    if cd <= minimum * two / epsilon { cc *= boost; dd *= boost; scale *= boost; }
    let mut real = F::new(0.0); let mut imag = F::new(0.0);
    if d.abs() <= c.abs() {
        let ratio = dd / cc; let reciprocal = F::new(1.0) / (cc + dd * ratio);
        real = ladiv2::<F>(aa, bb, cc, dd, ratio, reciprocal);
        imag = ladiv2::<F>(bb, -aa, cc, dd, ratio, reciprocal);
    } else {
        let ratio = cc / dd; let reciprocal = F::new(1.0) / (dd + cc * ratio);
        real = ladiv2::<F>(bb, aa, dd, cc, ratio, reciprocal);
        imag = -ladiv2::<F>(aa, -bb, dd, cc, ratio, reciprocal);
    }
    (real * scale, imag * scale)
}

#[ruda(launch)]
pub(super) fn initialize<F: Float>(
    ar: &Tensor<F>, ai: &Tensor<F>, real: &mut Array<F>, imag: &mut Array<F>, info: &mut Array<i32>,
    rows: u32, cols: u32, #[comptime] mode: u32, #[comptime] upper: bool,
) {
    let p = position();
    if p < info.len() { info[p] = 0; }
    if p < real.len() {
        let m = rows as usize; let n = cols as usize;
        let batch = p / (m * n); let row = p / n % m; let col = p % n;
        let mut r = F::new(0.0); let mut i = F::new(0.0);
        if comptime!(mode == 2) { if row == col { r = F::new(1.0); } }
        else if comptime!(mode == 3) {
            r = ar[batch * ar.stride(0) + col * ar.stride(1) + row * ar.stride(2)];
            i = -ai[batch * ai.stride(0) + col * ai.stride(1) + row * ai.stride(2)];
        } else if comptime!(mode == 1) {
            if row >= col {
                let mut source_row = row; let mut source_col = col;
                if comptime!(upper) { source_row = col; source_col = row; }
                r = ar[batch * ar.stride(0) + source_row * ar.stride(1) + source_col * ar.stride(2)];
                if row != col {
                    i = ai[batch * ai.stride(0) + source_row * ai.stride(1) + source_col * ai.stride(2)];
                    if comptime!(upper) { i = -i; }
                }
            }
        } else {
            r = ar[batch * ar.stride(0) + row * ar.stride(1) + col * ar.stride(2)];
            i = ai[batch * ai.stride(0) + row * ai.stride(1) + col * ai.stride(2)];
        }
        real[p] = r; imag[p] = i;
    }
}

#[ruda(launch)]
pub(super) fn lu_panel<F: Float>(
    real: &mut Array<F>, imag: &mut Array<F>, pivots: &mut Array<i32>, info: &mut Array<i32>,
    rows: u32, cols: u32, column: u32, #[comptime] pivot: bool, #[comptime] wide: bool,
) {
    let batch = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if batch >= info.len() { terminate!(); }
    let lane = UNIT_POS as usize; let m = rows as usize; let n = cols as usize; let j = column as usize;
    let base = batch * m * n; let mut chosen = j;
    let mut scores = SharedMemory::<F>::new(256); let mut indices = SharedMemory::<u32>::new(256);
    if comptime!(pivot) {
        let mut score = F::new(-1.0); let mut index = rows; let mut row = j + lane;
        while row < m {
            let value = real[base + row * n + j].abs() + imag[base + row * n + j].abs();
            if value > score || value != value && score == score { score = value; index = row as u32; }
            row += 256;
        }
        scores[lane] = score; indices[lane] = index;
        sync_ruda();
        let mut width = 128usize;
        while width > 0 {
            if lane < width {
                let other = scores[lane + width]; let own = scores[lane];
                let oi = indices[lane + width]; let ii = indices[lane];
                if other > own || other == own && oi < ii || other != other && (own == own || oi < ii) {
                    scores[lane] = other; indices[lane] = oi;
                }
            }
            sync_ruda(); width /= 2;
        }
        chosen = indices[0usize] as usize;
    }
    if lane == 0 {
        pivots[batch * m.min(n) + j] = chosen as i32;
        if real[base + chosen * n + j] == F::new(0.0) && imag[base + chosen * n + j] == F::new(0.0) && info[batch] == 0 {
            info[batch] = (j + 1) as i32;
        }
    }
    sync_ruda();
    if chosen != j {
        let mut col = lane;
        while col < n {
            let p = base + j * n + col; let q = base + chosen * n + col;
            let r = real[p]; let i = imag[p]; real[p] = real[q]; imag[p] = imag[q]; real[q] = r; imag[q] = i;
            col += 256;
        }
    }
    sync_ruda();
    let dr = real[base + j * n + j]; let di = imag[base + j * n + j];
    if dr != F::new(0.0) || di != F::new(0.0) {
        let mut row = j + 1 + lane;
        while row < m {
            let p = base + row * n + j; let (r, i) = divide::<F>(real[p], imag[p], dr, di, wide);
            real[p] = r; imag[p] = i; row += 256;
        }
    }
}

#[ruda(launch)]
pub(super) fn trailing<F: Float>(
    real: &mut Array<F>, imag: &mut Array<F>, info: &Array<i32>, rows: u32, cols: u32, column: u32,
    #[comptime] cholesky: bool,
) {
    let p = position(); let m = rows as usize; let n = cols as usize; let j = column as usize;
    let height = m - j - 1; let width = n - j - 1;
    if p < info.len() * height * width {
        let batch = p / (height * width); let row = p / width % height + j + 1; let col = p % width + j + 1;
        let base = batch * m * n; let mut active = true;
        if comptime!(cholesky) { active = row >= col && info[batch] == 0; }
        if active {
            let a = real[base + row * n + j]; let b = imag[base + row * n + j];
            let mut index = base + j * n + col;
            if comptime!(cholesky) { index = base + col * n + j; }
            let c = real[index]; let mut d = imag[index];
            if comptime!(cholesky) { d = -d; }
            let target = base + row * n + col;
            real[target] -= a * c - b * d; imag[target] -= a * d + b * c;
        }
    }
}

#[ruda(launch)]
pub(super) fn cholesky_panel<F: Float>(real: &mut Array<F>, imag: &mut Array<F>, info: &mut Array<i32>, order: u32, column: u32) {
    let batch = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if batch >= info.len() { terminate!(); }
    let lane = UNIT_POS as usize; let n = order as usize; let j = column as usize; let base = batch * n * n;
    if lane == 0 && info[batch] == 0 {
        let diagonal = real[base + j * n + j]; imag[base + j * n + j] = F::new(0.0);
        if !(diagonal > F::new(0.0)) { info[batch] = (j + 1) as i32; }
        else { real[base + j * n + j] = diagonal.sqrt(); }
    }
    sync_ruda();
    if info[batch] == 0 {
        let diagonal = real[base + j * n + j]; let mut row = j + 1 + lane;
        while row < n {
            real[base + row * n + j] /= diagonal; imag[base + row * n + j] /= diagonal; row += 256;
        }
    }
}

#[ruda]
fn substitution<F: Float>(
    ar: &Array<F>, ai: &Array<F>, xr: &mut Array<F>, xi: &mut Array<F>, batch: usize, col: usize,
    n: usize, nr: usize, #[comptime] upper: bool, #[comptime] transpose: bool,
    #[comptime] conjugate: bool, #[comptime] unit: bool, #[comptime] wide: bool,
) -> i32 {
    let matrix = batch * n * n; let rhs = batch * n * nr; let mut code = 0i32;
    for step in 0..n {
        let mut row = step;
        if comptime!(upper != transpose) { row = n - 1 - step; }
        let p = rhs + row * nr + col; let mut r = xr[p]; let mut i = xi[p];
        for previous in 0..step {
            let mut k = previous;
            if comptime!(upper != transpose) { k = n - 1 - previous; }
            let mut index = matrix + row * n + k;
            if comptime!(transpose) { index = matrix + k * n + row; }
            let a = ar[index]; let mut b = ai[index];
            if comptime!(conjugate) { b = -b; }
            let c = xr[rhs + k * nr + col]; let d = xi[rhs + k * nr + col];
            r -= a * c - b * d; i -= a * d + b * c;
        }
        if comptime!(!unit) {
            let a = ar[matrix + row * n + row]; let mut b = ai[matrix + row * n + row];
            if comptime!(conjugate) { b = -b; }
            if a == F::new(0.0) && b == F::new(0.0) && (code == 0 || ((row + 1) as i32) < code) { code = (row + 1) as i32; }
            let (rr, ii) = divide::<F>(r, i, a, b, wide); r = rr; i = ii;
        }
        xr[p] = r; xi[p] = i;
    }
    code
}

#[ruda(launch)]
pub(super) fn solve<F: Float>(
    ar: &Array<F>, ai: &Array<F>, pivots: &Array<i32>, xr: &mut Array<F>, xi: &mut Array<F>, info: &mut Array<i32>,
    order: u32, rhs: u32, #[comptime] lu: bool, #[comptime] upper: bool, #[comptime] transpose: bool,
    #[comptime] conjugate: bool, #[comptime] unit: bool, #[comptime] wide: bool,
) {
    let p = position(); let n = order as usize; let nr = rhs as usize;
    if p < info.len() * nr {
        let batch = p / nr; let col = p % nr; let base = batch * n * nr; let mut code = 0i32;
        if comptime!(lu) {
            for row in 0..n { if pivots[batch * n + row] < 0 || pivots[batch * n + row] >= order as i32 { code = -2; } }
            if code == 0 {
                if comptime!(!transpose) {
                    for row in 0..n {
                        let other = pivots[batch * n + row] as usize; let a = base + row * nr + col; let b = base + other * nr + col;
                        let r = xr[a]; let i = xi[a]; xr[a] = xr[b]; xi[a] = xi[b]; xr[b] = r; xi[b] = i;
                    }
                    let _lower = substitution(ar, ai, xr, xi, batch, col, n, nr, false, false, false, true, wide);
                    code = substitution(ar, ai, xr, xi, batch, col, n, nr, true, false, false, false, wide);
                } else {
                    code = substitution(ar, ai, xr, xi, batch, col, n, nr, true, true, conjugate, false, wide);
                    let _lower = substitution(ar, ai, xr, xi, batch, col, n, nr, false, true, conjugate, true, wide);
                    for step in 0..n {
                        let row = n - 1 - step; let other = pivots[batch * n + row] as usize;
                        let a = base + row * nr + col; let b = base + other * nr + col;
                        let r = xr[a]; let i = xi[a]; xr[a] = xr[b]; xi[a] = xi[b]; xr[b] = r; xi[b] = i;
                    }
                }
            }
        } else { code = substitution(ar, ai, xr, xi, batch, col, n, nr, upper, transpose, conjugate, unit, wide); }
        if col == 0 { info[batch] = code; }
    }
}

#[ruda(launch)]
pub(super) fn determinant<F: Float>(
    ar: &Array<F>, ai: &Array<F>, pivots: &Array<i32>, dr: &mut Array<F>, di: &mut Array<F>,
    sr: &mut Array<F>, si: &mut Array<F>, logabs: &mut Array<F>, order: u32,
) {
    let batch = position(); let n = order as usize;
    if batch < dr.len() {
        let mut real = F::new(1.0); let mut imag = F::new(0.0);
        let mut phase_real = F::new(1.0); let mut phase_imag = F::new(0.0); let mut logarithm = F::new(0.0);
        for j in 0..n {
            let p = batch * n * n + j * n + j; let a = ar[p]; let b = ai[p];
            let r = real * a - imag * b; imag = real * b + imag * a; real = r;
            let magnitude = a.hypot(b); let mut x = F::new(0.0); let mut y = F::new(0.0);
            if magnitude != F::new(0.0) { x = a / magnitude; y = b / magnitude; }
            let sign_r = phase_real * x - phase_imag * y;
            phase_imag = phase_real * y + phase_imag * x; phase_real = sign_r;
            logarithm += magnitude.ln();
            if pivots[batch * n + j] != j as i32 { real = -real; imag = -imag; phase_real = -phase_real; phase_imag = -phase_imag; }
        }
        dr[batch] = real; di[batch] = imag; sr[batch] = phase_real; si[batch] = phase_imag; logabs[batch] = logarithm;
    }
}

#[ruda(launch)]
pub(super) fn unpack<F: Float>(
    ar: &Array<F>, ai: &Array<F>, pivots: &Array<i32>, real: &mut Array<F>, imag: &mut Array<F>,
    rows: u32, cols: u32, #[comptime] kind: u32,
) {
    let p = position(); let m = rows as usize; let n = cols as usize; let k = m.min(n);
    if p < real.len() {
        let mut out_rows = m; let mut out_cols = k;
        if comptime!(kind == 0) { out_cols = m; }
        else if comptime!(kind == 2) { out_rows = k; out_cols = n; }
        let batch = p / (out_rows * out_cols); let row = p / out_cols % out_rows; let col = p % out_cols;
        let mut r = F::new(0.0); let mut i = F::new(0.0);
        if comptime!(kind == 0) {
            let mut target = row;
            for j in 0..k {
                let other = pivots[batch * k + j];
                if other >= 0 && other < rows as i32 {
                    if target == j { target = other as usize; } else if target == other as usize { target = j; }
                }
            }
            if target == col { r = F::new(1.0); }
        } else if comptime!(kind == 1) {
            if row == col { r = F::new(1.0); }
            else if row > col { r = ar[batch * m * n + row * n + col]; i = ai[batch * m * n + row * n + col]; }
        } else if row <= col { r = ar[batch * m * n + row * n + col]; i = ai[batch * m * n + row * n + col]; }
        real[p] = r; imag[p] = i;
    }
}
