// PyTorch optimized Taylor evaluation; full copyright/license in matrix_exp.rs.
use super::matrix_exp::{T12, T18};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda]
fn maximum<F: Float>(value: F) -> F {
    let lane = UNIT_POS as usize; let mut values = SharedMemory::<F>::new(256);
    values[lane] = value; sync_ruda(); let mut width = 128usize;
    while width > 0 {
        if lane < width { values[lane] = values[lane].max(values[lane + width]); }
        sync_ruda(); width /= 2;
    }
    let result = values[0usize]; sync_ruda(); result
}

#[ruda]
fn coefficient<F: Float>(#[comptime] bits: u64) -> F { F::cast_from(comptime!(f64::from_bits(bits))) }

#[ruda]
fn multiply<F: Float>(
    real: &mut Array<F>, imag: &mut Array<F>, base: usize, n: usize, a: usize, b: usize, out: usize, #[comptime] complex_input: bool,
) {
    let size = n * n; let mut p = UNIT_POS as usize;
    while p < size {
        let row = p / n; let col = p % n; let mut r = F::new(0.0); let mut i = F::new(0.0);
        for k in 0..n {
            let u = base + a * size + row * n + k; let v = base + b * size + k * n + col;
            if comptime!(complex_input) {
                r += real[u] * real[v] - imag[u] * imag[v]; i += real[u] * imag[v] + imag[u] * real[v];
            } else { r += real[u] * real[v]; }
        }
        real[base + out * size + p] = r;
        if comptime!(complex_input) { imag[base + out * size + p] = i; }
        p += 256;
    }
    sync_ruda();
}

#[ruda]
fn combine<F: Float>(
    real: &mut Array<F>, imag: &mut Array<F>, base: usize, n: usize, out: usize,
    a: usize, b: usize, c: usize, d: usize, identity: F, ca: F, cb: F, cc: F, cd: F, #[comptime] complex_input: bool,
) {
    let size = n * n; let mut p = UNIT_POS as usize;
    while p < size {
        let mut diagonal = F::new(0.0); if p / n == p % n { diagonal = identity; }
        let ra = real[base + a * size + p]; let rb = real[base + b * size + p];
        let rc = real[base + c * size + p]; let rd = real[base + d * size + p];
        real[base + out * size + p] = diagonal + ca * ra + cb * rb + cc * rc + cd * rd;
        if comptime!(complex_input) {
            let ia = imag[base + a * size + p]; let ib = imag[base + b * size + p];
            let ic = imag[base + c * size + p]; let id = imag[base + d * size + p];
            imag[base + out * size + p] = ca * ia + cb * ib + cc * ic + cd * id;
        }
        p += 256;
    }
    sync_ruda();
}

#[ruda]
fn add<F: Float>(real: &mut Array<F>, imag: &mut Array<F>, base: usize, size: usize, out: usize, source: usize, #[comptime] complex_input: bool) {
    let mut p = UNIT_POS as usize;
    while p < size {
        real[base + out * size + p] = real[base + out * size + p] + real[base + source * size + p];
        if comptime!(complex_input) { imag[base + out * size + p] = imag[base + out * size + p] + imag[base + source * size + p]; }
        p += 256;
    }
    sync_ruda();
}

#[ruda]
fn mix_powers<F: Float>(
    real: &mut Array<F>, imag: &mut Array<F>, base: usize, n: usize, #[comptime] highest: bool, #[comptime] complex_input: bool,
) {
    let size = n * n; let mut p = UNIT_POS as usize;
    while p < size {
        let a = real[base + p]; let a2 = real[base + size + p]; let a3 = real[base + 2 * size + p];
        let mut a6 = F::new(0.0); if comptime!(highest) { a6 = real[base + 3 * size + p]; }
        let mut ia = F::new(0.0); let mut ia2 = F::new(0.0); let mut ia3 = F::new(0.0); let mut ia6 = F::new(0.0);
        if comptime!(complex_input) {
            ia = imag[base + p]; ia2 = imag[base + size + p]; ia3 = imag[base + 2 * size + p];
            if comptime!(highest) { ia6 = imag[base + 3 * size + p]; }
        }
        let mut diagonal = F::new(0.0); if p / n == p % n { diagonal = F::new(1.0); }
        if comptime!(highest) {
            #[unroll]
            for row in 0..5usize {
                let c0 = F::cast_from(comptime!(T18[row][0])); let c1 = F::cast_from(comptime!(T18[row][1]));
                let c2 = F::cast_from(comptime!(T18[row][2])); let c3 = F::cast_from(comptime!(T18[row][3])); let c4 = F::cast_from(comptime!(T18[row][4]));
                real[base + row * size + p] = c0 * diagonal + c1 * a + c2 * a2 + c3 * a3 + c4 * a6;
                if comptime!(complex_input) { imag[base + row * size + p] = c1 * ia + c2 * ia2 + c3 * ia3 + c4 * ia6; }
            }
        } else {
            #[unroll]
            for row in 0..4usize {
                let c0 = F::cast_from(comptime!(T12[row][0])); let c1 = F::cast_from(comptime!(T12[row][1]));
                let c2 = F::cast_from(comptime!(T12[row][2])); let c3 = F::cast_from(comptime!(T12[row][3]));
                real[base + row * size + p] = c0 * diagonal + c1 * a + c2 * a2 + c3 * a3;
                if comptime!(complex_input) { imag[base + row * size + p] = c1 * ia + c2 * ia2 + c3 * ia3; }
            }
        }
        p += 256;
    }
    sync_ruda();
}

#[ruda(launch)]
pub(super) fn evaluate<S: Float, F: Float>(
    ar: &Tensor<S>, ai: &Tensor<S>, real: &mut Array<F>, imag: &mut Array<F>, out_r: &mut Array<S>, out_i: &mut Array<S>,
    order: u32, batches: u32, #[comptime] complex_input: bool, #[comptime] wide: bool, #[comptime] c8: Vec<u64>,
) {
    let batch = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if batch >= batches as usize { terminate!(); }
    let lane = UNIT_POS as usize; let n = order as usize; let size = n * n; let base = batch * 6 * size; let output = batch * size;
    let zero = F::new(0.0); let one = F::new(1.0);
    if n == 1 {
        if lane == 0 {
            let r = F::cast_from(ar[batch * ar.stride(0)]); let er = r.exp(); let mut x = er; let mut y = zero;
            if comptime!(complex_input) {
                let i = F::cast_from(ai[batch * ai.stride(0)]); y = i;
                if i != zero { x = er * i.cos(); y = er * i.sin(); }
            }
            out_r[output] = S::cast_from(x); out_i[output] = S::cast_from(y);
        }
        terminate!();
    }
    let mut p = lane; let mut invalid = zero;
    while p < size {
        let row = p / n; let col = p % n;
        let r = F::cast_from(ar[batch * ar.stride(0) + row * ar.stride(1) + col * ar.stride(2)]); let mut i = zero;
        if comptime!(complex_input) { i = F::cast_from(ai[batch * ai.stride(0) + row * ai.stride(1) + col * ai.stride(2)]); imag[base + p] = i; }
        real[base + p] = r;
        if r != r || i != i || r.abs() == F::cast_from(f32::INFINITY) || i.abs() == F::cast_from(f32::INFINITY) { invalid = one; }
        p += 256;
    }
    sync_ruda();
    let mut col = lane; let mut norm = zero;
    while col < n {
        let mut total = zero;
        for row in 0..n {
            let p = base + row * n + col; let mut value = real[p].abs();
            if comptime!(complex_input) { value = real[p].hypot(imag[p]); }
            total += value;
        }
        norm = norm.max(total); col += 256;
    }
    norm = maximum::<F>(norm); invalid = maximum::<F>(invalid);
    if invalid != zero || norm == F::cast_from(f32::INFINITY) {
        p = lane;
        while p < size {
            out_r[output + p] = S::cast_from(f32::NAN); out_i[output + p] = S::cast_from(zero);
            if comptime!(complex_input) { if batches > 1 { out_i[output + p] = S::cast_from(f32::NAN); } }
            p += 256;
        }
        terminate!();
    }
    let mut t1 = F::cast_from(1.192092800768788e-07f64); let mut t2 = F::cast_from(5.978858893805233e-04f64);
    let mut t4 = F::cast_from(5.116619363445086e-02f64); let mut t8 = F::cast_from(5.800524627688768e-01f64);
    let mut t12 = F::cast_from(1.461661507209034e+00f64); let mut t18 = F::cast_from(3.010066362817634e+00f64);
    if comptime!(wide) {
        t1 = F::cast_from(2.220446049250313e-16f64); t2 = F::cast_from(2.580956802971767e-08f64);
        t4 = F::cast_from(3.397168839976962e-04f64); t8 = F::cast_from(4.991228871115323e-02f64);
        t12 = F::cast_from(2.996158913811580e-01f64); t18 = F::cast_from(1.090863719290036e+00f64);
    }
    let mut degree = 18u32;
    if batches == 1 {
        if norm <= t1 { degree = 1; } else if norm <= t2 { degree = 2; }
        else if norm <= t4 { degree = 4; } else if norm <= t8 { degree = 8; } else if norm < t12 { degree = 12; }
    }
    let mut squares = 0u32; let mut scale = one;
    if degree == 18 {
        while norm > t18 { norm *= F::new(0.5); scale *= F::new(0.5); squares += 1; }
    }
    p = lane;
    while p < size { real[base + p] *= scale; if comptime!(complex_input) { imag[base + p] *= scale; } p += 256; }
    sync_ruda();
    let mut result = 0usize;
    if degree == 1 {
        combine(real, imag, base, n, 0usize, 0usize, 0usize, 0usize, 0usize, one, one, zero, zero, zero, complex_input);
    } else {
        multiply(real, imag, base, n, 0usize, 0usize, 1usize, complex_input);
        if degree == 2 {
            combine(real, imag, base, n, 1usize, 0usize, 1usize, 0usize, 0usize, one, one, F::new(0.5), zero, zero, complex_input); result = 1;
        } else if degree == 4 {
            combine(real, imag, base, n, 2usize, 0usize, 1usize, 0usize, 0usize, F::new(0.5), F::cast_from(1.0f64 / 6.0), F::cast_from(1.0f64 / 24.0), zero, zero, complex_input);
            multiply(real, imag, base, n, 1usize, 2usize, 3usize, complex_input);
            combine(real, imag, base, n, 3usize, 0usize, 3usize, 0usize, 0usize, one, one, one, zero, zero, complex_input); result = 3;
        } else if degree == 8 {
            let x1 = coefficient::<F>(comptime!(c8[0])); let x2 = coefficient::<F>(comptime!(c8[1])); let x3 = coefficient::<F>(comptime!(c8[2]));
            let x4 = coefficient::<F>(comptime!(c8[3])); let x5 = coefficient::<F>(comptime!(c8[4])); let x6 = coefficient::<F>(comptime!(c8[5]));
            let x7 = coefficient::<F>(comptime!(c8[6])); let y2 = coefficient::<F>(comptime!(c8[7]));
            combine(real, imag, base, n, 2usize, 0usize, 1usize, 0usize, 0usize, zero, x1, x2, zero, zero, complex_input);
            multiply(real, imag, base, n, 1usize, 2usize, 3usize, complex_input);
            combine(real, imag, base, n, 2usize, 1usize, 3usize, 0usize, 0usize, zero, x3, one, zero, zero, complex_input);
            combine(real, imag, base, n, 4usize, 0usize, 1usize, 3usize, 0usize, x4, x5, x6, x7, zero, complex_input);
            multiply(real, imag, base, n, 2usize, 4usize, 3usize, complex_input);
            combine(real, imag, base, n, 3usize, 0usize, 1usize, 3usize, 0usize, one, one, y2, one, zero, complex_input); result = 3;
        } else {
            multiply(real, imag, base, n, 0usize, 1usize, 2usize, complex_input);
            if degree == 12 {
                mix_powers(real, imag, base, n, false, complex_input);
                multiply(real, imag, base, n, 3usize, 3usize, 4usize, complex_input);
                add(real, imag, base, size, 2usize, 4usize, complex_input);
                add(real, imag, base, size, 1usize, 2usize, complex_input);
                multiply(real, imag, base, n, 1usize, 2usize, 4usize, complex_input);
                add(real, imag, base, size, 4usize, 0usize, complex_input); result = 4;
            } else {
                multiply(real, imag, base, n, 2usize, 2usize, 3usize, complex_input);
                mix_powers(real, imag, base, n, true, complex_input);
                multiply(real, imag, base, n, 0usize, 4usize, 5usize, complex_input);
                add(real, imag, base, size, 3usize, 5usize, complex_input);
                add(real, imag, base, size, 2usize, 3usize, complex_input);
                multiply(real, imag, base, n, 2usize, 3usize, 0usize, complex_input);
                add(real, imag, base, size, 0usize, 1usize, complex_input); result = 0;
            }
        }
    }
    for _ in 0..squares {
        let next = 1usize - result;
        multiply(real, imag, base, n, result, result, next, complex_input); result = next;
    }
    p = lane;
    while p < size {
        out_r[output + p] = S::cast_from(real[base + result * size + p]); out_i[output + p] = S::cast_from(zero);
        if comptime!(complex_input) { out_i[output + p] = S::cast_from(imag[base + result * size + p]); }
        p += 256;
    }
}
