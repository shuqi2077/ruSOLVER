use super::{BatchedLuResult, BatchedQrResult, DeviceSolverError};
use ruda_core::{device::Device, tensor::DType};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{calculate_ruda_count_elemwise, prelude::*};
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_contiguous_dtype};

pub(super) trait HostFloat: Float + RudaElement {
    fn scalar(value: f64) -> Self;
}
impl HostFloat for f32 {
    fn scalar(value: f64) -> Self {
        value as f32
    }
}
impl HostFloat for f64 {
    fn scalar(value: f64) -> Self {
        value
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LargeSolverOptions {
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub block_size: usize,
}
impl Default for LargeSolverOptions {
    fn default() -> Self {
        Self {
            absolute_tolerance: 0.0,
            relative_tolerance: 1e-12,
            block_size: 32,
        }
    }
}
pub(super) fn validate<R: Runtime>(
    a: &RudaTensor<R>,
) -> Result<(usize, usize, usize), DeviceSolverError> {
    if !matches!(a.dtype, DType::F32 | DType::F64) || a.qparams.is_some() || a.meta.num_dims() != 3
    {
        return Err(DeviceSolverError::InvalidInput(
            "unquantized F32/F64 [batch,m,n] required",
        ));
    }
    super::elements(a.meta.shape())?;
    let s = a.meta.shape();
    if s[1] == 0 || s[2] == 0 || s.iter().any(|&n| n > u32::MAX as usize) {
        return Err(DeviceSolverError::InvalidInput(
            "positive matrix dimensions required",
        ));
    }
    if a.meta.strides().len() != 3 {
        return Err(DeviceSolverError::InvalidInput(
            "three matrix strides required",
        ));
    }
    if s[0] > 0 {
        let width = if a.dtype == DType::F64 {
            8usize
        } else {
            4usize
        };
        let span = s
            .iter()
            .zip(a.meta.strides().iter())
            .try_fold(0usize, |sum, (&size, &stride)| {
                (size - 1)
                    .checked_mul(stride)
                    .and_then(|x| sum.checked_add(x))
            })
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(width))
            .ok_or(DeviceSolverError::SizeOverflow)?;
        let start = a.handle.offset_start.unwrap_or(0);
        let available = a
            .handle
            .size()
            .checked_sub(start)
            .and_then(|n| n.checked_sub(a.handle.offset_end.unwrap_or(0)))
            .ok_or(DeviceSolverError::SizeOverflow)?;
        if start % width as u64 != 0 || span as u64 > available {
            return Err(DeviceSolverError::InvalidInput(
                "matrix layout exceeds/misaligns backing allocation",
            ));
        }
    }
    Ok((s[0], s[1], s[2]))
}
pub(super) fn alloc<R: Runtime>(
    a: &RudaTensor<R>,
    shape: impl Into<ruda_core::tensor::Shape>,
    dtype: DType,
) -> RudaTensor<R> {
    empty_device_contiguous_dtype(a.client.clone(), a.device.clone(), shape.into(), dtype)
}
fn options(o: LargeSolverOptions) -> Result<(), DeviceSolverError> {
    if !o.absolute_tolerance.is_finite()
        || !o.relative_tolerance.is_finite()
        || o.absolute_tolerance < 0.0
        || o.relative_tolerance < 0.0
        || o.block_size == 0
        || o.block_size > 256
    {
        return Err(DeviceSolverError::InvalidInput(
            "finite nonnegative tolerances and block_size=1..256 required",
        ));
    }
    Ok(())
}

/// Right-looking blocked partial-pivot LU. P A=L U; pivots are sequential,
/// zero-based row interchanges. Positive info identifies a singular pivot;
/// -1 nonfinite input, -3 arithmetic failure. Input tensors are unchanged.
pub fn lu_solve_blocked<R: Runtime>(
    a: &RudaTensor<R>,
    b: &RudaTensor<R>,
    o: LargeSolverOptions,
) -> Result<BatchedLuResult<R>, DeviceSolverError> {
    let (batch, n, n2) = validate(a)?;
    let (bb, m, nrhs) = validate(b)?;
    options(o)?;
    if n != n2 || bb != batch || m != n || a.dtype != b.dtype {
        return Err(DeviceSolverError::InvalidInput(
            "A=[batch,n,n], B=[batch,n,nrhs], matching dtype required",
        ));
    }
    if a.device.to_id() != b.device.to_id() || !a.client.same_execution_queue(&b.client) {
        return Err(DeviceSolverError::DifferentQueue);
    }
    let mut result = BatchedLuResult {
        packed_lu: alloc(a, [batch, n, n], a.dtype),
        pivots: alloc(a, [batch, n], DType::I32),
        solution: alloc(a, [batch, n, nrhs], a.dtype),
        info: alloc(a, [batch], DType::I32),
        submitted_kernels: 0,
    };
    if batch > 0 {
        match a.dtype {
            DType::F32 => lu_submit::<f32, R>(a, b, o, &mut result, n, nrhs),
            DType::F64 => lu_submit::<f64, R>(a, b, o, &mut result, n, nrhs),
            _ => unreachable!(),
        }
    }
    Ok(result)
}
fn lu_submit<F: HostFloat, R: Runtime>(
    a: &RudaTensor<R>,
    b: &RudaTensor<R>,
    o: LargeSolverOptions,
    r: &mut BatchedLuResult<R>,
    n: usize,
    nr: usize,
) {
    let batch = a.meta.shape()[0];
    let d = RudaDim::new(a.client.properties(), batch);
    lu_init::launch::<F, R>(
        &a.client,
        calculate_ruda_count_elemwise(&a.client, batch, d),
        d,
        a.clone().into_tensor_arg(),
        b.clone().into_tensor_arg(),
        r.packed_lu.clone().into_array_arg(),
        r.solution.clone().into_array_arg(),
        r.pivots.clone().into_array_arg(),
        r.info.clone().into_array_arg(),
        n as u32,
        nr as u32,
    );
    r.submitted_kernels += 1;
    for first in (0..n).step_by(o.block_size) {
        let end = (first + o.block_size).min(n);
        lu_panel::launch::<F, R>(
            &a.client,
            calculate_ruda_count_elemwise(&a.client, batch, d),
            d,
            r.packed_lu.clone().into_array_arg(),
            r.solution.clone().into_array_arg(),
            r.pivots.clone().into_array_arg(),
            r.info.clone().into_array_arg(),
            n as u32,
            nr as u32,
            first as u32,
            end as u32,
            F::scalar(o.absolute_tolerance),
            F::scalar(o.relative_tolerance),
        );
        r.submitted_kernels += 1;
        if end < n {
            let columns = batch * (n - end);
            let cd = RudaDim::new(a.client.properties(), columns);
            lu_upper::launch::<F, R>(
                &a.client,
                calculate_ruda_count_elemwise(&a.client, columns, cd),
                cd,
                r.packed_lu.clone().into_array_arg(),
                r.info.clone().into_array_arg(),
                n as u32,
                first as u32,
                end as u32,
            );
            let count = columns * (n - end);
            let ud = RudaDim::new(a.client.properties(), count);
            lu_trailing::launch::<F, R>(
                &a.client,
                calculate_ruda_count_elemwise(&a.client, count, ud),
                ud,
                r.packed_lu.clone().into_array_arg(),
                r.info.clone().into_array_arg(),
                n as u32,
                first as u32,
                end as u32,
            );
            r.submitted_kernels += 2;
        }
    }
    let count = batch * nr;
    let sd = RudaDim::new(a.client.properties(), count);
    lu_solve::launch::<F, R>(
        &a.client,
        calculate_ruda_count_elemwise(&a.client, count, sd),
        sd,
        r.packed_lu.clone().into_array_arg(),
        r.solution.clone().into_array_arg(),
        r.info.clone().into_array_arg(),
        n as u32,
        nr as u32,
    );
    lu_finish::launch::<F, R>(
        &a.client,
        calculate_ruda_count_elemwise(&a.client, batch, d),
        d,
        r.packed_lu.clone().into_array_arg(),
        r.solution.clone().into_array_arg(),
        r.pivots.clone().into_array_arg(),
        r.info.clone().into_array_arg(),
        n as u32,
        nr as u32,
    );
    r.submitted_kernels += 2;
}

/// Blocked Householder thin QR (m>=n), without column pivoting. Each panel
/// stores reflectors; trailing columns and Q columns are updated in parallel.
/// Positive info is the first numerical rank failure; -1/-3 are nonfinite
/// input/arithmetic failures. Failed Q/R are zeroed.
pub fn qr_blocked<R: Runtime>(
    a: &RudaTensor<R>,
    o: LargeSolverOptions,
) -> Result<BatchedQrResult<R>, DeviceSolverError> {
    let (batch, m, n) = validate(a)?;
    options(o)?;
    if m < n {
        return Err(DeviceSolverError::InvalidInput("thin QR requires m>=n"));
    }
    super::elements(&[batch, n, n])?;
    let mut r = BatchedQrResult {
        q: alloc(a, [batch, m, n], a.dtype),
        r: alloc(a, [batch, n, n], a.dtype),
        info: alloc(a, [batch], DType::I32),
        submitted_kernels: 0,
    };
    if batch > 0 {
        match a.dtype {
            DType::F32 => qr_submit::<f32, R>(a, o, &mut r, m, n),
            DType::F64 => qr_submit::<f64, R>(a, o, &mut r, m, n),
            _ => unreachable!(),
        }
    }
    Ok(r)
}
fn qr_submit<F: HostFloat, R: Runtime>(
    a: &RudaTensor<R>,
    o: LargeSolverOptions,
    r: &mut BatchedQrResult<R>,
    m: usize,
    n: usize,
) {
    let batch = a.meta.shape()[0];
    let work = alloc(a, [batch, m, n], a.dtype);
    let tau = alloc(a, [batch, n], a.dtype);
    let scales = alloc(a, [batch], a.dtype);
    let d = RudaDim::new(a.client.properties(), batch);
    qr_init::launch::<F, R>(
        &a.client,
        calculate_ruda_count_elemwise(&a.client, batch, d),
        d,
        a.clone().into_tensor_arg(),
        work.clone().into_array_arg(),
        r.q.clone().into_array_arg(),
        tau.clone().into_array_arg(),
        scales.clone().into_array_arg(),
        r.info.clone().into_array_arg(),
        m as u32,
        n as u32,
    );
    r.submitted_kernels += 1;
    for first in (0..n).step_by(o.block_size) {
        let end = (first + o.block_size).min(n);
        qr_panel::launch::<F, R>(
            &a.client,
            calculate_ruda_count_elemwise(&a.client, batch, d),
            d,
            work.clone().into_array_arg(),
            tau.clone().into_array_arg(),
            scales.clone().into_array_arg(),
            r.info.clone().into_array_arg(),
            m as u32,
            n as u32,
            first as u32,
            end as u32,
            F::scalar(o.absolute_tolerance),
            F::scalar(o.relative_tolerance),
        );
        r.submitted_kernels += 1;
        if end < n {
            let count = batch * (n - end);
            let td = RudaDim::new(a.client.properties(), count);
            qr_apply::launch::<F, R>(
                &a.client,
                calculate_ruda_count_elemwise(&a.client, count, td),
                td,
                work.clone().into_array_arg(),
                tau.clone().into_array_arg(),
                r.info.clone().into_array_arg(),
                m as u32,
                n as u32,
                first as u32,
                end as u32,
            );
            r.submitted_kernels += 1;
        }
    }
    let count = batch * n;
    let qd = RudaDim::new(a.client.properties(), count);
    qr_form::launch::<F, R>(
        &a.client,
        calculate_ruda_count_elemwise(&a.client, count, qd),
        qd,
        work.clone().into_array_arg(),
        tau.into_array_arg(),
        r.q.clone().into_array_arg(),
        r.r.clone().into_array_arg(),
        r.info.clone().into_array_arg(),
        m as u32,
        n as u32,
    );
    qr_finish::launch::<F, R>(
        &a.client,
        calculate_ruda_count_elemwise(&a.client, batch, d),
        d,
        r.q.clone().into_array_arg(),
        r.r.clone().into_array_arg(),
        r.info.clone().into_array_arg(),
        m as u32,
        n as u32,
    );
    r.submitted_kernels += 2;
}

#[ruda]
pub(super) fn bad<F: Float>(x: F) -> bool {
    bool::cast_from(x.is_nan()) || bool::cast_from(x.is_inf())
}
#[ruda(launch)]
fn lu_init<F: Float>(
    a: &Tensor<F>,
    b: &Tensor<F>,
    lu: &mut Array<F>,
    x: &mut Array<F>,
    p: &mut Array<i32>,
    info: &mut Array<i32>,
    order: u32,
    rhs: u32,
) {
    let sys = ABSOLUTE_POS;
    if sys >= info.len() {
        terminate!();
    }
    let n = order as usize;
    let nr = rhs as usize;
    let ab = sys * n * n;
    let bb = sys * n * nr;
    let mut code = 0i32;
    for i in 0..n * n {
        let z = a[sys * a.stride(0) + (i / n) * a.stride(1) + (i % n) * a.stride(2)];
        lu[ab + i] = z;
        if bad::<F>(z) {
            code = -1;
        }
    }
    for i in 0..n * nr {
        let z = b[sys * b.stride(0) + (i / nr) * b.stride(1) + (i % nr) * b.stride(2)];
        x[bb + i] = z;
        if bad::<F>(z) {
            code = -1;
        }
    }
    for i in 0..n {
        p[sys * n + i] = -1;
    }
    info[sys] = code;
}
#[ruda(launch)]
fn lu_panel<F: Float + RudaElement>(
    lu: &mut Array<F>,
    x: &mut Array<F>,
    p: &mut Array<i32>,
    info: &mut Array<i32>,
    order: u32,
    rhs: u32,
    first: u32,
    last: u32,
    atol: F,
    rtol: F,
) {
    let sys = ABSOLUTE_POS;
    if sys >= info.len() {
        terminate!();
    }
    if info[sys] != 0 {
        terminate!();
    }
    let n = order as usize;
    let nr = rhs as usize;
    let base = sys * n * n;
    let xb = sys * n * nr;
    let start = first as usize;
    let end = last as usize;
    let mut code = 0i32;
    for k in start..end {
        if code == 0 {
            let mut pivot = k;
            let mut peak = lu[base + k * n + k].abs();
            let mut scale = F::new(0.0);
            for i in k..n {
                let z = lu[base + i * n + k].abs();
                if z > peak {
                    peak = z;
                    pivot = i;
                }
                for j in k..end {
                    scale = F::max(scale, lu[base + i * n + j].abs());
                }
            }
            if bad::<F>(peak) {
                code = -3;
            } else if peak <= atol + rtol * scale {
                code = (k + 1) as i32;
            } else {
                p[sys * n + k] = pivot as i32;
                if pivot != k {
                    for j in 0..n {
                        let z = lu[base + k * n + j];
                        lu[base + k * n + j] = lu[base + pivot * n + j];
                        lu[base + pivot * n + j] = z;
                    }
                    for j in 0..nr {
                        let z = x[xb + k * nr + j];
                        x[xb + k * nr + j] = x[xb + pivot * nr + j];
                        x[xb + pivot * nr + j] = z;
                    }
                }
                let diagonal = lu[base + k * n + k];
                for i in k + 1..n {
                    let z = lu[base + i * n + k] / diagonal;
                    lu[base + i * n + k] = z;
                    if bad::<F>(z) {
                        code = -3;
                    }
                    for j in k + 1..end {
                        let update = z * lu[base + k * n + j];
                        lu[base + i * n + j] -= update;
                    }
                }
            }
        }
    }
    info[sys] = code;
}
#[ruda(launch)]
fn lu_upper<F: Float>(lu: &mut Array<F>, info: &Array<i32>, order: u32, first: u32, last: u32) {
    let n = order as usize;
    let end = last as usize;
    let tail = n - end;
    let pos = ABSOLUTE_POS;
    let sys = pos / tail;
    if sys >= info.len() {
        terminate!();
    }
    if info[sys] != 0 {
        terminate!();
    }
    let col = end + pos % tail;
    let base = sys * n * n;
    for i in first as usize..end {
        let mut z = lu[base + i * n + col];
        for k in first as usize..i {
            z -= lu[base + i * n + k] * lu[base + k * n + col];
        }
        lu[base + i * n + col] = z;
    }
}
#[ruda(launch)]
fn lu_trailing<F: Float>(lu: &mut Array<F>, info: &Array<i32>, order: u32, first: u32, last: u32) {
    let n = order as usize;
    let end = last as usize;
    let tail = n - end;
    let pos = ABSOLUTE_POS;
    let sys = pos / (tail * tail);
    if sys >= info.len() {
        terminate!();
    }
    if info[sys] != 0 {
        terminate!();
    }
    let row = end + (pos / tail) % tail;
    let col = end + pos % tail;
    let base = sys * n * n;
    let mut z = lu[base + row * n + col];
    for k in first as usize..end {
        z -= lu[base + row * n + k] * lu[base + k * n + col];
    }
    lu[base + row * n + col] = z;
}
#[ruda(launch)]
fn lu_solve<F: Float>(lu: &Array<F>, x: &mut Array<F>, info: &Array<i32>, order: u32, rhs: u32) {
    let n = order as usize;
    let nr = rhs as usize;
    let pos = ABSOLUTE_POS;
    let sys = pos / nr;
    if sys >= info.len() {
        terminate!();
    }
    if info[sys] != 0 {
        terminate!();
    }
    let col = pos % nr;
    let ab = sys * n * n;
    let bb = sys * n * nr;
    for i in 0..n {
        let mut z = x[bb + i * nr + col];
        for k in 0..i {
            z -= lu[ab + i * n + k] * x[bb + k * nr + col];
        }
        x[bb + i * nr + col] = z;
    }
    let mut end = n;
    while end > 0 {
        let i = end - 1;
        let mut z = x[bb + i * nr + col];
        for k in i + 1..n {
            z -= lu[ab + i * n + k] * x[bb + k * nr + col];
        }
        x[bb + i * nr + col] = z / lu[ab + i * n + i];
        end -= 1;
    }
}
#[ruda(launch)]
fn lu_finish<F: Float>(
    lu: &mut Array<F>,
    x: &mut Array<F>,
    p: &mut Array<i32>,
    info: &mut Array<i32>,
    order: u32,
    rhs: u32,
) {
    let sys = ABSOLUTE_POS;
    if sys >= info.len() {
        terminate!();
    }
    let n = order as usize;
    let nr = rhs as usize;
    let ab = sys * n * n;
    let bb = sys * n * nr;
    let mut code = info[sys];
    if code == 0 {
        for i in 0..n * n {
            if bad::<F>(lu[ab + i]) {
                code = -3;
            }
        }
        for i in 0..n * nr {
            if bad::<F>(x[bb + i]) {
                code = -3;
            }
        }
    }
    if code != 0 {
        for i in 0..n * n {
            lu[ab + i] = F::new(0.0);
        }
        for i in 0..n * nr {
            x[bb + i] = F::new(0.0);
        }
        for i in 0..n {
            p[sys * n + i] = -1;
        }
    }
    info[sys] = code;
}
#[ruda(launch)]
fn qr_init<F: Float>(
    a: &Tensor<F>,
    w: &mut Array<F>,
    q: &mut Array<F>,
    tau: &mut Array<F>,
    scales: &mut Array<F>,
    info: &mut Array<i32>,
    rows: u32,
    cols: u32,
) {
    let sys = ABSOLUTE_POS;
    if sys >= info.len() {
        terminate!();
    }
    let m = rows as usize;
    let n = cols as usize;
    let base = sys * m * n;
    let mut code = 0i32;
    let mut scale = F::new(0.0);
    for i in 0..m * n {
        let z = a[sys * a.stride(0) + (i / n) * a.stride(1) + (i % n) * a.stride(2)];
        w[base + i] = z;
        q[base + i] = F::new(0.0);
        scale = F::max(scale, z.abs());
        if bad::<F>(z) {
            code = -1;
        }
    }
    for i in 0..n {
        tau[sys * n + i] = F::new(0.0);
        q[base + i * n + i] = F::new(1.0);
    }
    info[sys] = code;
    scales[sys] = scale;
}
#[ruda(launch)]
fn qr_panel<F: Float + RudaElement>(
    w: &mut Array<F>,
    tau: &mut Array<F>,
    scales: &Array<F>,
    info: &mut Array<i32>,
    rows: u32,
    cols: u32,
    first: u32,
    last: u32,
    atol: F,
    rtol: F,
) {
    let sys = ABSOLUTE_POS;
    if sys >= info.len() {
        terminate!();
    }
    if info[sys] != 0 {
        terminate!();
    }
    let m = rows as usize;
    let n = cols as usize;
    let base = sys * m * n;
    let end = last as usize;
    let mut code = 0i32;
    let scale = scales[sys];
    for k in first as usize..end {
        if code == 0 {
            let mut norm_scale = F::new(0.0);
            for i in k..m {
                norm_scale = F::max(norm_scale, w[base + i * n + k].abs());
            }
            let mut norm = F::new(0.0);
            if norm_scale > F::new(0.0) {
                for i in k..m {
                    let z = w[base + i * n + k] / norm_scale;
                    norm += z * z;
                }
                norm = norm.sqrt() * norm_scale;
            }
            if bad::<F>(norm) {
                code = -3;
            } else if norm <= atol + rtol * scale {
                code = (k + 1) as i32;
            } else {
                let alpha = w[base + k * n + k];
                let mut beta = norm;
                if alpha >= F::new(0.0) {
                    beta = -norm;
                }
                let t = (beta - alpha) / beta;
                tau[sys * n + k] = t;
                let divisor = alpha - beta;
                for i in k + 1..m {
                    w[base + i * n + k] /= divisor;
                }
                w[base + k * n + k] = beta;
                for j in k + 1..end {
                    let mut dot = w[base + k * n + j];
                    for i in k + 1..m {
                        dot += w[base + i * n + k] * w[base + i * n + j];
                    }
                    dot *= t;
                    w[base + k * n + j] -= dot;
                    for i in k + 1..m {
                        let update = w[base + i * n + k] * dot;
                        w[base + i * n + j] -= update;
                    }
                }
            }
        }
    }
    info[sys] = code;
}
#[ruda(launch)]
fn qr_apply<F: Float>(
    w: &mut Array<F>,
    tau: &Array<F>,
    info: &Array<i32>,
    rows: u32,
    cols: u32,
    first: u32,
    last: u32,
) {
    let m = rows as usize;
    let n = cols as usize;
    let end = last as usize;
    let tail = n - end;
    let pos = ABSOLUTE_POS;
    let sys = pos / tail;
    if sys >= info.len() {
        terminate!();
    }
    if info[sys] != 0 {
        terminate!();
    }
    let j = end + pos % tail;
    let base = sys * m * n;
    for k in first as usize..end {
        let mut dot = w[base + k * n + j];
        for i in k + 1..m {
            dot += w[base + i * n + k] * w[base + i * n + j];
        }
        dot *= tau[sys * n + k];
        w[base + k * n + j] -= dot;
        for i in k + 1..m {
            let update = w[base + i * n + k] * dot;
            w[base + i * n + j] -= update;
        }
    }
}
#[ruda(launch)]
fn qr_form<F: Float>(
    w: &Array<F>,
    tau: &Array<F>,
    q: &mut Array<F>,
    r: &mut Array<F>,
    info: &Array<i32>,
    rows: u32,
    cols: u32,
) {
    let m = rows as usize;
    let n = cols as usize;
    let pos = ABSOLUTE_POS;
    let sys = pos / n;
    if sys >= info.len() {
        terminate!();
    }
    let j = pos % n;
    let base = sys * m * n;
    let rb = sys * n * n;
    for i in 0..n {
        let mut z = F::new(0.0);
        if info[sys] == 0 && i <= j {
            z = w[base + i * n + j];
        }
        r[rb + i * n + j] = z;
    }
    if info[sys] != 0 {
        terminate!();
    }
    let mut end = n;
    while end > 0 {
        let k = end - 1;
        let mut dot = q[base + k * n + j];
        for i in k + 1..m {
            dot += w[base + i * n + k] * q[base + i * n + j];
        }
        dot *= tau[sys * n + k];
        q[base + k * n + j] -= dot;
        for i in k + 1..m {
            q[base + i * n + j] -= w[base + i * n + k] * dot;
        }
        end -= 1;
    }
}
#[ruda(launch)]
fn qr_finish<F: Float>(
    q: &mut Array<F>,
    r: &mut Array<F>,
    info: &mut Array<i32>,
    rows: u32,
    cols: u32,
) {
    let sys = ABSOLUTE_POS;
    if sys >= info.len() {
        terminate!();
    }
    let m = rows as usize;
    let n = cols as usize;
    let base = sys * m * n;
    let rb = sys * n * n;
    let mut code = info[sys];
    if code == 0 {
        for i in 0..m * n {
            if bad::<F>(q[base + i]) {
                code = -3;
            }
        }
        for i in 0..n * n {
            if bad::<F>(r[rb + i]) {
                code = -3;
            }
        }
    }
    if code != 0 {
        for i in 0..m * n {
            q[base + i] = F::new(0.0);
        }
        for i in 0..n * n {
            r[rb + i] = F::new(0.0);
        }
    }
    info[sys] = code;
}
