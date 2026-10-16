use super::{
    DeviceSolverError, LargeSolverOptions,
    large::{HostFloat, alloc},
};
use ruda_core::tensor::DType;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{calculate_ruda_count_elemwise, prelude::*};
use ruda_kernel::tensor::{RudaTensor, contiguous::into_contiguous, readback::into_data};

pub(super) fn shape<R: Runtime>(
    a: &RudaTensor<R>,
) -> Result<(usize, usize, usize), DeviceSolverError> {
    if !matches!(a.dtype, DType::F32 | DType::F64) || a.qparams.is_some() || a.meta.num_dims() != 3
    {
        return Err(DeviceSolverError::InvalidInput(
            "real F32/F64 [batch,m,n] required",
        ));
    }
    super::elements(a.meta.shape())?;
    let s = a.meta.shape();
    if s.iter().any(|&n| n > u32::MAX as usize) {
        return Err(DeviceSolverError::SizeOverflow);
    }
    if s[1] > 0 && s[2] > 0 {
        super::large::validate(a)?;
    }
    Ok((s[0], s[1], s[2]))
}
pub(super) fn same<R: Runtime>(
    a: &RudaTensor<R>,
    b: &RudaTensor<R>,
) -> Result<(), DeviceSolverError> {
    if a.dtype != b.dtype || !a.client.same_execution_queue(&b.client) {
        return Err(DeviceSolverError::DifferentQueue);
    }
    Ok(())
}
pub(super) fn check<R: Runtime>(info: &RudaTensor<R>) -> Result<(), DeviceSolverError> {
    let d = ruda_core::future::block_on(into_data(info.clone()))
        .map_err(|e| DeviceSolverError::Readback(format!("{e:?}")))?;
    for (batch, &info) in d
        .to_vec::<i32>()
        .map_err(|e| DeviceSolverError::Readback(format!("{e:?}")))?
        .iter()
        .enumerate()
    {
        if info != 0 {
            return Err(DeviceSolverError::MatrixFailure { batch, info });
        }
    }
    Ok(())
}
pub struct LuFactorResult<R: Runtime> {
    pub packed_lu: RudaTensor<R>,
    pub pivots: RudaTensor<R>,
    pub info: RudaTensor<R>,
    pub submitted_kernels: usize,
}
impl<R: Runtime> LuFactorResult<R> {
    pub fn check_status_sync(&self) -> Result<(), DeviceSolverError> {
        check(&self.info)
    }
}
/// Rectangular right-looking LU; sequential zero-based pivots, P A=L U.
/// Singular pivots retain the computed factor and the first positive info,
/// allowing determinant/log-determinant to represent singular input exactly.
pub fn lu_factor_blocked<R: Runtime>(
    a: &RudaTensor<R>,
    pivot: bool,
    o: LargeSolverOptions,
) -> Result<LuFactorResult<R>, DeviceSolverError> {
    let (batch, m, n) = shape(a)?;
    if o.block_size == 0
        || o.block_size > 256
        || !o.absolute_tolerance.is_finite()
        || !o.relative_tolerance.is_finite()
        || o.absolute_tolerance < 0.0
        || o.relative_tolerance < 0.0
    {
        return Err(DeviceSolverError::InvalidInput("LU factor block/tolerance"));
    }
    let k = m.min(n);
    let mut r = LuFactorResult {
        packed_lu: alloc(a, [batch, m, n], a.dtype),
        pivots: alloc(a, [batch, k], DType::I32),
        info: alloc(a, [batch], DType::I32),
        submitted_kernels: 0,
    };
    if batch > 0 {
        match a.dtype {
            DType::F32 => lu_factor_submit::<f32, R>(a, pivot, o, &mut r, m, n),
            DType::F64 => lu_factor_submit::<f64, R>(a, pivot, o, &mut r, m, n),
            _ => unreachable!(),
        }
    }
    Ok(r)
}
fn lu_factor_submit<F: HostFloat, R: Runtime>(
    a: &RudaTensor<R>,
    pivot: bool,
    o: LargeSolverOptions,
    r: &mut LuFactorResult<R>,
    m: usize,
    n: usize,
) {
    let batch = a.meta.shape()[0];
    let k = m.min(n);
    let dim = RudaDim::new(a.client.properties(), batch);
    factor_init::launch::<F, R>(
        &a.client,
        calculate_ruda_count_elemwise(&a.client, batch, dim),
        dim,
        a.clone().into_tensor_arg(),
        r.packed_lu.clone().into_array_arg(),
        r.pivots.clone().into_array_arg(),
        r.info.clone().into_array_arg(),
        m as u32,
        n as u32,
    );
    r.submitted_kernels += 1;
    for first in (0..k).step_by(o.block_size) {
        let end = (first + o.block_size).min(k);
        factor_panel::launch::<F, R>(
            &a.client,
            calculate_ruda_count_elemwise(&a.client, batch, dim),
            dim,
            r.packed_lu.clone().into_array_arg(),
            r.pivots.clone().into_array_arg(),
            r.info.clone().into_array_arg(),
            m as u32,
            n as u32,
            first as u32,
            end as u32,
            pivot,
            F::scalar(o.absolute_tolerance),
            F::scalar(o.relative_tolerance),
        );
        r.submitted_kernels += 1;
        if end < n {
            let count = batch * (n - end);
            let d = RudaDim::new(a.client.properties(), count);
            factor_upper::launch::<F, R>(
                &a.client,
                calculate_ruda_count_elemwise(&a.client, count, d),
                d,
                r.packed_lu.clone().into_array_arg(),
                r.info.clone().into_array_arg(),
                m as u32,
                n as u32,
                first as u32,
                end as u32,
            );
            r.submitted_kernels += 1;
            if end < m {
                let count = count * (m - end);
                let d = RudaDim::new(a.client.properties(), count);
                factor_update::launch::<F, R>(
                    &a.client,
                    calculate_ruda_count_elemwise(&a.client, count, d),
                    d,
                    r.packed_lu.clone().into_array_arg(),
                    r.info.clone().into_array_arg(),
                    m as u32,
                    n as u32,
                    first as u32,
                    end as u32,
                );
                r.submitted_kernels += 1;
            }
        }
    }
}
pub struct QrFactorResult<R: Runtime> {
    pub packed: RudaTensor<R>,
    pub tau: RudaTensor<R>,
    pub info: RudaTensor<R>,
    pub submitted_kernels: usize,
}
impl<R: Runtime> QrFactorResult<R> {
    pub fn check_status_sync(&self) -> Result<(), DeviceSolverError> {
        check(&self.info)
    }
}
/// Rectangular Householder factorization. Zero-norm columns use the identity
/// reflector; info reports the first numerical rank deficiency, not a failed
/// QR factorization. No column pivoting or normal-equation substitution.
pub fn qr_factor_blocked<R: Runtime>(
    a: &RudaTensor<R>,
    o: LargeSolverOptions,
) -> Result<QrFactorResult<R>, DeviceSolverError> {
    let (batch, m, n) = shape(a)?;
    if o.block_size == 0
        || o.block_size > 256
        || o.absolute_tolerance < 0.0
        || o.relative_tolerance < 0.0
        || !o.absolute_tolerance.is_finite()
        || !o.relative_tolerance.is_finite()
    {
        return Err(DeviceSolverError::InvalidInput("QR factor block/tolerance"));
    }
    let mut r = QrFactorResult {
        packed: alloc(a, [batch, m, n], a.dtype),
        tau: alloc(a, [batch, m.min(n)], a.dtype),
        info: alloc(a, [batch], DType::I32),
        submitted_kernels: 0,
    };
    if batch > 0 {
        match a.dtype {
            DType::F32 => qr_factor_submit::<f32, R>(a, o, &mut r, m, n),
            DType::F64 => qr_factor_submit::<f64, R>(a, o, &mut r, m, n),
            _ => unreachable!(),
        }
    }
    Ok(r)
}
fn qr_factor_submit<F: HostFloat, R: Runtime>(
    a: &RudaTensor<R>,
    o: LargeSolverOptions,
    r: &mut QrFactorResult<R>,
    m: usize,
    n: usize,
) {
    let batch = a.meta.shape()[0];
    let k = m.min(n);
    let scales = alloc(a, [batch], a.dtype);
    let d = RudaDim::new(a.client.properties(), batch);
    householder_init::launch::<F, R>(
        &a.client,
        calculate_ruda_count_elemwise(&a.client, batch, d),
        d,
        a.clone().into_tensor_arg(),
        r.packed.clone().into_array_arg(),
        r.tau.clone().into_array_arg(),
        scales.clone().into_array_arg(),
        r.info.clone().into_array_arg(),
        m as u32,
        n as u32,
    );
    r.submitted_kernels += 1;
    for first in (0..k).step_by(o.block_size) {
        let end = (first + o.block_size).min(k);
        householder_panel::launch::<F, R>(
            &a.client,
            calculate_ruda_count_elemwise(&a.client, batch, d),
            d,
            r.packed.clone().into_array_arg(),
            r.tau.clone().into_array_arg(),
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
            let d = RudaDim::new(a.client.properties(), count);
            householder_trailing::launch::<F, R>(
                &a.client,
                calculate_ruda_count_elemwise(&a.client, count, d),
                d,
                r.packed.clone().into_array_arg(),
                r.tau.clone().into_array_arg(),
                r.info.clone().into_array_arg(),
                m as u32,
                n as u32,
                first as u32,
                end as u32,
            );
            r.submitted_kernels += 1;
        }
    }
}
pub fn qr_form_q<R: Runtime>(
    packed: &RudaTensor<R>,
    tau: &RudaTensor<R>,
    columns: usize,
) -> Result<RudaTensor<R>, DeviceSolverError> {
    let (batch, m, n) = shape(packed)?;
    let k = tau.meta.shape().last().copied().unwrap_or(0);
    if columns > m
        || k > m.min(n)
        || tau.meta.num_dims() == 0
        || tau.dtype != packed.dtype
        || tau.meta.num_elements() != batch * k
        || !packed.client.same_execution_queue(&tau.client)
    {
        return Err(DeviceSolverError::InvalidInput("QR Q/tau shape or queue"));
    }
    super::elements(&[batch, m, columns])?;
    let tau = into_contiguous(tau.clone());
    let q = alloc(packed, [batch, m, columns], packed.dtype);
    let count = batch * columns;
    if count > 0 {
        let d = RudaDim::new(packed.client.properties(), count);
        match packed.dtype {
            DType::F32 => form_q::launch::<f32, R>(
                &packed.client,
                calculate_ruda_count_elemwise(&packed.client, count, d),
                d,
                packed.clone().into_tensor_arg(),
                tau.clone().into_array_arg(),
                q.clone().into_array_arg(),
                m as u32,
                k as u32,
                columns as u32,
            ),
            DType::F64 => form_q::launch::<f64, R>(
                &packed.client,
                calculate_ruda_count_elemwise(&packed.client, count, d),
                d,
                packed.clone().into_tensor_arg(),
                tau.clone().into_array_arg(),
                q.clone().into_array_arg(),
                m as u32,
                k as u32,
                columns as u32,
            ),
            _ => unreachable!(),
        }
    }
    Ok(q)
}
pub fn qr_extract_r<R: Runtime>(
    packed: &RudaTensor<R>,
    rows: usize,
) -> Result<RudaTensor<R>, DeviceSolverError> {
    let (batch, m, n) = shape(packed)?;
    if rows > m {
        return Err(DeviceSolverError::InvalidInput("QR R rows"));
    }
    let r = alloc(packed, [batch, rows, n], packed.dtype);
    let count = batch * rows * n;
    if count > 0 {
        let d = RudaDim::new(packed.client.properties(), count);
        match packed.dtype {
            DType::F32 => extract_r::launch::<f32, R>(
                &packed.client,
                calculate_ruda_count_elemwise(&packed.client, count, d),
                d,
                packed.clone().into_tensor_arg(),
                r.clone().into_array_arg(),
                rows as u32,
                n as u32,
            ),
            DType::F64 => extract_r::launch::<f64, R>(
                &packed.client,
                calculate_ruda_count_elemwise(&packed.client, count, d),
                d,
                packed.clone().into_tensor_arg(),
                r.clone().into_array_arg(),
                rows as u32,
                n as u32,
            ),
            _ => unreachable!(),
        }
    }
    Ok(r)
}
/// Apply Q or Q^T from stored reflectors to B. B is not modified.
pub fn qr_apply_q<R: Runtime>(
    packed: &RudaTensor<R>,
    tau: &RudaTensor<R>,
    b: &RudaTensor<R>,
    transpose: bool,
) -> Result<RudaTensor<R>, DeviceSolverError> {
    let (batch, m, n) = shape(packed)?;
    let (bb, mb, nrhs) = shape(b)?;
    same(packed, b)?;
    let k = tau.meta.shape().last().copied().unwrap_or(0);
    if bb != batch
        || mb != m
        || k > m.min(n)
        || tau.meta.num_elements() != batch * k
        || tau.dtype != packed.dtype
        || !packed.client.same_execution_queue(&tau.client)
    {
        return Err(DeviceSolverError::InvalidInput("apply Q shape"));
    }
    let tau = into_contiguous(tau.clone());
    let x = alloc(b, [batch, m, nrhs], b.dtype);
    let count = batch * nrhs;
    if count > 0 {
        let d = RudaDim::new(b.client.properties(), count);
        match b.dtype {
            DType::F32 => apply_q::launch::<f32, R>(
                &b.client,
                calculate_ruda_count_elemwise(&b.client, count, d),
                d,
                packed.clone().into_tensor_arg(),
                tau.clone().into_array_arg(),
                b.clone().into_tensor_arg(),
                x.clone().into_array_arg(),
                m as u32,
                k as u32,
                nrhs as u32,
                transpose,
            ),
            DType::F64 => apply_q::launch::<f64, R>(
                &b.client,
                calculate_ruda_count_elemwise(&b.client, count, d),
                d,
                packed.clone().into_tensor_arg(),
                tau.clone().into_array_arg(),
                b.clone().into_tensor_arg(),
                x.clone().into_array_arg(),
                m as u32,
                k as u32,
                nrhs as u32,
                transpose,
            ),
            _ => unreachable!(),
        }
    }
    Ok(x)
}

#[ruda(launch)]
fn factor_init<F: Float>(
    a: &Tensor<F>,
    lu: &mut Array<F>,
    p: &mut Array<i32>,
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
    let mut code = 0i32;
    for i in 0..m * n {
        let z = a[sys * a.stride(0) + (i / n) * a.stride(1) + (i % n) * a.stride(2)];
        lu[sys * m * n + i] = z;
        if crate::tensor::large::bad::<F>(z) {
            code = -1;
        }
    }
    for i in 0..m.min(n) {
        p[sys * m.min(n) + i] = i as i32;
    }
    info[sys] = code;
}
#[ruda(launch)]
fn factor_panel<F: Float + RudaElement>(
    lu: &mut Array<F>,
    p: &mut Array<i32>,
    info: &mut Array<i32>,
    rows: u32,
    cols: u32,
    first: u32,
    last: u32,
    #[comptime] pivot: bool,
    atol: F,
    rtol: F,
) {
    let sys = ABSOLUTE_POS;
    if sys >= info.len() {
        terminate!();
    }
    if info[sys] < 0 {
        terminate!();
    }
    let m = rows as usize;
    let n = cols as usize;
    let base = sys * m * n;
    let end = last as usize;
    let mut code = info[sys];
    for k in first as usize..end {
        if code >= 0 {
            let mut row = k;
            let mut peak = lu[base + k * n + k].abs();
            let mut scale = F::new(0.0);
            for i in k..m {
                let z = lu[base + i * n + k].abs();
                if comptime!(pivot) && z > peak {
                    peak = z;
                    row = i;
                }
                for j in k..end {
                    scale = F::max(scale, lu[base + i * n + j].abs());
                }
            }
            p[sys * m.min(n) + k] = row as i32;
            if row != k {
                for j in 0..n {
                    let z = lu[base + k * n + j];
                    lu[base + k * n + j] = lu[base + row * n + j];
                    lu[base + row * n + j] = z;
                }
            }
            let diag = lu[base + k * n + k];
            if crate::tensor::large::bad::<F>(diag) {
                code = -3;
            } else if peak <= atol + rtol * scale {
                if code == 0 {
                    code = (k + 1) as i32;
                }
            }
            if diag != F::new(0.0) && code >= 0 {
                for i in k + 1..m {
                    let z = lu[base + i * n + k] / diag;
                    lu[base + i * n + k] = z;
                    if crate::tensor::large::bad::<F>(z) {
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
fn factor_upper<F: Float>(
    lu: &mut Array<F>,
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
    if info[sys] < 0 {
        terminate!();
    }
    let col = end + pos % tail;
    let base = sys * m * n;
    for i in first as usize..end {
        let mut z = lu[base + i * n + col];
        for k in first as usize..i {
            z -= lu[base + i * n + k] * lu[base + k * n + col];
        }
        lu[base + i * n + col] = z;
    }
}
#[ruda(launch)]
fn factor_update<F: Float>(
    lu: &mut Array<F>,
    info: &Array<i32>,
    rows: u32,
    cols: u32,
    first: u32,
    last: u32,
) {
    let m = rows as usize;
    let n = cols as usize;
    let end = last as usize;
    let nc = n - end;
    let mr = m - end;
    let pos = ABSOLUTE_POS;
    let sys = pos / (mr * nc);
    if sys >= info.len() {
        terminate!();
    }
    if info[sys] < 0 {
        terminate!();
    }
    let i = end + (pos / nc) % mr;
    let j = end + pos % nc;
    let base = sys * m * n;
    let mut z = lu[base + i * n + j];
    for k in first as usize..end {
        z -= lu[base + i * n + k] * lu[base + k * n + j];
    }
    lu[base + i * n + j] = z;
}
#[ruda(launch)]
fn householder_init<F: Float>(
    a: &Tensor<F>,
    w: &mut Array<F>,
    tau: &mut Array<F>,
    scale: &mut Array<F>,
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
    let mut code = 0i32;
    let mut peak = F::new(0.0);
    for i in 0..m * n {
        let z = a[sys * a.stride(0) + (i / n) * a.stride(1) + (i % n) * a.stride(2)];
        w[sys * m * n + i] = z;
        peak = F::max(peak, z.abs());
        if crate::tensor::large::bad::<F>(z) {
            code = -1;
        }
    }
    for i in 0..m.min(n) {
        tau[sys * m.min(n) + i] = F::new(0.0);
    }
    scale[sys] = peak;
    info[sys] = code;
}
#[ruda(launch)]
fn householder_panel<F: Float + RudaElement>(
    w: &mut Array<F>,
    tau: &mut Array<F>,
    scale: &Array<F>,
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
    if info[sys] < 0 {
        terminate!();
    }
    let m = rows as usize;
    let n = cols as usize;
    let base = sys * m * n;
    let end = last as usize;
    let mut code = info[sys];
    for k in first as usize..end {
        if code >= 0 {
            let mut peak = F::new(0.0);
            for i in k..m {
                peak = F::max(peak, w[base + i * n + k].abs());
            }
            let mut norm = F::new(0.0);
            if peak > F::new(0.0) {
                for i in k..m {
                    let z = w[base + i * n + k] / peak;
                    norm += z * z;
                }
                norm = norm.sqrt() * peak;
            }
            if crate::tensor::large::bad::<F>(norm) {
                code = -3;
            } else {
                if norm <= atol + rtol * scale[sys] && code == 0 {
                    code = (k + 1) as i32;
                }
                if norm > F::new(0.0) {
                    let alpha = w[base + k * n + k];
                    let mut beta = norm;
                    if alpha >= F::new(0.0) {
                        beta = -norm;
                    }
                    let ratio = alpha / beta;
                    let t = F::new(1.0) - ratio;
                    tau[sys * m.min(n) + k] = t;
                    for i in k + 1..m {
                        let value = w[base + i * n + k];
                        w[base + i * n + k] = (value / beta) / (ratio - F::new(1.0));
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
    }
    info[sys] = code;
}
#[ruda(launch)]
fn householder_trailing<F: Float>(
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
    let pos = ABSOLUTE_POS;
    let sys = pos / (n - end);
    if sys >= info.len() {
        terminate!();
    }
    if info[sys] < 0 {
        terminate!();
    }
    let j = end + pos % (n - end);
    let base = sys * m * n;
    for k in first as usize..end {
        let mut dot = w[base + k * n + j];
        for i in k + 1..m {
            dot += w[base + i * n + k] * w[base + i * n + j];
        }
        dot *= tau[sys * m.min(n) + k];
        w[base + k * n + j] -= dot;
        for i in k + 1..m {
            let update = w[base + i * n + k] * dot;
            w[base + i * n + j] -= update;
        }
    }
}
#[ruda(launch)]
fn form_q<F: Float>(
    w: &Tensor<F>,
    tau: &Array<F>,
    q: &mut Array<F>,
    rows: u32,
    reflectors: u32,
    qcols: u32,
) {
    let m = rows as usize;
    let count = reflectors as usize;
    let nq = qcols as usize;
    let pos = ABSOLUTE_POS;
    let sys = pos / nq;
    let j = pos % nq;
    if sys >= w.shape(0) {
        terminate!();
    }
    let base = sys * m * nq;
    for i in 0..m {
        let mut z = F::new(0.0);
        if i == j {
            z = F::new(1.0);
        }
        q[base + i * nq + j] = z;
    }
    let mut end = count;
    while end > 0 {
        let k = end - 1;
        let factor = tau[sys * count + k];
        if factor != F::new(0.0) {
            let mut dot = q[base + k * nq + j];
            for i in k + 1..m {
                let wi = sys * w.stride(0) + i * w.stride(1) + k * w.stride(2);
                dot += w[wi] * q[base + i * nq + j];
            }
            dot *= factor;
            q[base + k * nq + j] -= dot;
            for i in k + 1..m {
                let wi = sys * w.stride(0) + i * w.stride(1) + k * w.stride(2);
                q[base + i * nq + j] -= w[wi] * dot;
            }
        }
        end -= 1;
    }
}
#[ruda(launch)]
fn extract_r<F: Float>(w: &Tensor<F>, r: &mut Array<F>, rows: u32, cols: u32) {
    let nr = rows as usize;
    let n = cols as usize;
    let pos = ABSOLUTE_POS;
    if pos >= r.len() {
        terminate!();
    }
    let sys = pos / (nr * n);
    let i = (pos / n) % nr;
    let j = pos % n;
    let mut z = F::new(0.0);
    if i <= j {
        z = w[sys * w.stride(0) + i * w.stride(1) + j * w.stride(2)];
    }
    r[pos] = z;
}
#[ruda(launch)]
fn apply_q<F: Float>(
    w: &Tensor<F>,
    tau: &Array<F>,
    b: &Tensor<F>,
    x: &mut Array<F>,
    rows: u32,
    cols: u32,
    rhs: u32,
    #[comptime] transpose: bool,
) {
    let m = rows as usize;
    let n = cols as usize;
    let nr = rhs as usize;
    let pos = ABSOLUTE_POS;
    let sys = pos / nr;
    if sys >= b.shape(0) {
        terminate!();
    }
    let j = pos % nr;
    let base = sys * m * nr;
    for i in 0..m {
        x[base + i * nr + j] = b[sys * b.stride(0) + i * b.stride(1) + j * b.stride(2)];
    }
    for step in 0..m.min(n) {
        let k = if comptime!(transpose) {
            step
        } else {
            m.min(n) - 1 - step
        };
        let mut dot = x[base + k * nr + j];
        for i in k + 1..m {
            dot += w[sys * w.stride(0) + i * w.stride(1) + k * w.stride(2)] * x[base + i * nr + j];
        }
        dot *= tau[sys * m.min(n) + k];
        x[base + k * nr + j] -= dot;
        for i in k + 1..m {
            let update = w[sys * w.stride(0) + i * w.stride(1) + k * w.stride(2)] * dot;
            x[base + i * nr + j] -= update;
        }
    }
}
