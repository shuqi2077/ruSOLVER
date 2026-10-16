use super::{
    BatchedEigenResult, DeviceSolverError,
    large::{HostFloat, alloc, validate},
};
use ruda_core::tensor::DType;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{calculate_ruda_count_elemwise, prelude::*};
use ruda_kernel::tensor::{RudaTensor, permutation::swap_dims, readback::into_data};

#[derive(Clone, Copy, Debug)]
pub struct DeviceJacobiOptions {
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub symmetry_tolerance: f64,
    pub max_sweeps: u32,
}
impl Default for DeviceJacobiOptions {
    fn default() -> Self {
        Self {
            absolute_tolerance: 0.0,
            relative_tolerance: 1e-12,
            symmetry_tolerance: 1e-12,
            max_sweeps: 64,
        }
    }
}
pub(super) fn options(o: DeviceJacobiOptions) -> Result<(), DeviceSolverError> {
    if [
        o.absolute_tolerance,
        o.relative_tolerance,
        o.symmetry_tolerance,
    ]
    .iter()
    .any(|x| !x.is_finite() || *x < 0.0)
        || o.absolute_tolerance + o.relative_tolerance == 0.0
        || o.max_sweeps == 0
        || o.max_sweeps > 1024
    {
        return Err(DeviceSolverError::InvalidInput(
            "finite nonnegative tolerances, positive convergence tolerance, sweeps=1..1024 required",
        ));
    }
    Ok(())
}
/// Parallel round-robin symmetric Jacobi. F32/F64 [batch,n,n]; eigenvalues
/// ascending, eigenvectors in columns. -2 nonsymmetric, -4 nonconvergence,
/// -1 nonfinite input, -3 arithmetic failure. Failed outputs are zero.
pub fn symmetric_eigen_device<R: Runtime>(
    a: &RudaTensor<R>,
    o: DeviceJacobiOptions,
) -> Result<BatchedEigenResult<R>, DeviceSolverError> {
    let (batch, n, n2) = validate(a)?;
    options(o)?;
    if n != n2 {
        return Err(DeviceSolverError::InvalidInput(
            "square symmetric matrix required",
        ));
    }
    let mut r = BatchedEigenResult {
        values: alloc(a, [batch, n], a.dtype),
        vectors: alloc(a, [batch, n, n], a.dtype),
        sweeps: alloc(a, [batch], DType::I32),
        info: alloc(a, [batch], DType::I32),
        submitted_kernels: 0,
    };
    if batch > 0 {
        match a.dtype {
            DType::F32 => eigen_submit::<f32, R>(a, o, &mut r, n),
            DType::F64 => eigen_submit::<f64, R>(a, o, &mut r, n),
            _ => unreachable!(),
        }
    }
    Ok(r)
}
fn eigen_submit<F: HostFloat, R: Runtime>(
    a: &RudaTensor<R>,
    o: DeviceJacobiOptions,
    r: &mut BatchedEigenResult<R>,
    n: usize,
) {
    let batch = a.meta.shape()[0];
    let slots = n + n % 2;
    let pairs = slots / 2;
    let w = alloc(a, [batch, n, n], a.dtype);
    let cs = alloc(a, [batch, pairs, 2], a.dtype);
    let d = RudaDim::new(a.client.properties(), batch);
    eigen_init::launch::<F, R>(
        &a.client,
        calculate_ruda_count_elemwise(&a.client, batch, d),
        d,
        a.clone().into_tensor_arg(),
        w.clone().into_array_arg(),
        r.vectors.clone().into_array_arg(),
        r.info.clone().into_array_arg(),
        r.sweeps.clone().into_array_arg(),
        n as u32,
        F::scalar(o.symmetry_tolerance),
    );
    r.submitted_kernels += 1;
    // A convergence check before the first sweep accepts already diagonal input.
    eigen_check::launch::<F, R>(
        &a.client,
        calculate_ruda_count_elemwise(&a.client, batch, d),
        d,
        w.clone().into_array_arg(),
        r.info.clone().into_array_arg(),
        r.sweeps.clone().into_array_arg(),
        n as u32,
        0,
        F::scalar(o.absolute_tolerance),
        F::scalar(o.relative_tolerance),
    );
    r.submitted_kernels += 1;
    for sweep in 0..o.max_sweeps {
        for round in 0..slots - 1 {
            let count = batch * pairs;
            let pd = RudaDim::new(a.client.properties(), count);
            eigen_angles::launch::<F, R>(
                &a.client,
                calculate_ruda_count_elemwise(&a.client, count, pd),
                pd,
                w.clone().into_array_arg(),
                cs.clone().into_array_arg(),
                r.info.clone().into_array_arg(),
                n as u32,
                round as u32,
            );
            let count = batch * pairs * n;
            let rd = RudaDim::new(a.client.properties(), count);
            eigen_rows::launch::<F, R>(
                &a.client,
                calculate_ruda_count_elemwise(&a.client, count, rd),
                rd,
                w.clone().into_array_arg(),
                cs.clone().into_array_arg(),
                r.info.clone().into_array_arg(),
                n as u32,
                round as u32,
            );
            eigen_columns::launch::<F, R>(
                &a.client,
                calculate_ruda_count_elemwise(&a.client, count, rd),
                rd,
                w.clone().into_array_arg(),
                r.vectors.clone().into_array_arg(),
                cs.clone().into_array_arg(),
                r.info.clone().into_array_arg(),
                n as u32,
                round as u32,
            );
            r.submitted_kernels += 3;
        }
        eigen_check::launch::<F, R>(
            &a.client,
            calculate_ruda_count_elemwise(&a.client, batch, d),
            d,
            w.clone().into_array_arg(),
            r.info.clone().into_array_arg(),
            r.sweeps.clone().into_array_arg(),
            n as u32,
            sweep + 1,
            F::scalar(o.absolute_tolerance),
            F::scalar(o.relative_tolerance),
        );
        r.submitted_kernels += 1;
    }
    eigen_finish::launch::<F, R>(
        &a.client,
        calculate_ruda_count_elemwise(&a.client, batch, d),
        d,
        w.into_array_arg(),
        r.values.clone().into_array_arg(),
        r.vectors.clone().into_array_arg(),
        r.info.clone().into_array_arg(),
        n as u32,
    );
    r.submitted_kernels += 1;
}

pub struct DeviceSvdResult<R: Runtime> {
    /// Thin factorization A=U diag(singular_values) V^T; values descending.
    pub u: RudaTensor<R>,
    pub singular_values: RudaTensor<R>,
    pub v: RudaTensor<R>,
    pub info: RudaTensor<R>,
    pub sweeps: RudaTensor<R>,
    pub submitted_kernels: usize,
}
impl<R: Runtime> DeviceSvdResult<R> {
    pub fn check_status_sync(&self) -> Result<(), DeviceSolverError> {
        let data = ruda_core::future::block_on(into_data(self.info.clone()))
            .map_err(|e| DeviceSolverError::Readback(format!("{e:?}")))?;
        for (batch, &info) in data
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
}
/// One-sided cyclic Jacobi SVD, without forming A^T A. F32/F64 rectangular
/// input, thin U/V, rank-deficient U completed to an orthonormal basis on GPU.
/// -4 means nonconvergence, -1 nonfinite input, -3 arithmetic failure.
pub fn svd_device<R: Runtime>(
    a: &RudaTensor<R>,
    o: DeviceJacobiOptions,
) -> Result<DeviceSvdResult<R>, DeviceSolverError> {
    let (batch, m, n) = validate(a)?;
    options(o)?;
    let wide = m < n;
    let input = if wide {
        swap_dims(a.clone(), 1, 2)
    } else {
        a.clone()
    };
    let rows = m.max(n);
    let cols = m.min(n);
    super::elements(&[batch, cols, cols])?;
    let mut r = DeviceSvdResult {
        u: alloc(a, [batch, rows, cols], a.dtype),
        singular_values: alloc(a, [batch, cols], a.dtype),
        v: alloc(a, [batch, cols, cols], a.dtype),
        info: alloc(a, [batch], DType::I32),
        sweeps: alloc(a, [batch], DType::I32),
        submitted_kernels: 0,
    };
    if batch > 0 {
        match a.dtype {
            DType::F32 => svd_submit::<f32, R>(&input, o, &mut r, rows, cols),
            DType::F64 => svd_submit::<f64, R>(&input, o, &mut r, rows, cols),
            _ => unreachable!(),
        }
    }
    if wide {
        std::mem::swap(&mut r.u, &mut r.v);
    }
    Ok(r)
}
fn svd_submit<F: HostFloat, R: Runtime>(
    a: &RudaTensor<R>,
    o: DeviceJacobiOptions,
    r: &mut DeviceSvdResult<R>,
    m: usize,
    n: usize,
) {
    let batch = a.meta.shape()[0];
    let slots = n + n % 2;
    let pairs = slots / 2;
    let correlations = alloc(a, [batch, pairs], a.dtype);
    let d = RudaDim::new(a.client.properties(), batch);
    svd_init::launch::<F, R>(
        &a.client,
        calculate_ruda_count_elemwise(&a.client, batch, d),
        d,
        a.clone().into_tensor_arg(),
        r.u.clone().into_array_arg(),
        r.v.clone().into_array_arg(),
        r.info.clone().into_array_arg(),
        r.sweeps.clone().into_array_arg(),
        m as u32,
        n as u32,
    );
    r.submitted_kernels += 1;
    for sweep in 0..o.max_sweeps {
        for round in 0..slots - 1 {
            let count = batch * pairs;
            let pd = RudaDim::new(a.client.properties(), count);
            svd_pairs::launch::<F, R>(
                &a.client,
                calculate_ruda_count_elemwise(&a.client, count, pd),
                pd,
                r.u.clone().into_array_arg(),
                r.v.clone().into_array_arg(),
                correlations.clone().into_array_arg(),
                r.info.clone().into_array_arg(),
                m as u32,
                n as u32,
                round as u32,
                F::scalar(o.absolute_tolerance),
                F::scalar(o.relative_tolerance),
            );
            r.submitted_kernels += 1;
        }
        svd_check::launch::<F, R>(
            &a.client,
            calculate_ruda_count_elemwise(&a.client, batch, d),
            d,
            r.u.clone().into_array_arg(),
            r.info.clone().into_array_arg(),
            r.sweeps.clone().into_array_arg(),
            m as u32,
            n as u32,
            sweep + 1,
            F::scalar(o.absolute_tolerance),
            F::scalar(o.relative_tolerance),
        );
        r.submitted_kernels += 1;
    }
    svd_finish::launch::<F, R>(
        &a.client,
        calculate_ruda_count_elemwise(&a.client, batch, d),
        d,
        r.u.clone().into_array_arg(),
        r.v.clone().into_array_arg(),
        r.singular_values.clone().into_array_arg(),
        r.info.clone().into_array_arg(),
        m as u32,
        n as u32,
    );
    r.submitted_kernels += 1;
}

#[ruda]
fn pair(order: usize, round: usize, t: usize) -> (usize, usize) {
    let slots = order + order % 2;
    let p = (round + t) % (slots - 1);
    let mut q = slots - 1;
    if t != 0 {
        q = (round + slots - 1 - t) % (slots - 1);
    }
    (p, q)
}
#[ruda]
fn rotation<F: Float>(a: F, b: F, g: F) -> (F, F) {
    let mut c = F::new(1.0);
    let mut s = F::new(0.0);
    if g != F::new(0.0) {
        let delta = b - a;
        let scale = F::max(delta.abs(), F::new(2.0) * g.abs());
        let d = delta / scale;
        let h = F::new(2.0) * g / scale;
        let root = (d * d + h * h).sqrt();
        let mut t = F::new(1.0);
        if delta > F::new(0.0) {
            t = h / (d + root);
        } else if delta < F::new(0.0) {
            t = h / (d - root);
        }
        c = F::new(1.0) / (F::new(1.0) + t * t).sqrt();
        s = t * c;
    }
    (c, s)
}
#[ruda(launch)]
fn eigen_init<F: Float + RudaElement>(
    a: &Tensor<F>,
    w: &mut Array<F>,
    v: &mut Array<F>,
    info: &mut Array<i32>,
    sweeps: &mut Array<i32>,
    order: u32,
    symtol: F,
) {
    let sys = ABSOLUTE_POS;
    if sys >= info.len() {
        terminate!();
    }
    let n = order as usize;
    let base = sys * n * n;
    let mut code = 1i32;
    let mut scale = F::new(0.0);
    for i in 0..n * n {
        let z = a[sys * a.stride(0) + (i / n) * a.stride(1) + (i % n) * a.stride(2)];
        w[base + i] = z;
        v[base + i] = F::new(0.0);
        scale = F::max(scale, z.abs());
        if crate::tensor::large::bad::<F>(z) {
            code = -1;
        }
    }
    for i in 0..n {
        v[base + i * n + i] = F::new(1.0);
        for j in 0..i {
            if code == 1
                && scale > F::new(0.0)
                && (w[base + i * n + j] / scale - w[base + j * n + i] / scale).abs() > symtol
            {
                code = -2;
            }
        }
    }
    if code == 1 {
        for i in 0..n {
            for j in i + 1..n {
                w[base + i * n + j] = w[base + j * n + i];
            }
        }
    }
    info[sys] = code;
    sweeps[sys] = 0;
}
#[ruda(launch)]
fn eigen_angles<F: Float>(
    w: &Array<F>,
    cs: &mut Array<F>,
    info: &Array<i32>,
    order: u32,
    round: u32,
) {
    let n = order as usize;
    let pairs = (n + n % 2) / 2;
    let pos = ABSOLUTE_POS;
    let sys = pos / pairs;
    if sys >= info.len() {
        terminate!();
    }
    let (p, q) = pair(n, round as usize, pos % pairs);
    let mut c = F::new(1.0);
    let mut s = F::new(0.0);
    if info[sys] == 1 && p < n && q < n {
        let base = sys * n * n;
        let scale = F::max(
            w[base + p * n + p].abs(),
            F::max(w[base + q * n + q].abs(), w[base + p * n + q].abs()),
        );
        if scale > F::new(0.0) {
            let r = rotation::<F>(
                w[base + p * n + p] / scale,
                w[base + q * n + q] / scale,
                w[base + p * n + q] / scale,
            );
            c = r.0;
            s = r.1;
        }
    }
    cs[2 * pos] = c;
    cs[2 * pos + 1] = s;
}
#[ruda(launch)]
fn eigen_rows<F: Float>(
    w: &mut Array<F>,
    cs: &Array<F>,
    info: &Array<i32>,
    order: u32,
    round: u32,
) {
    let n = order as usize;
    let pairs = (n + n % 2) / 2;
    let pos = ABSOLUTE_POS;
    let sys = pos / (pairs * n);
    if sys >= info.len() {
        terminate!();
    }
    if info[sys] != 1 {
        terminate!();
    }
    let t = (pos / n) % pairs;
    let j = pos % n;
    let (p, q) = pair(n, round as usize, t);
    if p >= n || q >= n {
        terminate!();
    }
    let c = cs[(sys * pairs + t) * 2];
    let s = cs[(sys * pairs + t) * 2 + 1];
    let base = sys * n * n;
    let x = w[base + p * n + j];
    let y = w[base + q * n + j];
    w[base + p * n + j] = c * x - s * y;
    w[base + q * n + j] = s * x + c * y;
}
#[ruda(launch)]
fn eigen_columns<F: Float>(
    w: &mut Array<F>,
    v: &mut Array<F>,
    cs: &Array<F>,
    info: &Array<i32>,
    order: u32,
    round: u32,
) {
    let n = order as usize;
    let pairs = (n + n % 2) / 2;
    let pos = ABSOLUTE_POS;
    let sys = pos / (pairs * n);
    if sys >= info.len() {
        terminate!();
    }
    if info[sys] != 1 {
        terminate!();
    }
    let t = (pos / n) % pairs;
    let i = pos % n;
    let (p, q) = pair(n, round as usize, t);
    if p >= n || q >= n {
        terminate!();
    }
    let c = cs[(sys * pairs + t) * 2];
    let s = cs[(sys * pairs + t) * 2 + 1];
    let base = sys * n * n;
    let x = w[base + i * n + p];
    let y = w[base + i * n + q];
    w[base + i * n + p] = c * x - s * y;
    w[base + i * n + q] = s * x + c * y;
    let x = v[base + i * n + p];
    let y = v[base + i * n + q];
    v[base + i * n + p] = c * x - s * y;
    v[base + i * n + q] = s * x + c * y;
}
#[ruda(launch)]
fn eigen_check<F: Float + RudaElement>(
    w: &Array<F>,
    info: &mut Array<i32>,
    sweeps: &mut Array<i32>,
    order: u32,
    sweep: u32,
    atol: F,
    rtol: F,
) {
    let sys = ABSOLUTE_POS;
    if sys >= info.len() {
        terminate!();
    }
    if info[sys] != 1 {
        terminate!();
    }
    let n = order as usize;
    let base = sys * n * n;
    let mut off = F::new(0.0);
    let mut scale = F::new(0.0);
    let mut code = 1i32;
    for i in 0..n {
        for j in 0..n {
            let z = w[base + i * n + j];
            if crate::tensor::large::bad::<F>(z) {
                code = -3;
            }
            scale = F::max(scale, z.abs());
            if i != j {
                off = F::max(off, z.abs());
            }
        }
    }
    if code == 1 && off <= atol + rtol * scale {
        code = 0;
    }
    info[sys] = code;
    sweeps[sys] = sweep as i32;
}
#[ruda(launch)]
fn eigen_finish<F: Float>(
    w: &Array<F>,
    values: &mut Array<F>,
    v: &mut Array<F>,
    info: &mut Array<i32>,
    order: u32,
) {
    let sys = ABSOLUTE_POS;
    if sys >= info.len() {
        terminate!();
    }
    let n = order as usize;
    let base = sys * n * n;
    let mut code = info[sys];
    if code == 1 {
        code = -4;
    }
    if code == 0 {
        for i in 0..n {
            values[sys * n + i] = w[base + i * n + i];
        }
        for i in 0..n {
            let mut p = i;
            for j in i + 1..n {
                if values[sys * n + j] < values[sys * n + p] {
                    p = j;
                }
            }
            if p != i {
                let z = values[sys * n + i];
                values[sys * n + i] = values[sys * n + p];
                values[sys * n + p] = z;
                for k in 0..n {
                    let z = v[base + k * n + i];
                    v[base + k * n + i] = v[base + k * n + p];
                    v[base + k * n + p] = z;
                }
            }
        }
    }
    if code != 0 {
        for i in 0..n {
            values[sys * n + i] = F::new(0.0);
        }
        for i in 0..n * n {
            v[base + i] = F::new(0.0);
        }
    }
    info[sys] = code;
}
#[ruda(launch)]
fn svd_init<F: Float>(
    a: &Tensor<F>,
    u: &mut Array<F>,
    v: &mut Array<F>,
    info: &mut Array<i32>,
    sweeps: &mut Array<i32>,
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
    let vb = sys * n * n;
    let mut code = 1i32;
    for i in 0..m * n {
        let z = a[sys * a.stride(0) + (i / n) * a.stride(1) + (i % n) * a.stride(2)];
        u[base + i] = z;
        if crate::tensor::large::bad::<F>(z) {
            code = -1;
        }
    }
    for i in 0..n * n {
        v[vb + i] = F::new(0.0);
    }
    for i in 0..n {
        v[vb + i * n + i] = F::new(1.0);
    }
    info[sys] = code;
    sweeps[sys] = 0;
}
#[ruda(launch)]
fn svd_pairs<F: Float + RudaElement>(
    u: &mut Array<F>,
    v: &mut Array<F>,
    correlation: &mut Array<F>,
    info: &Array<i32>,
    rows: u32,
    cols: u32,
    round: u32,
    atol: F,
    rtol: F,
) {
    let m = rows as usize;
    let n = cols as usize;
    let pairs = (n + n % 2) / 2;
    let pos = ABSOLUTE_POS;
    let sys = pos / pairs;
    if sys >= info.len() {
        terminate!();
    }
    correlation[pos] = F::new(0.0);
    if info[sys] != 1 {
        terminate!();
    }
    let (p, q) = pair(n, round as usize, pos % pairs);
    if p >= n || q >= n {
        terminate!();
    }
    let base = sys * m * n;
    let vb = sys * n * n;
    let mut scale = F::new(0.0);
    for i in 0..m {
        scale = F::max(
            scale,
            F::max(u[base + i * n + p].abs(), u[base + i * n + q].abs()),
        );
    }
    if scale > F::new(0.0) {
        let mut a = F::new(0.0);
        let mut b = F::new(0.0);
        let mut g = F::new(0.0);
        for i in 0..m {
            let x = u[base + i * n + p] / scale;
            let y = u[base + i * n + q] / scale;
            a += x * x;
            b += y * y;
            g += x * y;
        }
        let bound = atol / scale / scale + rtol * a.sqrt() * b.sqrt();
        if g.abs() > bound {
            correlation[pos] = F::new(1.0);
            let (c, s) = rotation::<F>(a, b, g);
            for i in 0..m {
                let x = u[base + i * n + p];
                let y = u[base + i * n + q];
                u[base + i * n + p] = c * x - s * y;
                u[base + i * n + q] = s * x + c * y;
            }
            for i in 0..n {
                let x = v[vb + i * n + p];
                let y = v[vb + i * n + q];
                v[vb + i * n + p] = c * x - s * y;
                v[vb + i * n + q] = s * x + c * y;
            }
        }
    }
}
#[ruda(launch)]
fn svd_check<F: Float + RudaElement>(
    u: &Array<F>,
    info: &mut Array<i32>,
    sweeps: &mut Array<i32>,
    rows: u32,
    cols: u32,
    sweep: u32,
    atol: F,
    rtol: F,
) {
    let sys = ABSOLUTE_POS;
    if sys >= info.len() {
        terminate!();
    }
    if info[sys] != 1 {
        terminate!();
    }
    let m = rows as usize;
    let n = cols as usize;
    let base = sys * m * n;
    let mut done = true;
    let mut code = 1i32;
    for i in 0..m * n {
        if crate::tensor::large::bad::<F>(u[base + i]) {
            code = -3;
        }
    }
    for p in 0..n {
        for q in p + 1..n {
            let mut scale = F::new(0.0);
            for i in 0..m {
                scale = F::max(
                    scale,
                    F::max(u[base + i * n + p].abs(), u[base + i * n + q].abs()),
                );
            }
            if scale > F::new(0.0) {
                let mut a = F::new(0.0);
                let mut b = F::new(0.0);
                let mut g = F::new(0.0);
                for i in 0..m {
                    let x = u[base + i * n + p] / scale;
                    let y = u[base + i * n + q] / scale;
                    a += x * x;
                    b += y * y;
                    g += x * y;
                }
                if g.abs() > atol / scale / scale + rtol * a.sqrt() * b.sqrt() {
                    done = false;
                }
            }
        }
    }
    if code == 1 && done {
        code = 0;
    }
    info[sys] = code;
    sweeps[sys] = sweep as i32;
}
#[ruda(launch)]
fn svd_finish<F: Float>(
    u: &mut Array<F>,
    v: &mut Array<F>,
    values: &mut Array<F>,
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
    let vb = sys * n * n;
    let mut code = info[sys];
    if code == 1 {
        code = -4;
    }
    if code == 0 {
        for j in 0..n {
            let mut scale = F::new(0.0);
            for i in 0..m {
                scale = F::max(scale, u[base + i * n + j].abs());
            }
            let mut norm = F::new(0.0);
            if scale > F::new(0.0) {
                for i in 0..m {
                    let x = u[base + i * n + j] / scale;
                    norm += x * x;
                }
                norm = norm.sqrt() * scale;
            }
            values[sys * n + j] = norm;
            if crate::tensor::large::bad::<F>(norm) {
                code = -3;
            }
        }
        for j in 0..n {
            let mut p = j;
            for k in j + 1..n {
                if values[sys * n + k] > values[sys * n + p] {
                    p = k;
                }
            }
            if p != j {
                let z = values[sys * n + j];
                values[sys * n + j] = values[sys * n + p];
                values[sys * n + p] = z;
                for i in 0..m {
                    let z = u[base + i * n + j];
                    u[base + i * n + j] = u[base + i * n + p];
                    u[base + i * n + p] = z;
                }
                for i in 0..n {
                    let z = v[vb + i * n + j];
                    v[vb + i * n + j] = v[vb + i * n + p];
                    v[vb + i * n + p] = z;
                }
            }
        }
        for j in 0..n {
            let norm = values[sys * n + j];
            if norm > F::new(0.0) {
                for i in 0..m {
                    u[base + i * n + j] /= norm;
                }
            } else {
                let mut found = false;
                for candidate in 0..m {
                    if !found {
                        for i in 0..m {
                            u[base + i * n + j] = F::new(0.0);
                        }
                        u[base + candidate * n + j] = F::new(1.0);
                        // Two projection passes stabilize completion against previously formed U.
                        for _pass in 0..2 {
                            for k in 0..j {
                                let mut dot = F::new(0.0);
                                for i in 0..m {
                                    dot += u[base + i * n + k] * u[base + i * n + j];
                                }
                                for i in 0..m {
                                    let update = dot * u[base + i * n + k];
                                    u[base + i * n + j] -= update;
                                }
                            }
                        }
                        let mut norm = F::new(0.0);
                        for i in 0..m {
                            let x = u[base + i * n + j];
                            norm += x * x;
                        }
                        if norm > F::new(0.5) / F::cast_from(m) {
                            norm = norm.sqrt();
                            for i in 0..m {
                                u[base + i * n + j] /= norm;
                            }
                            found = true;
                        }
                    }
                }
                if !found {
                    code = -3;
                }
            }
        }
        for i in 0..m * n {
            if crate::tensor::large::bad::<F>(u[base + i]) {
                code = -3;
            }
        }
        for i in 0..n * n {
            if crate::tensor::large::bad::<F>(v[vb + i]) {
                code = -3;
            }
        }
    }
    if code != 0 {
        for i in 0..m * n {
            u[base + i] = F::new(0.0);
        }
        for i in 0..n * n {
            v[vb + i] = F::new(0.0);
        }
        for i in 0..n {
            values[sys * n + i] = F::new(0.0);
        }
    }
    info[sys] = code;
}
