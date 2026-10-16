use super::{
    DeviceJacobiOptions, DeviceSolverError, complex::{Matrix, empty, grid, initialize, shape},
    complex_qr, complex_spectral_kernel as kernels, jacobi::options, large::{HostFloat, alloc},
};
use ruda_core::tensor::DType;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::tensor::RudaTensor;

pub struct Eigen<R: Runtime> {
    pub values: RudaTensor<R>,
    pub vectors: Matrix<R>,
    pub info: RudaTensor<R>,
    pub sweeps: RudaTensor<R>,
    pub submitted_kernels: usize,
}

pub struct Svd<R: Runtime> {
    pub u: Matrix<R>,
    pub singular_values: RudaTensor<R>,
    pub vh: Matrix<R>,
    pub info: RudaTensor<R>,
    pub sweeps: RudaTensor<R>,
    pub submitted_kernels: usize,
}

fn prepare<R: Runtime>(a: &Matrix<R>, work: &Matrix<R>, info: &RudaTensor<R>, sweeps: &RudaTensor<R>, hermitian: bool, upper: bool, adjoint: bool) -> usize {
    let sizes = work.real.meta.shape(); let (batch, m, n) = (sizes[0], sizes[1], sizes[2]);
    if batch == 0 { return 0; }
    macro_rules! run { ($f:ty) => {{
        kernels::prepare::launch::<$f, R>(&a.real.client, grid(batch), RudaDim::new_1d(256),
            a.real.clone().into_tensor_arg(), a.imag.clone().into_tensor_arg(), work.real.clone().into_array_arg(), work.imag.clone().into_array_arg(),
            info.clone().into_array_arg(), sweeps.clone().into_array_arg(), m as u32, n as u32, hermitian, upper, adjoint);
    }}; }
    match a.real.dtype { DType::F32 => run!(f32), DType::F64 => run!(f64), _ => unreachable!() }
    1
}

fn identity<R: Runtime>(a: &Matrix<R>, value: &Matrix<R>) -> usize {
    let dummy = alloc(&a.real, [0], DType::I32); initialize(a, value, &dummy, 2, false)
}

fn finish<R: Runtime>(info: &RudaTensor<R>) -> usize {
    let count = info.meta.num_elements(); if count == 0 { return 0; }
    kernels::seal::launch::<R>(&info.client, grid(count.div_ceil(256)), RudaDim::new_1d(256), info.clone().into_array_arg()); 1
}

fn sorted<R: Runtime>(a: &Matrix<R>, values: &RudaTensor<R>, n: usize, descending: bool) -> (RudaTensor<R>, RudaTensor<R>, usize) {
    let out = alloc(&a.real, values.meta.shape().clone(), a.real.dtype);
    let permutation = alloc(&a.real, values.meta.shape().clone(), DType::U32); let count = values.meta.num_elements();
    if count == 0 { return (out, permutation, 0); }
    macro_rules! run { ($f:ty) => {{
        kernels::sort_values::launch::<$f, R>(&a.real.client, grid(count.div_ceil(256)), RudaDim::new_1d(256),
            values.clone().into_array_arg(), out.clone().into_array_arg(), permutation.clone().into_array_arg(), n as u32, descending);
    }}; }
    match a.real.dtype { DType::F32 => run!(f32), DType::F64 => run!(f64), _ => unreachable!() }
    (out, permutation, 1)
}

fn gather<R: Runtime>(a: &Matrix<R>, permutation: &RudaTensor<R>, info: &RudaTensor<R>) -> Result<complex_qr::Transform<R>, DeviceSolverError> {
    let (batch, m, n) = shape(a)?; let out = empty(a, &[batch, m, n])?; let count = out.real.meta.num_elements();
    if count == 0 { return Ok(complex_qr::Transform { value: out, submitted_kernels: 0 }); }
    macro_rules! run { ($f:ty) => {{
        kernels::gather::launch::<$f, R>(&a.real.client, grid(count.div_ceil(256)), RudaDim::new_1d(256),
            a.real.clone().into_array_arg(), a.imag.clone().into_array_arg(), out.real.clone().into_array_arg(), out.imag.clone().into_array_arg(),
            permutation.clone().into_array_arg(), info.clone().into_array_arg(), m as u32, n as u32);
    }}; }
    match a.real.dtype { DType::F32 => run!(f32), DType::F64 => run!(f64), _ => unreachable!() }
    Ok(complex_qr::Transform { value: out, submitted_kernels: 1 })
}

pub fn eigh<R: Runtime>(a: &Matrix<R>, upper: bool, vectors: bool, o: DeviceJacobiOptions) -> Result<Eigen<R>, DeviceSolverError> {
    let (batch, n, cols) = shape(a)?; options(o)?;
    if n != cols { return Err(DeviceSolverError::InvalidInput("square Hermitian matrix required")); }
    let work = empty(a, &[batch, n, n])?;
    let v = empty(a, &if vectors { [batch, n, n] } else { [0, 0, 0] })?;
    let values = alloc(&a.real, [batch, n], a.real.dtype); let info = alloc(&a.real, [batch], DType::I32); let sweeps = alloc(&a.real, [batch], DType::I32);
    let mut submitted = prepare(a, &work, &info, &sweeps, true, upper, false);
    if vectors { submitted += identity(a, &v); }
    if batch == 0 || n == 0 { return Ok(Eigen { values, vectors: v, info, sweeps, submitted_kernels: submitted }); }
    let pairs = (n + n % 2) / 2; super::elements(&[batch, pairs, 3])?;
    let angles = alloc(&a.real, [batch, pairs, 3], a.real.dtype);
    submitted += match a.real.dtype {
        DType::F32 => eigen_submit::<f32, R>(&work, &v, &angles, &values, &info, &sweeps, vectors, o),
        DType::F64 => eigen_submit::<f64, R>(&work, &v, &angles, &values, &info, &sweeps, vectors, o), _ => unreachable!(),
    };
    let (values, permutation, launches) = sorted(a, &values, n, false); submitted += launches;
    let vectors = if vectors { let result = gather(&v, &permutation, &info)?; submitted += result.submitted_kernels; result.value } else { v };
    Ok(Eigen { values, vectors, info, sweeps, submitted_kernels: submitted })
}

fn eigen_submit<F: HostFloat, R: Runtime>(
    w: &Matrix<R>, v: &Matrix<R>, angles: &RudaTensor<R>, values: &RudaTensor<R>, info: &RudaTensor<R>, sweeps: &RudaTensor<R>,
    vectors: bool, o: DeviceJacobiOptions,
) -> usize {
    let sizes = w.real.meta.shape(); let batch = sizes[0]; let n = sizes[1]; let slots = n + n % 2; let pairs = slots / 2;
    let check = |sweep: u32| {
        kernels::eigen_check::launch::<F, R>(&w.real.client, grid(batch), RudaDim::new_1d(256),
            w.real.clone().into_array_arg(), w.imag.clone().into_array_arg(), info.clone().into_array_arg(), sweeps.clone().into_array_arg(),
            n as u32, sweep, F::scalar(o.absolute_tolerance), F::scalar(o.relative_tolerance));
    };
    check(0); let mut submitted = 1;
    for sweep in 0..o.max_sweeps {
        for round in 0..slots - 1 {
            kernels::eigen_angles::launch::<F, R>(&w.real.client, grid((batch * pairs).div_ceil(256)), RudaDim::new_1d(256),
                w.real.clone().into_array_arg(), w.imag.clone().into_array_arg(), angles.clone().into_array_arg(), info.clone().into_array_arg(), n as u32, round as u32);
            for left in [true, false] {
                kernels::eigen_rotate::launch::<F, R>(&w.real.client, grid((batch * pairs * n).div_ceil(256)), RudaDim::new_1d(256),
                    w.real.clone().into_array_arg(), w.imag.clone().into_array_arg(), v.real.clone().into_array_arg(), v.imag.clone().into_array_arg(),
                    angles.clone().into_array_arg(), info.clone().into_array_arg(), n as u32, round as u32, left, vectors);
            }
            submitted += 3;
        }
        check(sweep + 1); submitted += 1;
    }
    submitted += finish(info);
    kernels::eigen_values::launch::<F, R>(&w.real.client, grid((batch * n).div_ceil(256)), RudaDim::new_1d(256),
        w.real.clone().into_array_arg(), values.clone().into_array_arg(), info.clone().into_array_arg(), n as u32);
    submitted + 1
}

fn adjoint<R: Runtime>(a: &Matrix<R>) -> Result<complex_qr::Transform<R>, DeviceSolverError> {
    let (batch, m, n) = shape(a)?; let value = empty(a, &[batch, n, m])?; let dummy = alloc(&a.real, [0], DType::I32);
    let submitted = initialize(a, &value, &dummy, 3, false);
    Ok(complex_qr::Transform { value, submitted_kernels: submitted })
}

fn complete<R: Runtime>(a: &Matrix<R>, values: &RudaTensor<R>, info: &RudaTensor<R>, full: bool) -> Result<complex_qr::Transform<R>, DeviceSolverError> {
    let (_, m, n) = shape(a)?; let columns = if full { m } else { n };
    let qr = complex_qr::factor(a)?; let mut result = complex_qr::form_q(&qr.packed, &qr.tau, columns)?;
    result.submitted_kernels += qr.submitted_kernels;
    let count = result.value.real.meta.num_elements();
    if count != 0 {
        macro_rules! run { ($f:ty) => {{
            kernels::complete::launch::<$f, R>(&a.real.client, grid(count.div_ceil(256)), RudaDim::new_1d(256),
                a.real.clone().into_array_arg(), a.imag.clone().into_array_arg(), result.value.real.clone().into_array_arg(), result.value.imag.clone().into_array_arg(),
                values.clone().into_array_arg(), info.clone().into_array_arg(), m as u32, n as u32, columns as u32);
        }}; }
        match a.real.dtype { DType::F32 => run!(f32), DType::F64 => run!(f64), _ => unreachable!() }
        result.submitted_kernels += 1;
    }
    Ok(result)
}

pub fn svd<R: Runtime>(a: &Matrix<R>, full: bool, vectors: bool, o: DeviceJacobiOptions) -> Result<Svd<R>, DeviceSolverError> {
    let (batch, m, n) = shape(a)?; options(o)?; let rows = m.max(n); let cols = m.min(n); let wide = m < n;
    let work = empty(a, &[batch, rows, cols])?;
    let info = alloc(&a.real, [batch], DType::I32); let sweeps = alloc(&a.real, [batch], DType::I32);
    let values = alloc(&a.real, [batch, cols], a.real.dtype);
    let mut submitted = prepare(a, &work, &info, &sweeps, false, false, wide);
    if batch == 0 || cols == 0 {
        let u = empty(a, &if vectors { [batch, m, if full { m } else { cols }] } else { [0, 0, 0] })?;
        let vh = empty(a, &if vectors { [batch, if full { n } else { cols }, n] } else { [0, 0, 0] })?;
        if vectors && full { submitted += identity(a, &u) + identity(a, &vh); }
        return Ok(Svd { u, singular_values: values, vh, info, sweeps, submitted_kernels: submitted });
    }
    let v = empty(a, &if vectors { [batch, cols, cols] } else { [0, 0, 0] })?;
    if vectors { submitted += identity(a, &v); }
    let slots = cols + cols % 2; let pairs = slots / 2; let rounds = slots - 1;
    super::elements(&[batch, rounds, pairs])?;
    let flags = alloc(&a.real, [batch, rounds, pairs], DType::I32);
    submitted += match a.real.dtype {
        DType::F32 => svd_submit::<f32, R>(&work, &v, &values, &flags, &info, &sweeps, vectors, o),
        DType::F64 => svd_submit::<f64, R>(&work, &v, &values, &flags, &info, &sweeps, vectors, o), _ => unreachable!(),
    };
    let (singular_values, permutation, launches) = sorted(a, &values, cols, true); submitted += launches;
    if !vectors { return Ok(Svd { u: empty(a, &[0, 0, 0])?, singular_values, vh: v, info, sweeps, submitted_kernels: submitted }); }
    let left = gather(&work, &permutation, &info)?; let right = gather(&v, &permutation, &info)?;
    submitted += left.submitted_kernels + right.submitted_kernels;
    let left = complete(&left.value, &singular_values, &info, full)?; submitted += left.submitted_kernels;
    let (u, vh) = if wide { let result = adjoint(&left.value)?; submitted += result.submitted_kernels; (right.value, result.value) }
        else { let result = adjoint(&right.value)?; submitted += result.submitted_kernels; (left.value, result.value) };
    Ok(Svd { u, singular_values, vh, info, sweeps, submitted_kernels: submitted })
}

fn svd_submit<F: HostFloat, R: Runtime>(
    w: &Matrix<R>, v: &Matrix<R>, values: &RudaTensor<R>, flags: &RudaTensor<R>, info: &RudaTensor<R>, sweeps: &RudaTensor<R>,
    vectors: bool, o: DeviceJacobiOptions,
) -> usize {
    let sizes = w.real.meta.shape(); let (batch, m, n) = (sizes[0], sizes[1], sizes[2]); let slots = n + n % 2; let pairs = slots / 2;
    let pair = |round: usize, apply: bool| {
        kernels::svd_pair::launch::<F, R>(&w.real.client, grid(batch * pairs), RudaDim::new_1d(256),
            w.real.clone().into_array_arg(), w.imag.clone().into_array_arg(), v.real.clone().into_array_arg(), v.imag.clone().into_array_arg(),
            flags.clone().into_array_arg(), info.clone().into_array_arg(), m as u32, n as u32, round as u32,
            F::scalar(o.absolute_tolerance), F::scalar(o.relative_tolerance), apply, vectors);
    };
    let check = |sweep: u32| {
        for round in 0..slots - 1 { pair(round, false); }
        kernels::svd_check::launch::<F, R>(&w.real.client, grid(batch), RudaDim::new_1d(256),
            w.real.clone().into_array_arg(), w.imag.clone().into_array_arg(), flags.clone().into_array_arg(), info.clone().into_array_arg(), sweeps.clone().into_array_arg(),
            m as u32, n as u32, sweep);
    };
    check(0); let mut submitted = slots;
    for sweep in 0..o.max_sweeps {
        for round in 0..slots - 1 { pair(round, true); }
        check(sweep + 1); submitted += 2 * slots - 1;
    }
    submitted += finish(info);
    kernels::svd_values::launch::<F, R>(&w.real.client, grid(batch * n), RudaDim::new_1d(256),
        w.real.clone().into_array_arg(), w.imag.clone().into_array_arg(), values.clone().into_array_arg(), info.clone().into_array_arg(), m as u32, n as u32);
    submitted + 1
}
