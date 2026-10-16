use super::{
    DeviceSolverError, complex::{self, Matrix, contiguous, empty, grid, initialize, shape},
    complex_qr_kernel as kernels, factorization, large::alloc,
};
use ruda_core::tensor::DType;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::tensor::RudaTensor;

pub struct Qr<R: Runtime> {
    pub packed: Matrix<R>,
    pub tau: Matrix<R>,
    pub info: RudaTensor<R>,
    pub submitted_kernels: usize,
}

pub struct Transform<R: Runtime> {
    pub value: Matrix<R>,
    pub submitted_kernels: usize,
}

pub struct LeastSquares<R: Runtime> {
    pub solution: Matrix<R>,
    pub residuals: RudaTensor<R>,
    pub info: RudaTensor<R>,
    pub submitted_kernels: usize,
}

fn reflector<R: Runtime>(packed: &Matrix<R>, tau: &Matrix<R>, out: &Matrix<R>, j: usize, first: usize, adjoint: bool) -> usize {
    let sizes = packed.real.meta.shape(); let (batch, m, n) = (sizes[0], sizes[1], sizes[2]);
    let k = tau.real.meta.shape()[2]; let nrhs = out.real.meta.shape()[2];
    if batch == 0 || nrhs == first { return 0; }
    macro_rules! run { ($f:ty) => {
        kernels::apply::launch::<$f, R>(&packed.real.client, grid(batch * (nrhs - first)), RudaDim::new_1d(256),
            packed.real.clone().into_array_arg(), packed.imag.clone().into_array_arg(),
            tau.real.clone().into_array_arg(), tau.imag.clone().into_array_arg(),
            out.real.clone().into_array_arg(), out.imag.clone().into_array_arg(), batch as u32, m as u32, n as u32,
            k as u32, nrhs as u32, j as u32, first as u32, adjoint);
    }; }
    match packed.real.dtype { DType::F32 => run!(f32), DType::F64 => run!(f64), _ => unreachable!() }
    1
}

pub fn factor<R: Runtime>(a: &Matrix<R>) -> Result<Qr<R>, DeviceSolverError> {
    let (batch, m, n) = shape(a)?; let k = m.min(n);
    let mut result = Qr { packed: empty(a, &[batch, m, n])?, tau: empty(a, &[batch, 1, k])?,
        info: alloc(&a.real, [batch], DType::I32), submitted_kernels: 0 };
    result.submitted_kernels += initialize(a, &result.packed, &result.info, 0, false);
    if batch == 0 { return Ok(result); }
    for j in 0..k {
        macro_rules! run { ($f:ty, $wide:expr) => {
            kernels::panel::launch::<$f, R>(&a.real.client, grid(batch), RudaDim::new_1d(256),
                result.packed.real.clone().into_array_arg(), result.packed.imag.clone().into_array_arg(),
                result.tau.real.clone().into_array_arg(), result.tau.imag.clone().into_array_arg(),
                batch as u32, m as u32, n as u32, j as u32, $wide);
        }; }
        match a.real.dtype { DType::F32 => run!(f32, false), DType::F64 => run!(f64, true), _ => unreachable!() }
        result.submitted_kernels += 1 + reflector(&result.packed, &result.tau, &result.packed, j, j + 1, true);
    }
    Ok(result)
}

fn compatible_tau<R: Runtime>(packed: &Matrix<R>, tau: &Matrix<R>) -> Result<usize, DeviceSolverError> {
    let (batch, m, n) = shape(packed)?; let (tb, one, k) = shape(tau)?;
    factorization::same(&packed.real, &tau.real)?;
    if tb != batch || one != 1 || k > m.min(n) {
        return Err(DeviceSolverError::InvalidInput("complex tau requires [batch,1,k], k <= min(m,n)"));
    }
    Ok(k)
}

fn apply_in_place<R: Runtime>(packed: &Matrix<R>, tau: &Matrix<R>, out: &Matrix<R>, k: usize, adjoint: bool) -> usize {
    let mut submitted = 0;
    for step in 0..k {
        let j = if adjoint { step } else { k - 1 - step };
        submitted += reflector(packed, tau, out, j, 0, adjoint);
    }
    submitted
}

pub fn apply_q<R: Runtime>(packed: &Matrix<R>, tau: &Matrix<R>, b: &Matrix<R>, adjoint: bool) -> Result<Transform<R>, DeviceSolverError> {
    let k = compatible_tau(packed, tau)?;
    let (batch, m, _) = shape(packed)?; let (bb, bm, nrhs) = shape(b)?;
    factorization::same(&packed.real, &b.real)?;
    if bb != batch || bm != m { return Err(DeviceSolverError::InvalidInput("complex Q and RHS dimensions differ")); }
    let packed = contiguous(packed); let tau = contiguous(tau);
    let mut result = Transform { value: empty(b, &[batch, m, nrhs])?, submitted_kernels: 0 };
    let dummy = alloc(&b.real, [0], DType::I32);
    result.submitted_kernels += initialize(b, &result.value, &dummy, 0, false);
    result.submitted_kernels += apply_in_place(&packed, &tau, &result.value, k, adjoint);
    Ok(result)
}

pub fn form_q<R: Runtime>(packed: &Matrix<R>, tau: &Matrix<R>, columns: usize) -> Result<Transform<R>, DeviceSolverError> {
    let k = compatible_tau(packed, tau)?; let (batch, m, _) = shape(packed)?;
    if columns > m { return Err(DeviceSolverError::InvalidInput("Q columns exceed its order")); }
    let packed = contiguous(packed); let tau = contiguous(tau);
    let mut result = Transform { value: empty(&packed, &[batch, m, columns])?, submitted_kernels: 0 };
    let dummy = alloc(&packed.real, [0], DType::I32);
    result.submitted_kernels += initialize(&packed, &result.value, &dummy, 2, false);
    result.submitted_kernels += apply_in_place(&packed, &tau, &result.value, k, false);
    Ok(result)
}

fn resize<R: Runtime>(a: &Matrix<R>, rows: usize, cols: usize, upper: bool) -> Result<Transform<R>, DeviceSolverError> {
    let (batch, m, n) = shape(a)?;
    if rows > u32::MAX as usize || cols > u32::MAX as usize { return Err(DeviceSolverError::SizeOverflow); }
    let a = contiguous(a);
    let mut result = Transform { value: empty(&a, &[batch, rows, cols])?, submitted_kernels: 0 };
    let count = result.value.real.meta.num_elements();
    if count != 0 {
        macro_rules! run { ($f:ty) => {
            kernels::resize::launch::<$f, R>(&a.real.client, grid(count.div_ceil(256)), RudaDim::new_1d(256),
                a.real.clone().into_array_arg(), a.imag.clone().into_array_arg(),
                result.value.real.clone().into_array_arg(), result.value.imag.clone().into_array_arg(),
                m as u32, n as u32, rows as u32, cols as u32, upper);
        }; }
        match a.real.dtype { DType::F32 => run!(f32), DType::F64 => run!(f64), _ => unreachable!() }
        result.submitted_kernels += 1;
    }
    Ok(result)
}

pub fn extract_r<R: Runtime>(packed: &Matrix<R>, rows: usize) -> Result<Transform<R>, DeviceSolverError> {
    let (_, m, n) = shape(packed)?;
    if rows > m { return Err(DeviceSolverError::InvalidInput("R rows exceed the packed matrix")); }
    resize(packed, rows, n, true)
}

pub fn least_squares<R: Runtime>(a: &Matrix<R>, b: &Matrix<R>) -> Result<LeastSquares<R>, DeviceSolverError> {
    let (batch, m, n) = shape(a)?; let (bb, bm, nrhs) = shape(b)?;
    factorization::same(&a.real, &b.real)?;
    if batch != bb || m != bm { return Err(DeviceSolverError::InvalidInput("complex least-squares RHS dimensions differ")); }
    super::elements(&[batch, n, nrhs])?;
    if m >= n {
        let qr = factor(a)?;
        let y = apply_q(&qr.packed, &qr.tau, b, true)?;
        let r = resize(&qr.packed, n, n, true)?;
        let rhs = resize(&y.value, n, nrhs, false)?;
        let solved = complex::triangular_solve(&r.value, &rhs.value, true, false, false, false)?;
        let residuals = if m > n { alloc(&a.real, [batch, nrhs], a.real.dtype) } else { alloc(&a.real, [0, 0], a.real.dtype) };
        let mut submitted = qr.submitted_kernels + y.submitted_kernels + r.submitted_kernels + rhs.submitted_kernels + solved.submitted_kernels;
        if residuals.meta.num_elements() != 0 {
            macro_rules! run { ($f:ty) => {
                kernels::residuals::launch::<$f, R>(&a.real.client, grid(batch * nrhs), RudaDim::new_1d(256),
                    y.value.real.clone().into_array_arg(), y.value.imag.clone().into_array_arg(), residuals.clone().into_array_arg(),
                    m as u32, nrhs as u32, n as u32);
            }; }
            match a.real.dtype { DType::F32 => run!(f32), DType::F64 => run!(f64), _ => unreachable!() }
            submitted += 1;
        }
        Ok(LeastSquares { solution: solved.solution, residuals, info: solved.info, submitted_kernels: submitted })
    } else {
        let ah = empty(a, &[batch, n, m])?; let dummy = alloc(&a.real, [0], DType::I32);
        let mut submitted = initialize(a, &ah, &dummy, 3, false);
        let qr = factor(&ah)?;
        let r = resize(&qr.packed, m, m, true)?;
        let solved = complex::triangular_solve(&r.value, b, true, true, true, false)?;
        let padded = resize(&solved.solution, n, nrhs, false)?;
        submitted += qr.submitted_kernels + r.submitted_kernels + solved.submitted_kernels + padded.submitted_kernels;
        submitted += apply_in_place(&qr.packed, &qr.tau, &padded.value, m, false);
        Ok(LeastSquares { solution: padded.value, residuals: alloc(&a.real, [0, 0], a.real.dtype),
            info: solved.info, submitted_kernels: submitted })
    }
}
