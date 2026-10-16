use super::{DeviceSolverError, factorization, large::alloc, complex_kernel as kernels};
use ruda_core::tensor::DType;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::tensor::{RudaTensor, contiguous::into_contiguous};

pub struct Matrix<R: Runtime> {
    pub real: RudaTensor<R>,
    pub imag: RudaTensor<R>,
}

pub struct Lu<R: Runtime> {
    pub packed: Matrix<R>,
    pub pivots: RudaTensor<R>,
    pub info: RudaTensor<R>,
    pub submitted_kernels: usize,
}

pub struct Factor<R: Runtime> {
    pub factor: Matrix<R>,
    pub info: RudaTensor<R>,
    pub submitted_kernels: usize,
}

pub struct Solution<R: Runtime> {
    pub solution: Matrix<R>,
    pub info: RudaTensor<R>,
    pub submitted_kernels: usize,
}

pub struct Determinant<R: Runtime> {
    pub determinant: Matrix<R>,
    pub sign: Matrix<R>,
    pub log_absolute: RudaTensor<R>,
    pub submitted_kernels: usize,
}

pub struct Unpacked<R: Runtime> {
    pub permutation: Matrix<R>,
    pub lower: Matrix<R>,
    pub upper: Matrix<R>,
    pub submitted_kernels: usize,
}

pub(super) fn shape<R: Runtime>(a: &Matrix<R>) -> Result<(usize, usize, usize), DeviceSolverError> {
    let result = factorization::shape(&a.real)?;
    if factorization::shape(&a.imag)? != result { return Err(DeviceSolverError::InvalidInput("complex component shapes differ")); }
    factorization::same(&a.real, &a.imag)?;
    Ok(result)
}

pub(super) fn empty<R: Runtime>(a: &Matrix<R>, sizes: &[usize]) -> Result<Matrix<R>, DeviceSolverError> {
    super::elements(sizes)?;
    Ok(Matrix { real: alloc(&a.real, sizes.to_vec(), a.real.dtype), imag: alloc(&a.real, sizes.to_vec(), a.real.dtype) })
}

pub(super) fn contiguous<R: Runtime>(a: &Matrix<R>) -> Matrix<R> {
    Matrix { real: into_contiguous(a.real.clone()), imag: into_contiguous(a.imag.clone()) }
}

pub(super) fn grid(blocks: usize) -> RudaCount {
    let x = blocks.min(131072);
    RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

pub(super) fn initialize<R: Runtime>(a: &Matrix<R>, out: &Matrix<R>, info: &RudaTensor<R>, mode: u32, upper: bool) -> usize {
    let count = out.real.meta.num_elements().max(info.meta.num_elements());
    if count == 0 { return 0; }
    let sizes = out.real.meta.shape(); let m = sizes[1] as u32; let n = sizes[2] as u32;
    macro_rules! run { ($f:ty) => {{
        kernels::initialize::launch::<$f, R>(&a.real.client, grid(count.div_ceil(256)), RudaDim::new_1d(256),
            a.real.clone().into_tensor_arg(), a.imag.clone().into_tensor_arg(), out.real.clone().into_array_arg(),
            out.imag.clone().into_array_arg(), info.clone().into_array_arg(), m, n, mode, upper);
    }}; }
    match a.real.dtype { DType::F32 => run!(f32), DType::F64 => run!(f64), _ => unreachable!() }
    1
}

pub fn identity<R: Runtime>(a: &Matrix<R>) -> Result<Matrix<R>, DeviceSolverError> {
    let (batch, n, m) = shape(a)?;
    if n != m { return Err(DeviceSolverError::InvalidInput("square matrix required")); }
    let result = empty(a, &[batch, n, n])?; let dummy = alloc(&a.real, [0], DType::I32);
    initialize(a, &result, &dummy, 2, false);
    Ok(result)
}

pub fn lu_factor<R: Runtime>(a: &Matrix<R>, pivot: bool) -> Result<Lu<R>, DeviceSolverError> {
    let (batch, m, n) = shape(a)?; let k = m.min(n);
    if m > i32::MAX as usize { return Err(DeviceSolverError::SizeOverflow); }
    let mut result = Lu { packed: empty(a, &[batch, m, n])?, pivots: alloc(&a.real, [batch, k], DType::I32),
        info: alloc(&a.real, [batch], DType::I32), submitted_kernels: 0 };
    result.submitted_kernels += initialize(a, &result.packed, &result.info, 0, false);
    if batch == 0 { return Ok(result); }
    for j in 0..k {
        macro_rules! run { ($f:ty, $wide:expr) => {{
            kernels::lu_panel::launch::<$f, R>(&a.real.client, grid(batch), RudaDim::new_1d(256),
                result.packed.real.clone().into_array_arg(), result.packed.imag.clone().into_array_arg(),
                result.pivots.clone().into_array_arg(), result.info.clone().into_array_arg(), m as u32, n as u32, j as u32, pivot, $wide);
            if j + 1 < m && j + 1 < n {
                kernels::trailing::launch::<$f, R>(&a.real.client, grid((batch * (m-j-1) * (n-j-1)).div_ceil(256)), RudaDim::new_1d(256),
                    result.packed.real.clone().into_array_arg(), result.packed.imag.clone().into_array_arg(), result.info.clone().into_array_arg(),
                    m as u32, n as u32, j as u32, false);
            }
        }}; }
        match a.real.dtype { DType::F32 => run!(f32, false), DType::F64 => run!(f64, true), _ => unreachable!() }
        result.submitted_kernels += 1 + usize::from(j + 1 < m && j + 1 < n);
    }
    Ok(result)
}

pub fn cholesky_factor<R: Runtime>(a: &Matrix<R>, upper: bool) -> Result<Factor<R>, DeviceSolverError> {
    let (batch, n, m) = shape(a)?;
    if n != m { return Err(DeviceSolverError::InvalidInput("square Hermitian matrix required")); }
    let mut result = Factor { factor: empty(a, &[batch, n, n])?, info: alloc(&a.real, [batch], DType::I32), submitted_kernels: 0 };
    result.submitted_kernels += initialize(a, &result.factor, &result.info, 1, upper);
    if batch == 0 { return Ok(result); }
    for j in 0..n {
        macro_rules! run { ($f:ty) => {{
            kernels::cholesky_panel::launch::<$f, R>(&a.real.client, grid(batch), RudaDim::new_1d(256),
                result.factor.real.clone().into_array_arg(), result.factor.imag.clone().into_array_arg(), result.info.clone().into_array_arg(), n as u32, j as u32);
            if j + 1 < n {
                kernels::trailing::launch::<$f, R>(&a.real.client, grid((batch * (n-j-1) * (n-j-1)).div_ceil(256)), RudaDim::new_1d(256),
                    result.factor.real.clone().into_array_arg(), result.factor.imag.clone().into_array_arg(), result.info.clone().into_array_arg(),
                    n as u32, n as u32, j as u32, true);
            }
        }}; }
        match a.real.dtype { DType::F32 => run!(f32), DType::F64 => run!(f64), _ => unreachable!() }
        result.submitted_kernels += 1 + usize::from(j + 1 < n);
    }
    if upper {
        let transposed = empty(a, &[batch, n, n])?; let dummy = alloc(&a.real, [0], DType::I32);
        result.submitted_kernels += initialize(&result.factor, &transposed, &dummy, 3, false);
        result.factor = transposed;
    }
    Ok(result)
}

fn solve_dispatch<R: Runtime>(
    a: &Matrix<R>, pivots: Option<&RudaTensor<R>>, b: &Matrix<R>, upper: bool, transpose: bool, conjugate: bool, unit: bool,
) -> Result<Solution<R>, DeviceSolverError> {
    let (batch, n, m) = shape(a)?; let (bb, rows, nrhs) = shape(b)?;
    factorization::same(&a.real, &b.real)?;
    if n != m || bb != batch || rows != n { return Err(DeviceSolverError::InvalidInput("square factors and compatible complex RHS required")); }
    if let Some(p) = pivots {
        if p.dtype != DType::I32 || p.meta.num_elements() != batch * n || !a.real.client.same_execution_queue(&p.client) {
            return Err(DeviceSolverError::InvalidInput("compatible zero-based int32 LU pivots required"));
        }
    }
    let a = contiguous(a); let dummy = alloc(&a.real, [0], DType::I32);
    let p = pivots.map(|value| into_contiguous(value.clone())).unwrap_or(dummy);
    let mut result = Solution { solution: empty(b, &[batch, n, nrhs])?, info: alloc(&a.real, [batch], DType::I32), submitted_kernels: 0 };
    result.submitted_kernels += initialize(b, &result.solution, &result.info, 0, false);
    if batch != 0 && nrhs != 0 && n != 0 {
        macro_rules! run { ($f:ty, $wide:expr) => {{
            kernels::solve::launch::<$f, R>(&a.real.client, grid((batch * nrhs).div_ceil(256)), RudaDim::new_1d(256),
                a.real.clone().into_array_arg(), a.imag.clone().into_array_arg(), p.clone().into_array_arg(),
                result.solution.real.clone().into_array_arg(), result.solution.imag.clone().into_array_arg(), result.info.clone().into_array_arg(),
                n as u32, nrhs as u32, pivots.is_some(), upper, transpose, conjugate, unit, $wide);
        }}; }
        match a.real.dtype { DType::F32 => run!(f32, false), DType::F64 => run!(f64, true), _ => unreachable!() }
        result.submitted_kernels += 1;
    }
    Ok(result)
}

pub fn lu_solve<R: Runtime>(a: &Matrix<R>, pivots: &RudaTensor<R>, b: &Matrix<R>, adjoint: bool) -> Result<Solution<R>, DeviceSolverError> {
    solve_dispatch(a, Some(pivots), b, false, adjoint, adjoint, false)
}

pub fn triangular_solve<R: Runtime>(a: &Matrix<R>, b: &Matrix<R>, upper: bool, transpose: bool, conjugate: bool, unit: bool) -> Result<Solution<R>, DeviceSolverError> {
    solve_dispatch(a, None, b, upper, transpose, conjugate, unit)
}

pub fn cholesky_solve<R: Runtime>(a: &Matrix<R>, b: &Matrix<R>, upper: bool) -> Result<Solution<R>, DeviceSolverError> {
    let first = triangular_solve(a, b, upper, upper, upper, false)?;
    let mut result = triangular_solve(a, &first.solution, upper, !upper, !upper, false)?;
    result.submitted_kernels += first.submitted_kernels;
    Ok(result)
}

pub fn determinant<R: Runtime>(factor: &Lu<R>) -> Result<Determinant<R>, DeviceSolverError> {
    let a = &factor.packed; let (batch, n, m) = shape(a)?;
    if n != m { return Err(DeviceSolverError::InvalidInput("square LU factor required")); }
    if factor.pivots.dtype != DType::I32 || factor.pivots.meta.num_elements() != batch * n
        || !a.real.client.same_execution_queue(&factor.pivots.client) {
        return Err(DeviceSolverError::InvalidInput("compatible zero-based int32 LU pivots required"));
    }
    let mut result = Determinant { determinant: empty(a, &[batch])?, sign: empty(a, &[batch])?,
        log_absolute: alloc(&a.real, [batch], a.real.dtype), submitted_kernels: 0 };
    if batch != 0 {
        let a = contiguous(a); let pivots = into_contiguous(factor.pivots.clone());
        macro_rules! run { ($f:ty) => {{
            kernels::determinant::launch::<$f, R>(&a.real.client, grid(batch.div_ceil(256)), RudaDim::new_1d(256),
                a.real.clone().into_array_arg(), a.imag.clone().into_array_arg(), pivots.clone().into_array_arg(),
                result.determinant.real.clone().into_array_arg(), result.determinant.imag.clone().into_array_arg(),
                result.sign.real.clone().into_array_arg(), result.sign.imag.clone().into_array_arg(), result.log_absolute.clone().into_array_arg(), n as u32);
        }}; }
        match a.real.dtype { DType::F32 => run!(f32), DType::F64 => run!(f64), _ => unreachable!() }
        result.submitted_kernels += 1;
    }
    Ok(result)
}

pub fn lu_unpack<R: Runtime>(a: &Matrix<R>, pivots: &RudaTensor<R>, data: bool, permutation: bool) -> Result<Unpacked<R>, DeviceSolverError> {
    let (batch, m, n) = shape(a)?; let k = m.min(n);
    if m > i32::MAX as usize { return Err(DeviceSolverError::SizeOverflow); }
    if pivots.dtype != DType::I32 || pivots.meta.num_elements() != batch * k || !a.real.client.same_execution_queue(&pivots.client) {
        return Err(DeviceSolverError::InvalidInput("compatible zero-based int32 LU pivots required"));
    }
    let mut result = Unpacked {
        permutation: empty(a, &if permutation { [batch, m, m] } else { [0, 0, 0] })?,
        lower: empty(a, &if data { [batch, m, k] } else { [0, 0, 0] })?,
        upper: empty(a, &if data { [batch, k, n] } else { [0, 0, 0] })?, submitted_kernels: 0,
    };
    let a = contiguous(a); let pivots = into_contiguous(pivots.clone());
    for (kind, out) in [&result.permutation, &result.lower, &result.upper].iter().enumerate() {
        let count = out.real.meta.num_elements(); if count == 0 { continue; }
        macro_rules! run { ($f:ty) => {{
            kernels::unpack::launch::<$f, R>(&a.real.client, grid(count.div_ceil(256)), RudaDim::new_1d(256),
                a.real.clone().into_array_arg(), a.imag.clone().into_array_arg(), pivots.clone().into_array_arg(),
                out.real.clone().into_array_arg(), out.imag.clone().into_array_arg(), m as u32, n as u32, kind as u32);
        }}; }
        match a.real.dtype { DType::F32 => run!(f32), DType::F64 => run!(f64), _ => unreachable!() }
        result.submitted_kernels += 1;
    }
    Ok(result)
}
