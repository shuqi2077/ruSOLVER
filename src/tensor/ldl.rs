use super::{
    DeviceSolverError, complex::{Matrix, Solution, contiguous, empty, grid, initialize, shape},
    factorization, large::alloc, ldl_kernel as kernels,
};
use ruda_core::tensor::DType;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::tensor::{RudaTensor, contiguous::into_contiguous};

pub struct Factor<R: Runtime> {
    pub packed: Matrix<R>,
    pub pivots: RudaTensor<R>,
    pub info: RudaTensor<R>,
    pub submitted_kernels: usize,
}

pub fn real_parts<R: Runtime>(a: &RudaTensor<R>) -> Result<(Matrix<R>, usize), DeviceSolverError> {
    factorization::shape(a)?;
    let real = into_contiguous(a.clone()); let imag = alloc(&real, real.meta.shape().clone(), real.dtype);
    let count = real.meta.num_elements(); let mut submitted = 0;
    if count != 0 {
        macro_rules! run { ($f:ty) => {{
            kernels::zero::launch::<$f, R>(&real.client, grid(count.div_ceil(256)), RudaDim::new_1d(256), imag.clone().into_array_arg());
        }}; }
        match real.dtype { DType::F32 => run!(f32), DType::F64 => run!(f64), _ => unreachable!() }
        submitted += 1;
    }
    Ok((Matrix { real, imag }, submitted))
}

pub fn factor<R: Runtime>(a: &Matrix<R>, hermitian: bool) -> Result<Factor<R>, DeviceSolverError> {
    let (batch, n, m) = shape(a)?;
    if n != m { return Err(DeviceSolverError::InvalidInput("square LDL matrix required")); }
    if n > i32::MAX as usize { return Err(DeviceSolverError::SizeOverflow); }
    super::elements(&[batch, 4])?;
    let mut result = Factor { packed: empty(a, &[batch, n, n])?, pivots: alloc(&a.real, [batch, n], DType::I32),
        info: alloc(&a.real, [batch], DType::I32), submitted_kernels: 0 };
    let state = alloc(&a.real, [batch, 4], DType::I32); let work = empty(a, &[batch, n, 2])?;
    let count = result.packed.real.meta.num_elements().max(state.meta.num_elements());
    if count != 0 {
        macro_rules! prepare { ($f:ty) => {{
            kernels::prepare::launch::<$f, R>(&a.real.client, grid(count.div_ceil(256)), RudaDim::new_1d(256),
                a.real.clone().into_tensor_arg(), a.imag.clone().into_tensor_arg(), result.packed.real.clone().into_array_arg(), result.packed.imag.clone().into_array_arg(),
                state.clone().into_array_arg(), result.info.clone().into_array_arg(), n as u32, hermitian);
        }}; }
        match a.real.dtype { DType::F32 => prepare!(f32), DType::F64 => prepare!(f64), _ => unreachable!() }
        result.submitted_kernels += 1;
    }
    if batch == 0 || n == 0 { return Ok(result); }
    for _ in 0..n {
        macro_rules! run { ($f:ty, $wide:expr) => {{
            kernels::panel::launch::<$f, R>(&a.real.client, grid(batch), RudaDim::new_1d(256),
                result.packed.real.clone().into_array_arg(), result.packed.imag.clone().into_array_arg(), result.pivots.clone().into_array_arg(),
                state.clone().into_array_arg(), result.info.clone().into_array_arg(), n as u32, hermitian);
            kernels::weights::launch::<$f, R>(&a.real.client, grid((batch * n).div_ceil(256)), RudaDim::new_1d(256),
                result.packed.real.clone().into_array_arg(), result.packed.imag.clone().into_array_arg(), work.real.clone().into_array_arg(), work.imag.clone().into_array_arg(),
                state.clone().into_array_arg(), n as u32, hermitian, $wide);
            kernels::trailing::launch::<$f, R>(&a.real.client, grid((batch * n * n).div_ceil(256)), RudaDim::new_1d(256),
                result.packed.real.clone().into_array_arg(), result.packed.imag.clone().into_array_arg(), work.real.clone().into_array_arg(), work.imag.clone().into_array_arg(),
                state.clone().into_array_arg(), n as u32, hermitian);
            kernels::advance::launch::<$f, R>(&a.real.client, grid(batch), RudaDim::new_1d(256),
                result.packed.real.clone().into_array_arg(), result.packed.imag.clone().into_array_arg(), work.real.clone().into_array_arg(), work.imag.clone().into_array_arg(),
                state.clone().into_array_arg(), n as u32);
        }}; }
        match a.real.dtype { DType::F32 => run!(f32, false), DType::F64 => run!(f64, true), _ => unreachable!() }
        result.submitted_kernels += 4;
    }
    Ok(result)
}

pub fn solve<R: Runtime>(a: &Matrix<R>, pivots: &RudaTensor<R>, b: &Matrix<R>, hermitian: bool) -> Result<Solution<R>, DeviceSolverError> {
    let (batch, n, m) = shape(a)?; let (bb, rows, nrhs) = shape(b)?;
    factorization::same(&a.real, &b.real)?;
    if n != m || batch != bb || rows != n { return Err(DeviceSolverError::InvalidInput("square LDL factors and compatible RHS required")); }
    if n > i32::MAX as usize { return Err(DeviceSolverError::SizeOverflow); }
    if pivots.dtype != DType::I32 || pivots.qparams.is_some() || pivots.meta.num_elements() != batch * n || !a.real.client.same_execution_queue(&pivots.client) {
        return Err(DeviceSolverError::InvalidInput("compatible signed one-based int32 LDL pivots required"));
    }
    let a = contiguous(a); let pivots = into_contiguous(pivots.clone());
    let mut result = Solution { solution: empty(b, &[batch, n, nrhs])?, info: alloc(&a.real, [batch], DType::I32), submitted_kernels: 0 };
    result.submitted_kernels += initialize(b, &result.solution, &result.info, 0, false);
    if batch != 0 && n != 0 && nrhs != 0 {
        macro_rules! run { ($f:ty, $wide:expr) => {{
            kernels::solve::launch::<$f, R>(&a.real.client, grid((batch * nrhs).div_ceil(256)), RudaDim::new_1d(256),
                a.real.clone().into_array_arg(), a.imag.clone().into_array_arg(), pivots.clone().into_array_arg(),
                result.solution.real.clone().into_array_arg(), result.solution.imag.clone().into_array_arg(), result.info.clone().into_array_arg(),
                n as u32, nrhs as u32, hermitian, $wide);
        }}; }
        match a.real.dtype { DType::F32 => run!(f32, false), DType::F64 => run!(f64, true), _ => unreachable!() }
        result.submitted_kernels += 1;
    }
    Ok(result)
}
