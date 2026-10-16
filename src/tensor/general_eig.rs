use super::{
    DeviceSolverError, complex::{Matrix, empty, grid, shape}, general_eig_kernel as kernels,
    large::{HostFloat, alloc},
};
use ruda_core::tensor::DType;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::tensor::RudaTensor;

pub struct Eigen<R: Runtime> {
    pub values: Matrix<R>,
    pub vectors: Matrix<R>,
    pub info: RudaTensor<R>,
    pub submitted_kernels: usize,
}

pub fn eig<R: Runtime>(a: &Matrix<R>, vectors: bool) -> Result<Eigen<R>, DeviceSolverError> {
    let (batch, n, cols) = shape(a)?;
    if n != cols { return Err(DeviceSolverError::InvalidInput("general eigen decomposition requires square matrices")); }
    super::elements(&[batch, 2])?;
    let work = empty(a, &[batch, n, n])?;
    let z = empty(a, &if vectors { [batch, n, n] } else { [0, 0, 0] })?;
    let scratch = empty(a, &[batch, 1, n])?;
    let values = empty(a, &[batch, 1, n])?;
    let balance = alloc(&a.real, [batch, n], a.real.dtype);
    let scaling = alloc(&a.real, [batch, 2], a.real.dtype);
    let info = alloc(&a.real, [batch], DType::I32);
    let mut submitted = 0;
    if batch != 0 {
        submitted += match a.real.dtype {
            DType::F32 => submit::<f32, R>(a, &work, &z, &scratch, &values, &balance, &scaling, &info, vectors, false),
            DType::F64 => submit::<f64, R>(a, &work, &z, &scratch, &values, &balance, &scaling, &info, vectors, true),
            _ => unreachable!(),
        };
    }
    let vectors = if vectors { work } else { z };
    Ok(Eigen { values, vectors, info, submitted_kernels: submitted })
}

fn submit<F: HostFloat, R: Runtime>(
    a: &Matrix<R>, h: &Matrix<R>, z: &Matrix<R>, scratch: &Matrix<R>, values: &Matrix<R>,
    balance: &RudaTensor<R>, scaling: &RudaTensor<R>, info: &RudaTensor<R>, vectors: bool, wide: bool,
) -> usize {
    let batch = a.real.meta.shape()[0]; let n = a.real.meta.shape()[1];
    let client = &a.real.client; let blocks = grid(batch); let dim = RudaDim::new_1d(256);
    kernels::prepare::launch::<F, R>(client, blocks, dim,
        a.real.clone().into_tensor_arg(), a.imag.clone().into_tensor_arg(),
        h.real.clone().into_array_arg(), h.imag.clone().into_array_arg(),
        z.real.clone().into_array_arg(), z.imag.clone().into_array_arg(),
        balance.clone().into_array_arg(), scaling.clone().into_array_arg(), info.clone().into_array_arg(), n as u32, vectors, wide);
    if n == 0 { return 1; }
    kernels::balance::launch::<F, R>(client, grid(batch), dim,
        h.real.clone().into_array_arg(), h.imag.clone().into_array_arg(), balance.clone().into_array_arg(),
        info.clone().into_array_arg(), n as u32, wide);
    kernels::hessenberg::launch::<F, R>(client, grid(batch), dim,
        h.real.clone().into_array_arg(), h.imag.clone().into_array_arg(),
        z.real.clone().into_array_arg(), z.imag.clone().into_array_arg(),
        scratch.real.clone().into_array_arg(), scratch.imag.clone().into_array_arg(),
        info.clone().into_array_arg(), n as u32, vectors, wide);
    kernels::schur::launch::<F, R>(client, grid(batch), dim,
        h.real.clone().into_array_arg(), h.imag.clone().into_array_arg(),
        z.real.clone().into_array_arg(), z.imag.clone().into_array_arg(),
        values.real.clone().into_array_arg(), values.imag.clone().into_array_arg(),
        scaling.clone().into_array_arg(), info.clone().into_array_arg(), n as u32, vectors, wide);
    if !vectors { return 4; }
    let x = empty(a, &[batch, n, n]).expect("eigenvector workspace size already checked");
    kernels::triangular_vectors::launch::<F, R>(client, grid(batch * n), dim,
        h.real.clone().into_array_arg(), h.imag.clone().into_array_arg(),
        x.real.clone().into_array_arg(), x.imag.clone().into_array_arg(),
        info.clone().into_array_arg(), n as u32, wide);
    kernels::backtransform::launch::<F, R>(client, grid(batch * n), dim,
        z.real.clone().into_array_arg(), z.imag.clone().into_array_arg(),
        x.real.clone().into_array_arg(), x.imag.clone().into_array_arg(),
        h.real.clone().into_array_arg(), h.imag.clone().into_array_arg(),
        balance.clone().into_array_arg(), info.clone().into_array_arg(), n as u32);
    kernels::check_vectors::launch::<F, R>(client, grid(batch), dim,
        h.real.clone().into_array_arg(), h.imag.clone().into_array_arg(), info.clone().into_array_arg(), n as u32);
    7
}
