use super::{
    BatchedCholeskyOptions, BatchedCholeskyResult, DeviceSolverError, LargeSolverOptions,
    LuFactorResult,
    factorization::{check, same, shape},
    large::{HostFloat, alloc},
};
use ruda_core::tensor::{DType, Metadata};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{calculate_ruda_count_elemwise, prelude::*};
use ruda_kernel::tensor::{RudaTensor, contiguous::into_contiguous, permutation::swap_dims};
pub struct LinearSolveResult<R: Runtime> {
    pub solution: RudaTensor<R>,
    pub info: RudaTensor<R>,
    pub submitted_kernels: usize,
}
impl<R: Runtime> LinearSolveResult<R> {
    pub fn check_status_sync(&self) -> Result<(), DeviceSolverError> {
        check(&self.info)
    }
}

/// Reuse a square LU factor; zero-based sequential pivots, real adjoint is
/// transpose. No refactorization and no source/RHS modification.
pub fn lu_solve_factor<R: Runtime>(
    lu: &RudaTensor<R>,
    pivots: &RudaTensor<R>,
    b: &RudaTensor<R>,
    transpose: bool,
) -> Result<LinearSolveResult<R>, DeviceSolverError> {
    let (batch, n, n2) = shape(lu)?;
    let (bb, mb, nrhs) = shape(b)?;
    same(lu, b)?;
    if n != n2
        || bb != batch
        || mb != n
        || pivots.dtype != DType::I32
        || pivots.meta.shape().iter().product::<usize>() != batch * n
        || !lu.client.same_execution_queue(&pivots.client)
    {
        return Err(DeviceSolverError::InvalidInput(
            "square LU, zero-based I32 pivots and compatible RHS required",
        ));
    }
    solve_dispatch(
        lu,
        Some(&into_contiguous(pivots.clone())),
        b,
        n,
        nrhs,
        transpose,
        false,
        false,
        true,
    )
}
pub fn triangular_solve<R: Runtime>(
    a: &RudaTensor<R>,
    b: &RudaTensor<R>,
    upper: bool,
    transpose: bool,
    unit_diagonal: bool,
) -> Result<LinearSolveResult<R>, DeviceSolverError> {
    let (batch, n, n2) = shape(a)?;
    let (bb, mb, nrhs) = shape(b)?;
    same(a, b)?;
    if n != n2 || bb != batch || mb != n {
        return Err(DeviceSolverError::InvalidInput(
            "triangular A=[batch,n,n], B=[batch,n,nrhs] required",
        ));
    }
    solve_dispatch(a, None, b, n, nrhs, transpose, upper, unit_diagonal, false)
}
fn solve_dispatch<R: Runtime>(
    a: &RudaTensor<R>,
    p: Option<&RudaTensor<R>>,
    b: &RudaTensor<R>,
    n: usize,
    nrhs: usize,
    transpose: bool,
    upper: bool,
    unit: bool,
    lu: bool,
) -> Result<LinearSolveResult<R>, DeviceSolverError> {
    let batch = a.meta.shape()[0];
    let mut r = LinearSolveResult {
        solution: alloc(b, [batch, n, nrhs], b.dtype),
        info: alloc(b, [batch], DType::I32),
        submitted_kernels: 0,
    };
    let flags = alloc(b, [batch, nrhs], DType::I32);
    let dummy = alloc(b, [0], DType::I32);
    let p = p.unwrap_or(&dummy);
    if batch > 0 {
        match a.dtype {
            DType::F32 => {
                solve_submit::<f32, R>(a, p, b, &flags, &mut r, n, nrhs, transpose, upper, unit, lu)
            }
            DType::F64 => {
                solve_submit::<f64, R>(a, p, b, &flags, &mut r, n, nrhs, transpose, upper, unit, lu)
            }
            _ => unreachable!(),
        }
    }
    Ok(r)
}
fn solve_submit<F: HostFloat, R: Runtime>(
    a: &RudaTensor<R>,
    p: &RudaTensor<R>,
    b: &RudaTensor<R>,
    flags: &RudaTensor<R>,
    r: &mut LinearSolveResult<R>,
    n: usize,
    nrhs: usize,
    transpose: bool,
    upper: bool,
    unit: bool,
    lu: bool,
) {
    let batch = a.meta.shape()[0];
    let count = batch * nrhs;
    if count > 0 {
        let d = RudaDim::new(a.client.properties(), count);
        solve_rhs::launch::<F, R>(
            &a.client,
            calculate_ruda_count_elemwise(&a.client, count, d),
            d,
            a.clone().into_tensor_arg(),
            p.clone().into_array_arg(),
            b.clone().into_tensor_arg(),
            r.solution.clone().into_array_arg(),
            flags.clone().into_array_arg(),
            n as u32,
            nrhs as u32,
            transpose,
            upper,
            unit,
            lu,
        );
        r.submitted_kernels += 1;
    }
    let d = RudaDim::new(a.client.properties(), batch);
    reduce_info::launch::<R>(
        &a.client,
        calculate_ruda_count_elemwise(&a.client, batch, d),
        d,
        flags.clone().into_array_arg(),
        r.info.clone().into_array_arg(),
        nrhs as u32,
    );
    r.submitted_kernels += 1;
}
pub fn identity_matrix<R: Runtime>(like: &RudaTensor<R>, batch: usize, n: usize) -> RudaTensor<R> {
    let t = alloc(like, [batch, n, n], like.dtype);
    let count = batch * n * n;
    if count > 0 {
        let d = RudaDim::new(like.client.properties(), count);
        match like.dtype {
            DType::F32 => identity::launch::<f32, R>(
                &like.client,
                calculate_ruda_count_elemwise(&like.client, count, d),
                d,
                t.clone().into_array_arg(),
                n as u32,
            ),
            DType::F64 => identity::launch::<f64, R>(
                &like.client,
                calculate_ruda_count_elemwise(&like.client, count, d),
                d,
                t.clone().into_array_arg(),
                n as u32,
            ),
            _ => panic!("identity requires real dtype"),
        }
    }
    t
}
pub fn lu_inverse_factor<R: Runtime>(
    factor: &LuFactorResult<R>,
) -> Result<LinearSolveResult<R>, DeviceSolverError> {
    let (batch, n, n2) = shape(&factor.packed_lu)?;
    if n != n2 {
        return Err(DeviceSolverError::InvalidInput("square inverse"));
    }
    lu_solve_factor(
        &factor.packed_lu,
        &factor.pivots,
        &identity_matrix(&factor.packed_lu, batch, n),
        false,
    )
}
pub fn triangular_inverse<R: Runtime>(
    a: &RudaTensor<R>,
    upper: bool,
    unit: bool,
) -> Result<LinearSolveResult<R>, DeviceSolverError> {
    let (batch, n, n2) = shape(a)?;
    if n != n2 {
        return Err(DeviceSolverError::InvalidInput("square triangular inverse"));
    }
    triangular_solve(a, &identity_matrix(a, batch, n), upper, false, unit)
}

/// Select one triangle, exactly as real LAPACK-style UPLO contracts require.
/// The other triangle (including its nonfinite values) is not read.
pub fn symmetric_from_triangle<R: Runtime>(
    a: &RudaTensor<R>,
    upper: bool,
) -> Result<RudaTensor<R>, DeviceSolverError> {
    let (batch, n, n2) = shape(a)?;
    if n != n2 {
        return Err(DeviceSolverError::InvalidInput("square symmetric matrix"));
    }
    let out = alloc(a, [batch, n, n], a.dtype);
    let count = batch * n * n;
    if count > 0 {
        let d = RudaDim::new(a.client.properties(), count);
        match a.dtype {
            DType::F32 => symmetrize::launch::<f32, R>(
                &a.client,
                calculate_ruda_count_elemwise(&a.client, count, d),
                d,
                a.clone().into_tensor_arg(),
                out.clone().into_array_arg(),
                n as u32,
                upper,
            ),
            DType::F64 => symmetrize::launch::<f64, R>(
                &a.client,
                calculate_ruda_count_elemwise(&a.client, count, d),
                d,
                a.clone().into_tensor_arg(),
                out.clone().into_array_arg(),
                n as u32,
                upper,
            ),
            _ => unreachable!(),
        }
    }
    Ok(out)
}
pub struct CholeskyFactorResult<R: Runtime> {
    pub factor: RudaTensor<R>,
    pub info: RudaTensor<R>,
    pub submitted_kernels: usize,
}
impl<R: Runtime> CholeskyFactorResult<R> {
    pub fn check_status_sync(&self) -> Result<(), DeviceSolverError> {
        check(&self.info)
    }
}
pub fn cholesky_factor_blocked<R: Runtime>(
    a: &RudaTensor<R>,
    upper: bool,
) -> Result<CholeskyFactorResult<R>, DeviceSolverError> {
    let (batch, n, n2) = shape(a)?;
    if n != n2 {
        return Err(DeviceSolverError::InvalidInput("square Cholesky matrix"));
    }
    let prepared = symmetric_from_triangle(a, upper)?;
    let empty = alloc(a, [batch, n, 0], a.dtype);
    let mut r = BatchedCholeskyResult {
        lower: alloc(a, [batch, n, n], a.dtype),
        solution: empty.clone(),
        info: alloc(a, [batch], DType::I32),
        submitted_kernels: 0,
    };
    if batch > 0 {
        match a.dtype {
            DType::F32 => super::blocked::submit::<f32, R>(
                &prepared,
                &empty,
                &mut r,
                n,
                0,
                BatchedCholeskyOptions {
                    symmetry_relative_tolerance: 0.0,
                    ..Default::default()
                },
                true,
            ),
            DType::F64 => super::blocked::submit::<f64, R>(
                &prepared,
                &empty,
                &mut r,
                n,
                0,
                BatchedCholeskyOptions {
                    symmetry_relative_tolerance: 0.0,
                    ..Default::default()
                },
                true,
            ),
            _ => unreachable!(),
        }
    }
    let factor = if upper {
        swap_dims(r.lower, 1, 2)
    } else {
        r.lower
    };
    Ok(CholeskyFactorResult {
        factor,
        info: r.info,
        submitted_kernels: r.submitted_kernels,
    })
}
pub fn cholesky_solve_factor<R: Runtime>(
    factor: &RudaTensor<R>,
    b: &RudaTensor<R>,
    upper: bool,
) -> Result<LinearSolveResult<R>, DeviceSolverError> {
    let y = triangular_solve(factor, b, upper, upper, false)?;
    let mut x = triangular_solve(factor, &y.solution, upper, !upper, false)?;
    x.submitted_kernels += y.submitted_kernels;
    Ok(x)
}
pub fn cholesky_inverse_factor<R: Runtime>(
    factor: &RudaTensor<R>,
    upper: bool,
) -> Result<LinearSolveResult<R>, DeviceSolverError> {
    let (batch, n, n2) = shape(factor)?;
    if n != n2 {
        return Err(DeviceSolverError::InvalidInput("square Cholesky inverse"));
    }
    cholesky_solve_factor(factor, &identity_matrix(factor, batch, n), upper)
}

pub struct LeastSquaresResult<R: Runtime> {
    pub solution: RudaTensor<R>,
    pub residuals: RudaTensor<R>,
    pub info: RudaTensor<R>,
    pub submitted_kernels: usize,
}
impl<R: Runtime> LeastSquaresResult<R> {
    pub fn check_status_sync(&self) -> Result<(), DeviceSolverError> {
        check(&self.info)
    }
}
/// CUDA-gels-style full-rank least squares. Tall input: Q^T B and R solve;
/// wide input: QR of A^T, R^T solve, then Q [Y;0] for minimum-norm solution.
pub fn least_squares_qr<R: Runtime>(
    a: &RudaTensor<R>,
    b: &RudaTensor<R>,
) -> Result<LeastSquaresResult<R>, DeviceSolverError> {
    let (batch, m, n) = shape(a)?;
    let (bb, mb, nrhs) = shape(b)?;
    same(a, b)?;
    if bb != batch || mb != m {
        return Err(DeviceSolverError::InvalidInput("least squares RHS"));
    }
    let o = LargeSolverOptions {
        absolute_tolerance: 0.0,
        relative_tolerance: 0.0,
        ..Default::default()
    };
    if m >= n {
        let f = super::qr_factor_blocked(a, o)?;
        let t = super::qr_apply_q(&f.packed, &f.tau, b, true)?;
        let r = super::qr_extract_r(&f.packed, n)?;
        let top = RudaTensor::new(
            t.client.clone(),
            t.handle.clone(),
            Metadata::new([batch, n, nrhs], t.meta.strides().clone()),
            t.device.clone(),
            t.dtype,
        );
        let x = triangular_solve(&r, &top, true, false, false)?;
        let residuals = alloc(
            a,
            if m > n {
                vec![batch, nrhs]
            } else {
                vec![batch, 0]
            },
            a.dtype,
        );
        if m > n && batch * nrhs > 0 {
            let count = batch * nrhs;
            let d = RudaDim::new(a.client.properties(), count);
            match a.dtype {
                DType::F32 => ls_residual::launch::<f32, R>(
                    &a.client,
                    calculate_ruda_count_elemwise(&a.client, count, d),
                    d,
                    t.into_array_arg(),
                    residuals.clone().into_array_arg(),
                    m as u32,
                    n as u32,
                    nrhs as u32,
                ),
                DType::F64 => ls_residual::launch::<f64, R>(
                    &a.client,
                    calculate_ruda_count_elemwise(&a.client, count, d),
                    d,
                    t.into_array_arg(),
                    residuals.clone().into_array_arg(),
                    m as u32,
                    n as u32,
                    nrhs as u32,
                ),
                _ => unreachable!(),
            }
        }
        merge_factor_info(&f.info, &x.info);
        Ok(LeastSquaresResult {
            solution: x.solution,
            residuals,
            info: x.info,
            submitted_kernels: f.submitted_kernels + x.submitted_kernels + 3,
        })
    } else {
        let f = super::qr_factor_blocked(&swap_dims(a.clone(), 1, 2), o)?;
        let r = super::qr_extract_r(&f.packed, m)?;
        let y = triangular_solve(&r, b, true, true, false)?;
        let seed = alloc(a, [batch, n, nrhs], a.dtype);
        let count = batch * n * nrhs;
        if count > 0 {
            let d = RudaDim::new(a.client.properties(), count);
            match a.dtype {
                DType::F32 => ls_embed::launch::<f32, R>(
                    &a.client,
                    calculate_ruda_count_elemwise(&a.client, count, d),
                    d,
                    y.solution.clone().into_tensor_arg(),
                    seed.clone().into_array_arg(),
                    m as u32,
                    n as u32,
                    nrhs as u32,
                ),
                DType::F64 => ls_embed::launch::<f64, R>(
                    &a.client,
                    calculate_ruda_count_elemwise(&a.client, count, d),
                    d,
                    y.solution.clone().into_tensor_arg(),
                    seed.clone().into_array_arg(),
                    m as u32,
                    n as u32,
                    nrhs as u32,
                ),
                _ => unreachable!(),
            }
        }
        let x = super::qr_apply_q(&f.packed, &f.tau, &seed, false)?;
        merge_factor_info(&f.info, &y.info);
        Ok(LeastSquaresResult {
            solution: x,
            residuals: alloc(a, [batch, 0], a.dtype),
            info: y.info,
            submitted_kernels: f.submitted_kernels + y.submitted_kernels + 3,
        })
    }
}
fn merge_factor_info<R: Runtime>(factor: &RudaTensor<R>, info: &RudaTensor<R>) {
    let count = info.meta.num_elements();
    if count > 0 {
        let d = RudaDim::new(info.client.properties(), count);
        merge_info::launch::<R>(
            &info.client,
            calculate_ruda_count_elemwise(&info.client, count, d),
            d,
            factor.clone().into_array_arg(),
            info.clone().into_array_arg(),
        );
    }
}
/// Complete orthonormal columns with the Householder Q nullspace, preserving
/// existing columns. Used for full SVD matrices; no signal leaves the GPU.
pub fn orthogonal_complete<R: Runtime>(
    thin: &RudaTensor<R>,
) -> Result<RudaTensor<R>, DeviceSolverError> {
    let (batch, m, k) = shape(thin)?;
    if k > m {
        return Err(DeviceSolverError::InvalidInput(
            "orthogonal completion columns",
        ));
    }
    let f = super::qr_factor_blocked(
        thin,
        LargeSolverOptions {
            absolute_tolerance: 0.0,
            relative_tolerance: 0.0,
            ..Default::default()
        },
    )?;
    let q = super::qr_form_q(&f.packed, &f.tau, m)?;
    let full = alloc(thin, [batch, m, m], thin.dtype);
    let count = batch * m * m;
    if count > 0 {
        let d = RudaDim::new(thin.client.properties(), count);
        match thin.dtype {
            DType::F32 => complete_basis::launch::<f32, R>(
                &thin.client,
                calculate_ruda_count_elemwise(&thin.client, count, d),
                d,
                thin.clone().into_tensor_arg(),
                q.into_tensor_arg(),
                full.clone().into_array_arg(),
                m as u32,
                k as u32,
            ),
            DType::F64 => complete_basis::launch::<f64, R>(
                &thin.client,
                calculate_ruda_count_elemwise(&thin.client, count, d),
                d,
                thin.clone().into_tensor_arg(),
                q.into_tensor_arg(),
                full.clone().into_array_arg(),
                m as u32,
                k as u32,
            ),
            _ => unreachable!(),
        }
    }
    Ok(full)
}

pub struct DeterminantResult<R: Runtime> {
    pub determinant: RudaTensor<R>,
    pub sign: RudaTensor<R>,
    pub log_absolute: RudaTensor<R>,
}
pub struct LuUnpackResult<R: Runtime> {
    pub permutation: RudaTensor<R>,
    pub lower: RudaTensor<R>,
    pub upper: RudaTensor<R>,
}
pub fn lu_unpack_factor<R: Runtime>(
    packed: &RudaTensor<R>,
    pivots: &RudaTensor<R>,
    data: bool,
    permutation: bool,
) -> Result<LuUnpackResult<R>, DeviceSolverError> {
    let (batch, m, n) = shape(packed)?;
    let k = m.min(n);
    if pivots.dtype != DType::I32
        || pivots.meta.num_elements() != batch * k
        || !packed.client.same_execution_queue(&pivots.client)
    {
        return Err(DeviceSolverError::InvalidInput("LU unpack pivots"));
    }
    let pivots = into_contiguous(pivots.clone());
    let r = LuUnpackResult {
        permutation: alloc(
            packed,
            if permutation {
                vec![batch, m, m]
            } else {
                vec![0]
            },
            packed.dtype,
        ),
        lower: alloc(
            packed,
            if data { vec![batch, m, k] } else { vec![0] },
            packed.dtype,
        ),
        upper: alloc(
            packed,
            if data { vec![batch, k, n] } else { vec![0] },
            packed.dtype,
        ),
    };
    if batch > 0 {
        let d = RudaDim::new(packed.client.properties(), batch);
        match packed.dtype {
            DType::F32 => unpack_lu::launch::<f32, R>(
                &packed.client,
                calculate_ruda_count_elemwise(&packed.client, batch, d),
                d,
                packed.clone().into_tensor_arg(),
                pivots.clone().into_array_arg(),
                r.permutation.clone().into_array_arg(),
                r.lower.clone().into_array_arg(),
                r.upper.clone().into_array_arg(),
                m as u32,
                n as u32,
                data,
                permutation,
            ),
            DType::F64 => unpack_lu::launch::<f64, R>(
                &packed.client,
                calculate_ruda_count_elemwise(&packed.client, batch, d),
                d,
                packed.clone().into_tensor_arg(),
                pivots.clone().into_array_arg(),
                r.permutation.clone().into_array_arg(),
                r.lower.clone().into_array_arg(),
                r.upper.clone().into_array_arg(),
                m as u32,
                n as u32,
                data,
                permutation,
            ),
            _ => unreachable!(),
        }
    }
    Ok(r)
}
pub fn determinant_from_lu<R: Runtime>(
    f: &LuFactorResult<R>,
) -> Result<DeterminantResult<R>, DeviceSolverError> {
    let (batch, n, n2) = shape(&f.packed_lu)?;
    if n != n2 {
        return Err(DeviceSolverError::InvalidInput("square determinant"));
    }
    let r = DeterminantResult {
        determinant: alloc(&f.packed_lu, [batch], f.packed_lu.dtype),
        sign: alloc(&f.packed_lu, [batch], f.packed_lu.dtype),
        log_absolute: alloc(&f.packed_lu, [batch], f.packed_lu.dtype),
    };
    if batch > 0 {
        let d = RudaDim::new(f.packed_lu.client.properties(), batch);
        match f.packed_lu.dtype {
            DType::F32 => determinant::launch::<f32, R>(
                &f.packed_lu.client,
                calculate_ruda_count_elemwise(&f.packed_lu.client, batch, d),
                d,
                f.packed_lu.clone().into_tensor_arg(),
                f.pivots.clone().into_array_arg(),
                f.info.clone().into_array_arg(),
                r.determinant.clone().into_array_arg(),
                r.sign.clone().into_array_arg(),
                r.log_absolute.clone().into_array_arg(),
                n as u32,
            ),
            DType::F64 => determinant::launch::<f64, R>(
                &f.packed_lu.client,
                calculate_ruda_count_elemwise(&f.packed_lu.client, batch, d),
                d,
                f.packed_lu.clone().into_tensor_arg(),
                f.pivots.clone().into_array_arg(),
                f.info.clone().into_array_arg(),
                r.determinant.clone().into_array_arg(),
                r.sign.clone().into_array_arg(),
                r.log_absolute.clone().into_array_arg(),
                n as u32,
            ),
            _ => unreachable!(),
        }
    }
    Ok(r)
}

#[ruda(launch)]
fn solve_rhs<F: Float>(
    a: &Tensor<F>,
    p: &Array<i32>,
    b: &Tensor<F>,
    x: &mut Array<F>,
    flags: &mut Array<i32>,
    order: u32,
    rhs: u32,
    #[comptime] transpose: bool,
    #[comptime] upper: bool,
    #[comptime] unit: bool,
    #[comptime] lu: bool,
) {
    let n = order as usize;
    let nr = rhs as usize;
    let pos = ABSOLUTE_POS;
    if pos >= flags.len() {
        terminate!();
    }
    let sys = pos / nr;
    let col = pos % nr;
    let base = sys * n * nr;
    let mut code = 0i32;
    for i in 0..n {
        let z = b[sys * b.stride(0) + i * b.stride(1) + col * b.stride(2)];
        x[base + i * nr + col] = z;
        if crate::tensor::large::bad::<F>(z) {
            code = -1;
        }
    }
    if comptime!(lu) {
        for i in 0..n {
            if p[sys * n + i] < 0 || p[sys * n + i] >= order as i32 {
                code = -2;
            }
        }
        if code != -2 {
            if comptime!(!transpose) {
                for i in 0..n {
                    let j = p[sys * n + i] as usize;
                    let z = x[base + i * nr + col];
                    x[base + i * nr + col] = x[base + j * nr + col];
                    x[base + j * nr + col] = z;
                }
                for i in 0..n {
                    let mut z = x[base + i * nr + col];
                    for j in 0..i {
                        z -= a[sys * a.stride(0) + i * a.stride(1) + j * a.stride(2)]
                            * x[base + j * nr + col];
                    }
                    x[base + i * nr + col] = z;
                }
                let mut end = n;
                while end > 0 {
                    let i = end - 1;
                    let mut z = x[base + i * nr + col];
                    for j in i + 1..n {
                        z -= a[sys * a.stride(0) + i * a.stride(1) + j * a.stride(2)]
                            * x[base + j * nr + col];
                    }
                    let diag = a[sys * a.stride(0) + i * a.stride(1) + i * a.stride(2)];
                    if diag == F::new(0.0) && code == 0 {
                        code = (i + 1) as i32;
                    }
                    x[base + i * nr + col] = z / diag;
                    end -= 1;
                }
            } else {
                for i in 0..n {
                    let mut z = x[base + i * nr + col];
                    for j in 0..i {
                        z -= a[sys * a.stride(0) + j * a.stride(1) + i * a.stride(2)]
                            * x[base + j * nr + col];
                    }
                    let diag = a[sys * a.stride(0) + i * a.stride(1) + i * a.stride(2)];
                    if diag == F::new(0.0) && code == 0 {
                        code = (i + 1) as i32;
                    }
                    x[base + i * nr + col] = z / diag;
                }
                let mut end = n;
                while end > 0 {
                    let i = end - 1;
                    let mut z = x[base + i * nr + col];
                    for j in i + 1..n {
                        z -= a[sys * a.stride(0) + j * a.stride(1) + i * a.stride(2)]
                            * x[base + j * nr + col];
                    }
                    x[base + i * nr + col] = z;
                    end -= 1;
                }
                let mut end = n;
                while end > 0 {
                    let i = end - 1;
                    let j = p[sys * n + i] as usize;
                    let z = x[base + i * nr + col];
                    x[base + i * nr + col] = x[base + j * nr + col];
                    x[base + j * nr + col] = z;
                    end -= 1;
                }
            }
        }
    } else {
        for step in 0..n {
            let i = if comptime!(upper != transpose) {
                n - 1 - step
            } else {
                step
            };
            let mut z = x[base + i * nr + col];
            if comptime!(upper != transpose) {
                for j in i + 1..n {
                    let off = if comptime!(transpose) {
                        sys * a.stride(0) + j * a.stride(1) + i * a.stride(2)
                    } else {
                        sys * a.stride(0) + i * a.stride(1) + j * a.stride(2)
                    };
                    z -= a[off] * x[base + j * nr + col];
                }
            } else {
                for j in 0..i {
                    let off = if comptime!(transpose) {
                        sys * a.stride(0) + j * a.stride(1) + i * a.stride(2)
                    } else {
                        sys * a.stride(0) + i * a.stride(1) + j * a.stride(2)
                    };
                    z -= a[off] * x[base + j * nr + col];
                }
            }
            let mut diag = F::new(1.0);
            if comptime!(!unit) {
                diag = a[sys * a.stride(0) + i * a.stride(1) + i * a.stride(2)];
                if diag == F::new(0.0) && code == 0 {
                    code = (i + 1) as i32;
                }
            }
            x[base + i * nr + col] = z / diag;
        }
    }
    if code == 0 {
        for i in 0..n {
            if crate::tensor::large::bad::<F>(x[base + i * nr + col]) {
                code = -3;
            }
        }
    }
    flags[pos] = code;
}
#[ruda(launch)]
fn reduce_info(flags: &Array<i32>, info: &mut Array<i32>, rhs: u32) {
    let sys = ABSOLUTE_POS;
    if sys >= info.len() {
        terminate!();
    }
    let mut code = 0i32;
    for j in 0..rhs as usize {
        let z = flags[sys * rhs as usize + j];
        if z < 0 {
            code = z;
        } else if z > 0 && code >= 0 && (code == 0 || z < code) {
            code = z;
        }
    }
    info[sys] = code;
}
#[ruda(launch)]
fn merge_info(factor: &Array<i32>, info: &mut Array<i32>) {
    let pos = ABSOLUTE_POS;
    if pos < info.len() {
        let code = factor[pos];
        if code != 0 {
            info[pos] = code;
        }
    }
}
#[ruda(launch)]
fn identity<F: Float>(out: &mut Array<F>, order: u32) {
    let pos = ABSOLUTE_POS;
    if pos >= out.len() {
        terminate!();
    }
    let n = order as usize;
    let mut z = F::new(0.0);
    if (pos / n) % n == pos % n {
        z = F::new(1.0);
    }
    out[pos] = z;
}
#[ruda(launch)]
fn symmetrize<F: Float>(a: &Tensor<F>, out: &mut Array<F>, order: u32, #[comptime] upper: bool) {
    let pos = ABSOLUTE_POS;
    if pos >= out.len() {
        terminate!();
    }
    let n = order as usize;
    let sys = pos / (n * n);
    let i = (pos / n) % n;
    let j = pos % n;
    let mut r = i;
    let mut c = j;
    if (comptime!(upper) && i > j) || (comptime!(!upper) && i < j) {
        r = j;
        c = i;
    }
    out[pos] = a[sys * a.stride(0) + r * a.stride(1) + c * a.stride(2)];
}
#[ruda(launch)]
fn ls_residual<F: Float>(t: &Array<F>, out: &mut Array<F>, rows: u32, solved: u32, rhs: u32) {
    let pos = ABSOLUTE_POS;
    if pos >= out.len() {
        terminate!();
    }
    let m = rows as usize;
    let n = solved as usize;
    let nr = rhs as usize;
    let sys = pos / nr;
    let j = pos % nr;
    let mut z = F::new(0.0);
    for i in n..m {
        let x = t[sys * m * nr + i * nr + j];
        z += x * x;
    }
    out[pos] = z;
}
#[ruda(launch)]
fn ls_embed<F: Float>(y: &Tensor<F>, out: &mut Array<F>, small: u32, rows: u32, rhs: u32) {
    let pos = ABSOLUTE_POS;
    if pos >= out.len() {
        terminate!();
    }
    let m = small as usize;
    let n = rows as usize;
    let nr = rhs as usize;
    let sys = pos / (n * nr);
    let i = (pos / nr) % n;
    let j = pos % nr;
    let mut z = F::new(0.0);
    if i < m {
        z = y[sys * y.stride(0) + i * y.stride(1) + j * y.stride(2)];
    }
    out[pos] = z;
}
#[ruda(launch)]
fn complete_basis<F: Float>(
    thin: &Tensor<F>,
    q: &Tensor<F>,
    full: &mut Array<F>,
    rows: u32,
    cols: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos >= full.len() {
        terminate!();
    }
    let m = rows as usize;
    let k = cols as usize;
    let sys = pos / (m * m);
    let i = (pos / m) % m;
    let j = pos % m;
    let z = if j < k {
        thin[sys * thin.stride(0) + i * thin.stride(1) + j * thin.stride(2)]
    } else {
        q[sys * q.stride(0) + i * q.stride(1) + j * q.stride(2)]
    };
    full[pos] = z;
}
#[ruda(launch)]
fn determinant<F: Float>(
    lu: &Tensor<F>,
    p: &Array<i32>,
    info: &Array<i32>,
    det: &mut Array<F>,
    sign: &mut Array<F>,
    logabs: &mut Array<F>,
    order: u32,
) {
    let sys = ABSOLUTE_POS;
    if sys >= det.len() {
        terminate!();
    }
    let n = order as usize;
    let mut value = F::new(1.0);
    let mut s = F::new(1.0);
    let mut l = F::new(0.0);
    let mut singular = false;
    for i in 0..n {
        let d = lu[sys * lu.stride(0) + i * lu.stride(1) + i * lu.stride(2)];
        if p[sys * n + i] != i as i32 {
            value = -value;
            s = -s;
        }
        value *= d;
        if d < F::new(0.0) {
            s = -s;
        }
        if d == F::new(0.0) {
            singular = true;
        }
        l += d.abs().ln();
    }
    if singular {
        value = F::new(0.0);
        s = F::new(0.0);
        l = F::new(0.0).ln();
    }
    if info[sys] < 0 {
        value = F::new(0.0) / F::new(0.0);
        s = value;
        l = value;
    }
    det[sys] = value;
    sign[sys] = s;
    logabs[sys] = l;
}
#[ruda(launch)]
fn unpack_lu<F: Float>(
    packed: &Tensor<F>,
    pivots: &Array<i32>,
    p: &mut Array<F>,
    l: &mut Array<F>,
    u: &mut Array<F>,
    rows: u32,
    cols: u32,
    #[comptime] data: bool,
    #[comptime] permutation: bool,
) {
    let sys = ABSOLUTE_POS;
    if sys >= packed.shape(0) {
        terminate!();
    }
    let m = rows as usize;
    let n = cols as usize;
    let k = m.min(n);
    if comptime!(data) {
        for i in 0..m {
            for j in 0..k {
                let mut z = F::new(0.0);
                if i == j {
                    z = F::new(1.0);
                } else if i > j {
                    z = packed
                        [sys * packed.stride(0) + i * packed.stride(1) + j * packed.stride(2)];
                }
                l[sys * m * k + i * k + j] = z;
            }
        }
        for i in 0..k {
            for j in 0..n {
                let mut z = F::new(0.0);
                if i <= j {
                    z = packed
                        [sys * packed.stride(0) + i * packed.stride(1) + j * packed.stride(2)];
                }
                u[sys * k * n + i * n + j] = z;
            }
        }
    }
    if comptime!(permutation) {
        let base = sys * m * m;
        for i in 0..m {
            for j in 0..m {
                let mut z = F::new(0.0);
                if i == j {
                    z = F::new(1.0);
                }
                p[base + i * m + j] = z;
            }
        }
        for i in 0..k {
            let row = pivots[sys * k + i] as usize;
            if row < m {
                for j in 0..m {
                    let z = p[base + j * m + i];
                    let other = p[base + j * m + row];
                    p[base + j * m + i] = other;
                    p[base + j * m + row] = z;
                }
            }
        }
    }
}
