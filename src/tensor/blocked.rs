use super::{BatchedCholeskyOptions, BatchedCholeskyResult, DeviceSolverError};
use ruda_core::tensor::DType;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_contiguous_dtype};

pub fn cholesky_solve_blocked<R: Runtime>(
    a: &RudaTensor<R>,
    b: &RudaTensor<R>,
    options: BatchedCholeskyOptions,
) -> Result<BatchedCholeskyResult<R>, DeviceSolverError> {
    if !matches!(a.dtype, DType::F32 | DType::F64)
        || a.dtype != b.dtype
        || a.qparams.is_some()
        || b.qparams.is_some()
        || a.meta.num_dims() != 3
        || b.meta.num_dims() != 3
    {
        return Err(DeviceSolverError::InvalidInput(
            "blocked Cholesky requires unquantized F32/F64 tensors with three axes",
        ));
    }
    let shape = a.meta.shape();
    let rhs = b.meta.shape();
    let (batch, n, nrhs) = (shape[0], shape[1], rhs[2]);
    if n == 0 || shape[2] != n || rhs[0] != batch || rhs[1] != n || nrhs == 0 {
        return Err(DeviceSolverError::InvalidInput(
            "A=[batch,n,n], B=[batch,n,nrhs] with positive n and nrhs",
        ));
    }
    if !a.client.same_execution_queue(&b.client) {
        return Err(DeviceSolverError::DifferentQueue);
    }
    for tensor in [a, b] {
        super::elements(tensor.meta.shape())?;
    }
    for value in [
        options.diagonal_shift,
        options.symmetry_absolute_tolerance,
        options.symmetry_relative_tolerance,
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(DeviceSolverError::InvalidInput(
                "finite nonnegative shift/tolerances required",
            ));
        }
    }
    let lower =
        empty_device_contiguous_dtype(a.client.clone(), a.device.clone(), shape.clone(), a.dtype);
    let solution =
        empty_device_contiguous_dtype(a.client.clone(), a.device.clone(), rhs.clone(), a.dtype);
    let info = empty_device_contiguous_dtype(
        a.client.clone(),
        a.device.clone(),
        [batch].into(),
        DType::I32,
    );
    let mut result = BatchedCholeskyResult {
        lower,
        solution,
        info,
        submitted_kernels: 0,
    };
    if batch == 0 {
        return Ok(result);
    }
    match a.dtype {
        DType::F32 => submit::<f32, R>(a, b, &mut result, n, nrhs, options, false),
        DType::F64 => submit::<f64, R>(a, b, &mut result, n, nrhs, options, false),
        _ => unreachable!(),
    }
    Ok(result)
}

pub(super) fn submit<F: Float + RudaElement, R: Runtime>(
    a: &RudaTensor<R>,
    b: &RudaTensor<R>,
    result: &mut BatchedCholeskyResult<R>,
    n: usize,
    nrhs: usize,
    options: BatchedCholeskyOptions,
    retain_failed_factor: bool,
) {
    let client = &a.client;
    let batch = a.meta.shape()[0];
    let dim = RudaDim::new(client.properties(), batch);
    initialize::launch::<F, R>(
        client,
        calculate_ruda_count_elemwise(client, batch, dim),
        dim,
        a.clone().into_tensor_arg(),
        b.clone().into_tensor_arg(),
        result.lower.clone().into_array_arg(),
        result.solution.clone().into_array_arg(),
        result.info.clone().into_array_arg(),
        n as u32,
        nrhs as u32,
        F::new(options.diagonal_shift),
        F::new(options.symmetry_absolute_tolerance),
        F::new(options.symmetry_relative_tolerance),
    );
    result.submitted_kernels += 1;
    for start in (0..n).step_by(32) {
        let end = (start + 32).min(n);
        diagonal::launch::<F, R>(
            client,
            calculate_ruda_count_elemwise(client, batch, dim),
            dim,
            result.lower.clone().into_array_arg(),
            result.info.clone().into_array_arg(),
            n as u32,
            start as u32,
            end as u32,
        );
        result.submitted_kernels += 1;
        if end < n {
            let rows = batch * (n - end);
            let row_dim = RudaDim::new(client.properties(), rows);
            panel::launch::<F, R>(
                client,
                calculate_ruda_count_elemwise(client, rows, row_dim),
                row_dim,
                result.lower.clone().into_array_arg(),
                result.info.clone().into_array_arg(),
                n as u32,
                start as u32,
                end as u32,
            );
            let count = rows * (n - end);
            let update_dim = RudaDim::new(client.properties(), count);
            update::launch::<F, R>(
                client,
                calculate_ruda_count_elemwise(client, count, update_dim),
                update_dim,
                result.lower.clone().into_array_arg(),
                result.info.clone().into_array_arg(),
                n as u32,
                start as u32,
                end as u32,
            );
            result.submitted_kernels += 2;
        }
    }
    let count = batch * nrhs;
    if count > 0 {
        let solve_dim = RudaDim::new(client.properties(), count);
        solve::launch::<F, R>(
            client,
            calculate_ruda_count_elemwise(client, count, solve_dim),
            solve_dim,
            result.lower.clone().into_array_arg(),
            result.solution.clone().into_array_arg(),
            result.info.clone().into_array_arg(),
            n as u32,
            nrhs as u32,
        );
    }
    cleanup::launch::<F, R>(
        client,
        calculate_ruda_count_elemwise(client, batch, dim),
        dim,
        result.lower.clone().into_array_arg(),
        result.solution.clone().into_array_arg(),
        result.info.clone().into_array_arg(),
        n as u32,
        nrhs as u32,
        retain_failed_factor,
    );
    result.submitted_kernels += 1 + usize::from(count > 0);
}

#[ruda(launch)]
fn initialize<F: Float + RudaElement>(
    a: &Tensor<F>,
    b: &Tensor<F>,
    lower: &mut Array<F>,
    solution: &mut Array<F>,
    info: &mut Array<i32>,
    order: u32,
    rhs: u32,
    shift: F,
    atol: F,
    rtol: F,
) {
    let sys = ABSOLUTE_POS;
    if sys >= info.len() {
        terminate!();
    }
    let n = order as usize;
    let nr = rhs as usize;
    let base = sys * n * n;
    let bb = sys * n * nr;
    let mut code = 0i32;
    for i in 0..n * n {
        let offset = sys * a.stride(0) + (i / n) * a.stride(1) + (i % n) * a.stride(2);
        let value = a[offset];
        lower[base + i] = value;
        if nonfinite::<F>(value) {
            code = -1;
        }
    }
    for i in 0..n * nr {
        let offset = sys * b.stride(0) + (i / nr) * b.stride(1) + (i % nr) * b.stride(2);
        let value = b[offset];
        solution[bb + i] = value;
        if nonfinite::<F>(value) {
            code = -1;
        }
    }
    if code == 0 {
        for i in 0..n {
            for j in 0..i {
                let x = lower[base + i * n + j];
                let y = lower[base + j * n + i];
                if (x - y).abs() > atol + rtol * F::max(x.abs(), y.abs()) {
                    code = -2;
                }
            }
            lower[base + i * n + i] += shift;
        }
    }
    info[sys] = code;
}

#[ruda(launch)]
fn diagonal<F: Float>(
    lower: &mut Array<F>,
    info: &mut Array<i32>,
    order: u32,
    first: u32,
    last: u32,
) {
    let sys = ABSOLUTE_POS;
    if sys >= info.len() {
        terminate!();
    }
    if info[sys] != 0 {
        terminate!();
    }
    let n = order as usize;
    let base = sys * n * n;
    let start = first as usize;
    let end = last as usize;
    let mut code = 0i32;
    for j in start..end {
        if code == 0 {
            let mut value = lower[base + j * n + j];
            for k in start..j {
                let z = lower[base + j * n + k];
                value -= z * z;
            }
            if nonfinite::<F>(value) {
                code = -3;
            } else if value <= F::new(0.0) {
                lower[base + j * n + j] = value;
                code = (j + 1) as i32;
            } else {
                let root = value.sqrt();
                lower[base + j * n + j] = root;
                for i in j + 1..end {
                    let mut v = lower[base + i * n + j];
                    for k in start..j {
                        v -= lower[base + i * n + k] * lower[base + j * n + k];
                    }
                    v /= root;
                    lower[base + i * n + j] = v;
                    if nonfinite::<F>(v) {
                        code = -3;
                    }
                }
            }
        }
    }
    info[sys] = code;
}

#[ruda(launch)]
fn panel<F: Float>(lower: &mut Array<F>, info: &Array<i32>, order: u32, first: u32, last: u32) {
    let pos = ABSOLUTE_POS;
    let n = order as usize;
    let start = first as usize;
    let end = last as usize;
    let tail = n - end;
    let sys = pos / tail;
    if sys >= info.len() {
        terminate!();
    }
    if info[sys] != 0 {
        terminate!();
    }
    let row = end + pos % tail;
    let base = sys * n * n;
    for j in start..end {
        let mut value = lower[base + row * n + j];
        for k in start..j {
            value -= lower[base + row * n + k] * lower[base + j * n + k];
        }
        lower[base + row * n + j] = value / lower[base + j * n + j];
    }
}

#[ruda(launch)]
fn update<F: Float>(lower: &mut Array<F>, info: &Array<i32>, order: u32, first: u32, last: u32) {
    let pos = ABSOLUTE_POS;
    let n = order as usize;
    let start = first as usize;
    let end = last as usize;
    let tail = n - end;
    let sys = pos / (tail * tail);
    if sys >= info.len() {
        terminate!();
    }
    if info[sys] != 0 {
        terminate!();
    }
    let i = end + (pos / tail) % tail;
    let j = end + pos % tail;
    if j > i {
        terminate!();
    }
    let base = sys * n * n;
    let mut value = lower[base + i * n + j];
    for k in start..end {
        value -= lower[base + i * n + k] * lower[base + j * n + k];
    }
    lower[base + i * n + j] = value;
}

#[ruda(launch)]
fn solve<F: Float>(
    lower: &Array<F>,
    solution: &mut Array<F>,
    info: &Array<i32>,
    order: u32,
    rhs: u32,
) {
    let pos = ABSOLUTE_POS;
    let n = order as usize;
    let nr = rhs as usize;
    let sys = pos / nr;
    if sys >= info.len() {
        terminate!();
    }
    if info[sys] != 0 {
        terminate!();
    }
    let column = pos % nr;
    let base = sys * n * n;
    let bb = sys * n * nr;
    for i in 0..n {
        let mut value = solution[bb + i * nr + column];
        for k in 0..i {
            value -= lower[base + i * n + k] * solution[bb + k * nr + column];
        }
        solution[bb + i * nr + column] = value / lower[base + i * n + i];
    }
    let mut end = n;
    while end > 0 {
        let i = end - 1;
        let mut value = solution[bb + i * nr + column];
        for k in i + 1..n {
            value -= lower[base + k * n + i] * solution[bb + k * nr + column];
        }
        solution[bb + i * nr + column] = value / lower[base + i * n + i];
        end -= 1;
    }
}

#[ruda(launch)]
fn cleanup<F: Float>(
    lower: &mut Array<F>,
    solution: &mut Array<F>,
    info: &mut Array<i32>,
    order: u32,
    rhs: u32,
    #[comptime] retain_failed_factor: bool,
) {
    let sys = ABSOLUTE_POS;
    if sys >= info.len() {
        terminate!();
    }
    let n = order as usize;
    let nr = rhs as usize;
    let base = sys * n * n;
    let bb = sys * n * nr;
    let mut code = info[sys];
    if code == 0 {
        for i in 0..n * n {
            let value = lower[base + i];
            if nonfinite::<F>(value) {
                code = -3;
            }
        }
        for i in 0..n * nr {
            let value = solution[bb + i];
            if nonfinite::<F>(value) {
                code = -3;
            }
        }
    }
    if code != 0 && comptime!(!retain_failed_factor) {
        for i in 0..n * n {
            lower[base + i] = F::new(0.0);
        }
        for i in 0..n * nr {
            solution[bb + i] = F::new(0.0);
        }
    } else {
        for i in 0..n {
            for j in i + 1..n {
                lower[base + i * n + j] = F::new(0.0);
            }
        }
    }
    info[sys] = code;
}

#[ruda]
fn nonfinite<F: Float>(value: F) -> bool {
    bool::cast_from(value.is_nan()) || bool::cast_from(value.is_inf())
}
