// Taylor evaluation and coefficients adapt PyTorch ATen/native/LinearAlgebra.cpp.
/*
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
use super::{DeviceSolverError, complex::{Matrix, empty, grid}, factorization, large::alloc, matrix_exp_kernel as kernels};
use half::{bf16, f16};
use ruda_core::tensor::DType;
use ruda_kernel::dsl::prelude::*;

pub struct Exponential<R: Runtime> {
    pub value: Matrix<R>,
    pub submitted_kernels: usize,
}

fn shape<R: Runtime>(a: &Matrix<R>) -> Result<(usize, usize, usize), DeviceSolverError> {
    if !matches!(a.real.dtype, DType::F16 | DType::BF16) { return super::complex::shape(a); }
    factorization::same(&a.real, &a.imag)?;
    for part in [&a.real, &a.imag] {
        if part.qparams.is_some() || part.meta.num_dims() != 3 || part.meta.strides().len() != 3 {
            return Err(DeviceSolverError::InvalidInput("unquantized [batch,m,n] matrix components required"));
        }
        let sizes = part.meta.shape();
        let count = super::elements(sizes)?;
        if sizes.iter().any(|&n| n > u32::MAX as usize) { return Err(DeviceSolverError::SizeOverflow); }
        if count != 0 {
            let span = sizes.iter().zip(part.meta.strides().iter()).try_fold(1usize, |sum, (&size, &stride)| {
                (size - 1).checked_mul(stride).and_then(|offset| sum.checked_add(offset))
            }).ok_or(DeviceSolverError::SizeOverflow)?;
            if span > u32::MAX as usize { return Err(DeviceSolverError::SizeOverflow); }
            let start = part.handle.offset_start.unwrap_or(0);
            let available = part.handle.size().checked_sub(start)
                .and_then(|n| n.checked_sub(part.handle.offset_end.unwrap_or(0)))
                .ok_or(DeviceSolverError::SizeOverflow)?;
            if start % 2 != 0 || span as u64 * 2 > available {
                return Err(DeviceSolverError::InvalidInput("matrix layout exceeds/misaligns backing allocation"));
            }
        }
    }
    if a.real.meta.shape() != a.imag.meta.shape() {
        return Err(DeviceSolverError::InvalidInput("complex component shapes differ"));
    }
    let sizes = a.real.meta.shape();
    Ok((sizes[0], sizes[1], sizes[2]))
}

pub fn exp<R: Runtime>(a: &Matrix<R>, complex_input: bool) -> Result<Exponential<R>, DeviceSolverError> {
    let (batch, n, cols) = shape(a)?;
    if n != cols { return Err(DeviceSolverError::InvalidInput("matrix exponential requires square matrices")); }
    let value = empty(a, &[batch, n, n])?;
    let mut submitted = 0;
    if batch != 0 && n != 0 {
        let count = if n == 1 { 0 } else { super::elements(&[batch, 6, n, n])? };
        let work_dtype = if a.real.dtype == DType::F64 { DType::F64 } else { DType::F32 };
        let real = alloc(&a.real, [count], work_dtype);
        let imag = alloc(&a.real, [if complex_input { count } else { 0 }], work_dtype);
        macro_rules! run { ($s:ty, $f:ty, $wide:expr) => {
            kernels::evaluate::launch::<$s, $f, R>(&a.real.client, grid(batch), RudaDim::new_1d(256),
                a.real.clone().into_tensor_arg(), a.imag.clone().into_tensor_arg(),
                real.clone().into_array_arg(), imag.clone().into_array_arg(),
                value.real.clone().into_array_arg(), value.imag.clone().into_array_arg(),
                n as u32, batch as u32, complex_input, $wide, t8_coefficients($wide));
        }; }
        match a.real.dtype {
            DType::F16 => run!(f16, f32, false), DType::BF16 => run!(bf16, f32, false),
            DType::F32 => run!(f32, f32, false), DType::F64 => run!(f64, f64, true), _ => unreachable!(),
        }
        submitted += 1;
    }
    Ok(Exponential { value, submitted_kernels: submitted })
}

fn t8_coefficients(wide: bool) -> Vec<u64> {
    let round = |v: f64| if wide { v } else { v as f32 as f64 };
    let root = round(0.1330413469565007072504e+2); let x3 = round(2.0 / 3.0);
    [round(x3 * ((1.0 + root) / 88.0)), round(x3 * ((1.0 + root) / 352.0)), x3,
     round((-271.0 + 29.0 * root) / (315.0 * x3)), round((-11.0 + 11.0 * root) / (1260.0 * x3)),
     round((-99.0 + 11.0 * root) / (5040.0 * x3)), round((89.0 - root) / (5040.0 * x3)),
     round((857.0 - 58.0 * root) / 630.0)].into_iter().map(f64::to_bits).collect()
}

pub(super) const T12: [[f64; 4]; 4] = [
    [9.0198e-16, 0.46932117595418237389, -0.20099424927047284052, -0.04623946134063071740],
    [5.31597895759871264183, 1.19926790417132231573, 0.01179296240992997031, 0.01108844528519167989],
    [0.18188869982170434744, 0.05502798439925399070, 0.09351590770535414968, 0.00610700528898058230],
    [-2.0861320e-13, -0.13181061013830184015, -0.02027855540589259079, -0.00675951846863086359],
];

pub(super) const T18: [[f64; 5]; 5] = [
    [0.0, -1.00365581030144618291e-01, -8.02924648241156932449e-03, -8.92138498045729985177e-04, 0.0],
    [0.0, 3.97849749499645077844e-01, 1.36783778460411720168e+00, 4.98289622525382669416e-01, -6.37898194594723280150e-04],
    [-1.09676396052962061844e+01, 1.68015813878906206114e+00, 5.71779846478865511061e-02, -6.98210122488052056106e-03, 3.34975017086070470649e-05],
    [-9.04316832390810593223e-02, -6.76404519071381882256e-02, 6.75961301770459654925e-02, 2.95552570429315521194e-02, -1.39180257516060693404e-05],
    [0.0, 0.0, -9.23364619367118555360e-02, -1.69364939002081722752e-02, -1.40086798182036094347e-05],
];
