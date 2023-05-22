use nalgebra::SMatrix;
use crate::DeviceCopy;

/// Implement DeviceCopy for SMatrix since nalgebra only supports specific versions of cust that
/// do not include this fork.
unsafe impl<T: DeviceCopy, const N: usize, const M: usize> DeviceCopy for SMatrix<T, N, M> {}
