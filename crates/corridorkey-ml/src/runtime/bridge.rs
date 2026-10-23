//! Zero-copy bridge between [`corridorkey_core::ComputeTarget`], CubeCL [`Handle`]s and Burn
//! [`Tensor`]s (`burn_cubecl::CubeBackend`).
//!
//! Both [`corridorkey_base`] kernels and Burn's [`CubeBackend`] execute on the same CubeCL
//! [`Client`] and allocate from the same device pool. Converting a `Handle` into a `Tensor<D>` or
//! extracting a contiguous `CubeTensor` from a `Tensor<D>` only exchanges metadata (`Handle` +
//! `Shape`), with zero device-to-device or host-to-device copies.

use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{CubeBackend, CubeDevice};
use burn_tensor::{DType, Device as BurnDevice, Shape, Tensor};
use corridorkey_base::core::CkFloat;
use corridorkey_base::core::buffer::{download, empty, upload};
use corridorkey_core::compute::open_device;
use corridorkey_core::{CkError, CkResult, ComputeTarget};
use cubecl::prelude::Client;
use cubecl::server::Handle;

/// Shared execution context for a [`ComputeTarget`]: owns the CubeCL [`Client`] (registered with
/// `corridorkey_core`'s exit-shutdown guard) and the matching [`BurnDevice`].
#[derive(Clone)]
pub struct MlContext {
    target: ComputeTarget,
    cube_device: CubeDevice,
    client: Client,
    burn_device: BurnDevice,
}

impl std::fmt::Debug for MlContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlContext")
            .field("target", &self.target)
            .field("runtime", &self.client.name())
            .finish()
    }
}

impl MlContext {
    /// Opens the default device for `target` and initializes both CubeCL and Burn handles on it.
    pub fn new(target: ComputeTarget) -> CkResult<Self> {
        let cube_device = target.device()?;
        let client = open_device(cube_device.clone());
        let burn_device = BurnDevice::from(cube_device.clone());
        Ok(Self {
            target,
            cube_device,
            client,
            burn_device,
        })
    }

    /// Context from `CORRIDORKEY_RUNTIME` or the first compiled-in runtime.
    pub fn from_env() -> CkResult<Self> {
        Self::new(ComputeTarget::from_env()?)
    }

    pub fn target(&self) -> ComputeTarget {
        self.target
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn cube_device(&self) -> &CubeDevice {
        &self.cube_device
    }

    pub fn burn_device(&self) -> &BurnDevice {
        &self.burn_device
    }

    /// Wraps an existing contiguous CubeCL [`Handle`] into a Burn [`Tensor<D>`] without copying.
    pub fn tensor_from_handle<const D: usize>(
        &self,
        handle: Handle,
        shape: [usize; D],
        dtype: DType,
    ) -> Tensor<D> {
        let cube = CubeTensor::new_contiguous(
            self.client.clone(),
            self.cube_device.clone(),
            Shape::from(shape),
            handle,
            dtype,
        );
        Tensor::<D>::from_primitive::<CubeBackend>(cube)
    }

    /// Allocates an uninitialized contiguous `Tensor<D>` of `F` on this device.
    pub fn empty_tensor<F: CkFloat, const D: usize>(&self, shape: [usize; D]) -> Tensor<D> {
        let len: usize = shape.iter().product();
        let handle = empty::<F>(&self.client, len);
        self.tensor_from_handle(handle, shape, elem_dtype::<F>())
    }

    /// Uploads a host slice `&[F]` into a contiguous `Tensor<D>` on this device.
    pub fn upload_tensor<F: CkFloat, const D: usize>(
        &self,
        host: &[F],
        shape: [usize; D],
    ) -> CkResult<Tensor<D>> {
        let expected: usize = shape.iter().product();
        if host.len() != expected {
            return Err(CkError::Shape(format!(
                "upload_tensor: slice len {} != shape {shape:?} ({expected})",
                host.len()
            )));
        }
        let handle = upload(&self.client, host);
        Ok(self.tensor_from_handle(handle, shape, elem_dtype::<F>()))
    }

    /// Downloads a `Tensor<D>` to a host `Vec<F>` (materializing a contiguous buffer on device if
    /// the tensor is a strided view).
    pub fn download_tensor<F: CkFloat, const D: usize>(&self, tensor: Tensor<D>) -> CkResult<Vec<F>> {
        let cube = into_contiguous_cube(tensor)?;
        Ok(download(&cube.client, cube.handle))
    }
}

/// Maps `F: CkFloat` to its Burn [`DType`].
pub fn elem_dtype<F: CkFloat>() -> DType {
    match size_of::<F>() {
        4 => DType::F32,
        8 => DType::F64,
        2 => DType::F16,
        other => unreachable!("unsupported CkFloat byte width {other}"),
    }
}

/// Extracts the underlying [`CubeTensor`] from a Burn [`Tensor<D>`] (zero-copy).
pub fn into_cube<const D: usize>(tensor: Tensor<D>) -> CkResult<CubeTensor> {
    tensor
        .try_into_primitive::<CubeBackend>()
        .map_err(|e| CkError::InvalidConfig(format!("expected CubeBackend tensor: {e:?}")))
}

/// Extracts a contiguous [`CubeTensor`] from a Burn [`Tensor<D>`]. No-op when already contiguous.
pub fn into_contiguous_cube<const D: usize>(tensor: Tensor<D>) -> CkResult<CubeTensor> {
    Ok(into_contiguous(into_cube(tensor)?))
}

/// Re-wraps a [`CubeTensor`] into a Burn [`Tensor<D>`] (zero-copy).
pub fn from_cube<const D: usize>(cube: CubeTensor) -> Tensor<D> {
    Tensor::<D>::from_primitive::<CubeBackend>(cube)
}

#[cfg(test)]
mod tests {
    use super::*;
    use corridorkey_core::ColorSpace;
    use corridorkey_core::LinearAdjust;

    #[test]
    fn shares_handles_between_base_kernels_and_burn_without_copy() {
        for target in ComputeTarget::available() {
            let ctx = MlContext::new(target).unwrap();
            let input: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
            let t = ctx.upload_tensor(&input, [16, 4]).unwrap();

            // Extract raw CubeCL handle, run a corridorkey-base kernel in place, and continue in Burn
            let cube = into_contiguous_cube(t).unwrap();
            corridorkey_base::color::convert::<f32>(
                &cube.client,
                &cube.handle,
                &cube.handle,
                64,
                ColorSpace::Srgb,
                ColorSpace::Linear,
                LinearAdjust::default(),
            );
            let t_linear = from_cube::<2>(cube);
            let scaled = t_linear * 2.0;
            let host = ctx.download_tensor::<f32, 2>(scaled).unwrap();
            assert_eq!(host.len(), 64);
            assert!((host[0] - 0.0).abs() < 1e-6);
            assert!(host[63] > 1.8);
        }
    }
}
