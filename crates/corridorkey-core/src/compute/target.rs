//! Running work on a [`ComputeTarget`].
//!
//! Library code takes a [`Client`] (cubecl 0.11 resolves the runtime when the client is made, not
//! through a generic); the CLI and the wasm bindings pick a target once and get its client,
//! or run a [`ComputeTask`] on it.
//!
//! **Devices are shut down at exit.** CubeCL runs each device on a runner thread that nothing joins,
//! so the device is never destroyed and the GPU driver's own worker threads (NVIDIA's shader
//! compiler) were still running while the process exited: an intermittent crash or hang after all
//! the work was done (SIGSEGV in `libnvidia-glvkspirv`, "corrupted double-linked list" in
//! `libnvidia-gpucomp`, with the main thread inside `exit()`). Every device a client is made for is
//! recorded, and an exit handler, registered after each device is opened (so it runs before the
//! handlers of the libraries and layers loaded until then), shuts them down: [`shutdown_devices`].

use std::sync::Mutex;

use cubecl::device::{AmdDevice, CpuDevice, CudaDevice, WgpuDevice};
use cubecl::device_handle::DeviceHandle;
use cubecl::prelude::*;
use cubecl::Device;

use crate::device::{ComputeError, ComputeTarget};

/// Work to run on whichever runtime a [`ComputeTarget`] selects.
pub trait ComputeTask {
    type Output;
    fn run(self, client: Client) -> Self::Output;
}

impl ComputeTarget {
    /// The default device of this runtime, if it is compiled in.
    pub fn device(self) -> Result<Device, ComputeError> {
        Ok(match self.checked()? {
            Self::Wgpu => Device::Wgpu(WgpuDevice::default()),
            Self::Cpu => Device::Cpu(CpuDevice),
            Self::Cuda => Device::Cuda(CudaDevice::default()),
            Self::Hip => Device::Hip(AmdDevice::default()),
        })
    }

    /// A client for this runtime's default device (recorded for [`shutdown_devices`]).
    pub fn client(self) -> Result<Client, ComputeError> {
        Ok(open_device(self.device()?))
    }

    /// Runs `task` with a client for this runtime's default device.
    pub fn run<T: ComputeTask>(self, task: T) -> Result<T::Output, ComputeError> {
        Ok(task.run(self.client()?))
    }
}

/// Devices a client was made for, shut down at exit.
static OPENED: Mutex<Vec<Device>> = Mutex::new(Vec::new());

// No C exit handlers in the browser: the page owns the WebGPU device
#[cfg(not(target_arch = "wasm32"))]
unsafe extern "C" {
    /// The C library's exit-handler registration (handlers run in reverse order of registration).
    fn atexit(handler: extern "C" fn()) -> std::ffi::c_int;
}

#[cfg(not(target_arch = "wasm32"))]
extern "C" fn shutdown_at_exit() {
    shutdown_devices();
}

/// Records `device`, newly opened, and registers [`shutdown_devices`] to run at exit. Called after
/// the client exists: exit handlers run in reverse order of registration, and opening a device
/// loads Vulkan layers that register their own (the validation layer destroys its device records
/// in one), so ours must come later to run first. Registered again for each newly opened device,
/// since a device reopened after a shutdown may load those layers again; running it more than once
/// is harmless (the list is empty after the first run).
fn opened(device: &Device) {
    let mut devices = OPENED.lock().unwrap_or_else(|e| e.into_inner());
    if devices.contains(device) {
        return;
    }
    devices.push(device.clone());
    drop(devices);
    // SAFETY: registers a plain `extern "C"` function; failure (no slot left) only means the
    // device is not shut down at exit, as before
    #[cfg(not(target_arch = "wasm32"))]
    unsafe {
        atexit(shutdown_at_exit);
    }
}

/// The client of `device`, recorded so that [`shutdown_devices`] shuts it down at exit. Use this
/// instead of `Device::client` wherever a specific device is opened (e.g. one GPU of several).
pub fn open_device(device: Device) -> Client {
    let client = device.client();
    opened(&device);
    client
}

/// Shuts down every device a client was made for: their runner threads stop (queued work runs
/// first) and the devices are destroyed, so the GPU driver's threads end before the process does.
/// Runs at exit by itself; call it earlier to release the GPUs. Clients still alive keep their
/// runner going (CubeCL waits up to 30 s for it, then leaves it), so drop them first. A later
/// [`ComputeTarget::client`] opens the device again.
pub fn shutdown_devices() {
    let devices = std::mem::take(&mut *OPENED.lock().unwrap_or_else(|e| e.into_inner()));
    for device in devices {
        // The runner is registered under the runtime's own id, not `Device`'s runtime-stamped one
        DeviceHandle::<()>::shutdown(cubecl::RuntimeId::strip(device.to_id()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct RuntimeName;
    impl ComputeTask for RuntimeName {
        type Output = String;
        fn run(self, client: Client) -> String {
            client.name().to_string()
        }
    }

    /// Threads of this process (Linux).
    fn threads() -> usize {
        std::fs::read_dir("/proc/self/task").map(|d| d.count()).unwrap_or(0)
    }

    /// Shutting the devices down stops their runner (and driver) threads; a later client opens the
    /// device again.
    #[test]
    #[cfg(target_os = "linux")]
    fn shutdown_stops_the_device_threads() {
        let Some(target) = ComputeTarget::available().into_iter().find(|t| *t != ComputeTarget::Cpu) else { return };
        let client = target.client().unwrap();
        let data: Vec<u8> = (0..64u8).collect();
        let handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(data.clone()));
        assert_eq!(client.read_one_unchecked(handle).to_vec(), data);
        drop(client);
        let open = threads();
        shutdown_devices();
        let closed = threads();
        assert!(closed < open, "{open} threads with the device open, {closed} after shutdown");
        let again = target.client().unwrap();
        assert!(!again.name().is_empty());
    }

    #[test]
    fn runs_on_available_targets_and_refuses_missing_ones() {
        for target in ComputeTarget::available() {
            assert!(!target.run(RuntimeName).unwrap().is_empty());
        }
        for target in ComputeTarget::ALL.into_iter().filter(|t| !t.is_available()) {
            assert!(matches!(target.run(RuntimeName), Err(ComputeError::Unavailable(..))));
        }
    }
}
