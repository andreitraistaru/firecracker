// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::sync::{Arc, Mutex};

use devices::virtio::{Vsock, VsockError, VsockUnixBackend, VsockUnixBackendError};

/// Errors associated with `NetworkInterfaceConfig`.
#[derive(Debug)]
pub enum VsockConfigError {
    /// Failed to create the backend for the vsock device.
    CreateVsockBackend(VsockUnixBackendError),
    /// Failed to create the vsock device.
    CreateVsockDevice(VsockError),
}

impl fmt::Display for VsockConfigError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::VsockConfigError::*;
        match *self {
            CreateVsockBackend(ref e) => {
                write!(f, "Cannot create backend for vsock device: {:?}", e)
            }
            CreateVsockDevice(ref e) => write!(f, "Cannot create vsock device: {:?}", e),
        }
    }
}

type Result<T> = std::result::Result<T, VsockConfigError>;

/// This struct represents the strongly typed equivalent of the json body
/// from vsock related requests.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VsockDeviceConfig {
    /// ID of the vsock device.
    pub vsock_id: String,
    /// A 32-bit Context Identifier (CID) used to identify the guest.
    pub guest_cid: u32,
    /// Path to local unix socket.
    pub uds_path: String,
}

pub(crate) struct VsockAndUnixPath {
    pub(crate) vsock: Arc<Mutex<Vsock<VsockUnixBackend>>>,
    uds_path: String,
}

/// A builder of Vsock with Unix backend from 'VsockDeviceConfig'.
#[derive(Default)]
pub struct VsockStore {
    pub(crate) inner: Option<VsockAndUnixPath>,
}

impl VsockStore {
    /// Creates an empty Vsock with Unix backend Store.
    pub fn new() -> Self {
        Self { inner: None }
    }

    /// Inserts a Unix backend Vsock in the store.
    /// If an entry already exists, it will overwrite it.
    pub fn insert(&mut self, cfg: VsockDeviceConfig) -> Result<()> {
        // Make sure to drop the old one and remove the socket before creating a new one.
        if let Some(existing) = self.inner.take() {
            let _ = std::fs::remove_file(existing.uds_path);
        }
        self.inner = Some(VsockAndUnixPath {
            uds_path: cfg.uds_path.clone(),
            vsock: Arc::new(Mutex::new(Self::create_unixsock_vsock(cfg)?)),
        });
        Ok(())
    }

    /// Provides a reference to the Vsock if present.
    pub fn get(&self) -> Option<&Arc<Mutex<Vsock<VsockUnixBackend>>>> {
        self.inner.as_ref().map(|pair| &pair.vsock)
    }

    /// Creates a Vsock device from a VsockDeviceConfig.
    pub fn create_unixsock_vsock(cfg: VsockDeviceConfig) -> Result<Vsock<VsockUnixBackend>> {
        let backend = VsockUnixBackend::new(u64::from(cfg.guest_cid), cfg.uds_path)
            .map_err(VsockConfigError::CreateVsockBackend)?;

        Ok(Vsock::new(cfg.vsock_id, u64::from(cfg.guest_cid), backend)
            .map_err(VsockConfigError::CreateVsockDevice)?)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builder::tests::TempFilePath;
    use utils::tempfile::TempFile;

    pub(crate) fn default_config(tmp_sock_file: &TempFilePath) -> VsockDeviceConfig {
        let vsock_dev_id = "vsock_1";
        VsockDeviceConfig {
            vsock_id: vsock_dev_id.to_string(),
            guest_cid: 3,
            uds_path: tmp_sock_file.path().clone(),
        }
    }

    #[test]
    fn test_vsock_create() {
        let tmp_sock_file = TempFilePath::new(TempFile::new().unwrap());
        let vsock_config = default_config(&tmp_sock_file);
        VsockStore::create_unixsock_vsock(vsock_config).unwrap();
    }

    #[test]
    fn test_vsock_insert() {
        let mut store = VsockStore::new();
        let tmp_sock_file = TempFilePath::new(TempFile::new().unwrap());
        let mut vsock_config = default_config(&tmp_sock_file);

        store.insert(vsock_config.clone()).unwrap();
        let vsock = store.get().unwrap();
        assert_eq!(vsock.lock().unwrap().id(), &vsock_config.vsock_id);

        let new_id = "another".to_string();
        vsock_config.vsock_id = new_id.clone();
        store.insert(vsock_config).unwrap();
        let vsock = store.get().unwrap();
        assert_eq!(vsock.lock().unwrap().id(), &new_id);
    }

    #[test]
    fn test_error_messages() {
        use super::VsockConfigError::*;
        use std::io;
        let err = CreateVsockBackend(devices::virtio::VsockUnixBackendError::EpollAdd(
            io::Error::from_raw_os_error(0),
        ));
        assert_eq!(
            format!("{}", err),
            format!(
                "Cannot create backend for vsock device: {:?}",
                devices::virtio::VsockUnixBackendError::EpollAdd(io::Error::from_raw_os_error(0))
            )
        );

        let err = CreateVsockDevice(devices::virtio::VsockError::EventFd(
            io::Error::from_raw_os_error(0),
        ));
        assert_eq!(
            format!("{}", err),
            format!(
                "Cannot create vsock device: {:?}",
                devices::virtio::VsockError::EventFd(io::Error::from_raw_os_error(0))
            )
        );
    }
}
