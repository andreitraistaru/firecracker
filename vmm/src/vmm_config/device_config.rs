// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(warnings)]

use vmm_config::balloon::*;
use vmm_config::drive::*;
use vmm_config::net::*;
use vmm_config::vsock::*;

/// A data structure that encapsulates the device configurations
/// held in the Vmm.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct DeviceConfigs {
    /// The configurations for balloon devices.
    pub balloon: BalloonConfigs,
    /// The configurations for block devices.
    pub block: BlockDeviceConfigs,
    /// The configurations for network interface devices.
    pub network_interface: NetworkInterfaceConfigs,
    /// The configurations for vsock devices.
    pub vsock: Option<VsockDeviceConfig>,
}

impl DeviceConfigs {
    /// Construct a `DeviceConfigs` structure from its constituent parts.
    pub fn new(
        balloon: BalloonConfigs,
        block: BlockDeviceConfigs,
        network_interface: NetworkInterfaceConfigs,
        vsock: Option<VsockDeviceConfig>,
    ) -> DeviceConfigs {
        DeviceConfigs {
            balloon,
            block,
            network_interface,
            vsock,
        }
    }
}
