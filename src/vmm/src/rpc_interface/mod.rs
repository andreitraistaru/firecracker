// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use libc::O_NONBLOCK;

/// Wrapper for configuring the microVM boot source.
pub mod boot_source;
/// Pre-boot and post-boot/runtime controllers that handle RPC commands.
pub mod controllers;
/// Wrapper for configuring the block devices.
pub mod drive;
/// Wrapper over the microVM general information attached to the microVM.
pub mod instance_info;
/// Wrapper for configuring the logger.
pub mod logger;
/// Wrapper for configuring the memory and CPU of the microVM.
pub mod machine_config;
/// Wrapper for configuring the metrics.
pub mod metrics;
/// Wrapper for configuring the network devices attached to the microVM.
pub mod net;
/// Wrapper for configuring rate limiters.
pub mod rate_limiter;
/// Wrapper for configuring the vsock devices attached to the microVM.
pub mod vsock;

/// This enum represents the public interface of the VMM. Each action contains various
/// bits of information (ids, paths, etc.).
#[derive(PartialEq)]
pub enum VmmAction {
    /// Configure the boot source of the microVM using as input the `ConfigureBootSource`. This
    /// action can only be called before the microVM has booted.
    ConfigureBootSource(boot_source::BootSourceConfig),
    /// Configure the logger using as input the `LoggerConfig`. This action can only be called
    /// before the microVM has booted.
    ConfigureLogger(logger::LoggerConfig),
    /// Configure the metrics using as input the `MetricsConfig`. This action can only be called
    /// before the microVM has booted.
    ConfigureMetrics(metrics::MetricsConfig),
    /// Get the configuration of the microVM.
    GetVmConfiguration,
    /// Flush the metrics. This action can only be called after the logger has been configured.
    FlushMetrics,
    /// Add a new block device or update one that already exists using the `BlockDeviceConfig` as
    /// input. This action can only be called before the microVM has booted.
    InsertBlockDevice(drive::BlockDeviceConfig),
    /// Add a new network interface config or update one that already exists using the
    /// `NetworkInterfaceConfig` as input. This action can only be called before the microVM has
    /// booted.
    InsertNetworkDevice(net::NetworkInterfaceConfig),
    /// Set the vsock device or update the one that already exists using the
    /// `VsockDeviceConfig` as input. This action can only be called before the microVM has
    /// booted.
    SetVsockDevice(vsock::VsockDeviceConfig),
    /// Set the microVM configuration (memory & vcpu) using `VmConfig` as input. This
    /// action can only be called before the microVM has booted.
    SetVmConfiguration(machine_config::VmConfig),
    /// Launch the microVM. This action can only be called before the microVM has booted.
    StartMicroVm,
    /// Send CTRL+ALT+DEL to the microVM, using the i8042 keyboard function. If an AT-keyboard
    /// driver is listening on the guest end, this can be used to shut down the microVM gracefully.
    #[cfg(target_arch = "x86_64")]
    SendCtrlAltDel,
    /// Update the path of an existing block device. The data associated with this variant
    /// represents the `drive_id` and the `path_on_host`.
    UpdateBlockDevicePath(String, String),
    /// Update a network interface, after microVM start. Currently, the only updatable properties
    /// are the RX and TX rate limiters.
    UpdateNetworkInterface(net::NetworkInterfaceUpdateConfig),
}

/// Wrapper for all errors associated with VMM actions.
#[derive(Debug)]
pub enum VmmActionError {
    /// The action `ConfigureBootSource` failed because of bad user input.
    BootSource(boot_source::BootSourceConfigError),
    /// One of the actions `InsertBlockDevice` or `UpdateBlockDevicePath`
    /// failed because of bad user input.
    DriveConfig(drive::DriveError),
    /// Internal Vmm error.
    InternalVmm(crate::Error),
    /// The action `ConfigureLogger` failed because of bad user input.
    Logger(logger::LoggerConfigError),
    /// One of the actions `GetVmConfiguration` or `SetVmConfiguration` failed because of bad input.
    MachineConfig(machine_config::VmConfigError),
    /// The action `ConfigureMetrics` failed because of bad user input.
    Metrics(metrics::MetricsConfigError),
    /// The action `InsertNetworkDevice` failed because of bad user input.
    NetworkConfig(net::NetworkInterfaceError),
    /// The requested operation is not supported after starting the microVM.
    OperationNotSupportedPostBoot,
    /// The requested operation is not supported before starting the microVM.
    OperationNotSupportedPreBoot,
    /// The action `StartMicroVm` failed because of an internal error.
    StartMicrovm(super::builder::StartMicrovmError),
    /// The action `SetVsockDevice` failed because of bad user input.
    VsockConfig(vsock::VsockConfigError),
}

impl fmt::Display for VmmActionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::VmmActionError::*;

        write!(
            f,
            "{}",
            match self {
                BootSource(err) => err.to_string(),
                DriveConfig(err) => err.to_string(),
                InternalVmm(err) => format!("Internal Vmm error: {}", err),
                Logger(err) => err.to_string(),
                MachineConfig(err) => err.to_string(),
                Metrics(err) => err.to_string(),
                NetworkConfig(err) => err.to_string(),
                OperationNotSupportedPostBoot => {
                    "The requested operation is not supported after starting the microVM."
                        .to_string()
                }
                OperationNotSupportedPreBoot => {
                    "The requested operation is not supported before starting the microVM."
                        .to_string()
                }
                StartMicrovm(err) => err.to_string(),
                /// The action `SetVsockDevice` failed because of bad user input.
                VsockConfig(err) => err.to_string(),
            }
        )
    }
}

/// The enum represents the response sent by the VMM in case of success. The response is either
/// empty, when no data needs to be sent, or an internal VMM structure.
#[derive(Debug)]
pub enum VmmData {
    /// No data is sent on the channel.
    Empty,
    /// The microVM configuration represented by `VmConfig`.
    MachineConfiguration(machine_config::VmConfig),
}

type IoResult<T> = std::result::Result<T, std::io::Error>;

/// Structure `Writer` used for writing to a FIFO.
pub struct Writer {
    line_writer: Mutex<io::LineWriter<File>>,
}

impl Writer {
    /// Create and open a FIFO for writing to it.
    /// In order to not block the instance if nobody is consuming the message that is flushed to the
    /// two pipes, we are opening it with `O_NONBLOCK` flag. In this case, writing to a pipe will
    /// start failing when reaching 64K of unconsumed content. Simultaneously,
    /// the `missed_metrics_count` metric will get increased.
    pub fn new(fifo_path: PathBuf) -> IoResult<Writer> {
        OpenOptions::new()
            .custom_flags(O_NONBLOCK)
            .read(true)
            .write(true)
            .open(&fifo_path)
            .map(|t| Writer {
                line_writer: Mutex::new(io::LineWriter::new(t)),
            })
    }

    fn get_line_writer(&self) -> MutexGuard<io::LineWriter<File>> {
        match self.line_writer.lock() {
            Ok(guard) => guard,
            // If a thread panics while holding this lock, the writer within should still be usable.
            // (we might get an incomplete log line or something like that).
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

impl io::Write for Writer {
    fn write(&mut self, msg: &[u8]) -> IoResult<(usize)> {
        let mut line_writer = self.get_line_writer();
        line_writer.write_all(msg).map(|()| msg.len())
    }

    fn flush(&mut self) -> IoResult<()> {
        let mut line_writer = self.get_line_writer();
        line_writer.flush()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use utils::tempfile::TempFile;

    use super::*;

    #[test]
    fn test_log_writer() {
        let log_file_temp =
            TempFile::new().expect("Failed to create temporary output logging file.");
        let good_file = log_file_temp.as_path().to_path_buf();
        let res = Writer::new(good_file);
        assert!(res.is_ok());

        let mut fw = res.unwrap();
        let msg = String::from("some message");
        assert!(fw.write(&msg.as_bytes()).is_ok());
        assert!(fw.flush().is_ok());
    }
}
