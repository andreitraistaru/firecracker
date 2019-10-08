// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std;
use std::fmt::{Display, Formatter, Result};

use device_manager;
use devices;
use devices::virtio::{MmioDeviceStateError, SpecificVirtioDeviceStateError};
use kernel::loader as kernel_loader;
use memory_model::GuestMemoryError;
use seccomp;
#[cfg(target_arch = "x86_64")]
use snapshot;
use vstate;

/// The microvm state. When Firecracker starts, the instance state is `Uninitialized`.
/// Configuring any microVM resources (logger not included as that's a Firecracker
/// process resource and not a VM resource) will change the state from `Uninitialized`
/// to `Configuring`.
/// Once `start_microvm()` method is called, the state goes to `Starting`. The state
/// is then changed to `Running` if the start function succeeds.
/// Loading a microVM from a snapshot is only permitted on an `Uninitialized` microVM.
/// In such a case the flow is `Uninitialized` -> `Resuming` -> `Running`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum InstanceState {
    /// Microvm is not initialized.
    Uninitialized,
    /// Microvm is being configured.
    Configuring,
    /// Microvm is starting.
    Starting,
    /// Microvm is resuming.
    Resuming,
    /// Microvm is running.
    Running,
}

/// The strongly typed that contains general information about the microVM.
#[derive(Serialize)]
pub struct InstanceInfo {
    /// The ID of the microVM.
    pub id: String,
    /// The state of the microVM.
    pub state: InstanceState,
    /// The version of the VMM that runs the microVM.
    pub vmm_version: String,
}

/// Errors associated with failed state validations.
#[derive(Debug)]
pub enum StateError {
    /// This microVM has been configured therefore cannot be loaded from snapshot.
    MicroVMAlreadyConfigured,
    /// The start/resume command was issued more than once.
    MicroVMAlreadyRunning,
    /// The microVM hasn't been started.
    MicroVMIsNotRunning,
    /// vCPUs are in an invalid state.
    VcpusInvalidState,
}

impl Display for StateError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        use self::StateError::*;
        match *self {
            MicroVMAlreadyConfigured => write!(
                f,
                "Microvm has been configured therefore cannot be loaded from snapshot."
            ),
            MicroVMAlreadyRunning => write!(f, "Microvm is already running."),
            MicroVMIsNotRunning => write!(f, "Microvm is not running."),
            VcpusInvalidState => write!(f, "vCPUs are in an invalid state."),
        }
    }
}

/// Errors associated with stopping the microVM threads.
#[derive(Debug)]
pub enum KillVcpusError {
    /// Sanity checks failed.
    MicroVMInvalidState(StateError),
    /// Failed to signal vcpu.
    SignalVcpu(vstate::Error),
}

impl Display for KillVcpusError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        use self::KillVcpusError::*;
        match *self {
            MicroVMInvalidState(ref e) => write!(f, "{}", e),
            SignalVcpu(ref err) => write!(f, "Failed to signal vCPU: {:?}", err),
        }
    }
}

/// Errors associated with pausing the microVM.
#[derive(Debug)]
pub enum PauseMicrovmError {
    /// Invalid snapshot header.
    #[cfg(target_arch = "x86_64")]
    InvalidHeader(snapshot::Error),
    /// Sanity checks failed.
    MicroVMInvalidState(StateError),
    /// Missing snapshot file.
    #[cfg(target_arch = "x86_64")]
    MissingSnapshot,
    /// Failed to save MMIO device state.
    SaveMmioDeviceState(MmioDeviceStateError),
    /// Failed to save vCPU state.
    SaveVcpuState(Option<vstate::Error>),
    /// Failed to save VM state.
    SaveVmState(vstate::Error),
    /// Failed to serialize microVM state.
    #[cfg(target_arch = "x86_64")]
    SerializeMicrovmState(snapshot::Error),
    /// Failed to send event.
    SignalVcpu(vstate::Error),
    /// Cannot create snapshot backing file.
    #[cfg(target_arch = "x86_64")]
    SnapshotBackingFile(snapshot::Error),
    /// Failed to stop vcpus.
    StopVcpus(KillVcpusError),
    /// Failed to sync memory to snapshot.
    SyncMemory(GuestMemoryError),
    /// vCPU pause failed.
    VcpuPause,
}

impl Display for PauseMicrovmError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        use self::PauseMicrovmError::*;
        match *self {
            #[cfg(target_arch = "x86_64")]
            InvalidHeader(ref e) => write!(f, "Failed to sync snapshot: {}", e),
            MicroVMInvalidState(ref e) => write!(f, "{}", e),
            MissingSnapshot => write!(f, "Missing snapshot file"),
            SaveMmioDeviceState(ref e) => write!(f, "Cannot save a mmio device. {:?}", e),
            SaveVcpuState(ref e) => match e {
                None => write!(f, "Failed to save vCPU state."),
                Some(err) => write!(f, "Failed to save vCPU state: {:?}", err),
            },
            SaveVmState(ref e) => write!(f, "Failed to save VM state: {:?}", e),
            #[cfg(target_arch = "x86_64")]
            SerializeMicrovmState(ref e) => write!(f, "Failed to serialize VM state: {}", e),
            #[cfg(target_arch = "x86_64")]
            SnapshotBackingFile(ref err) => {
                write!(f, "Cannot create snapshot backing file: {}", err)
            }
            SignalVcpu(ref e) => write!(f, "Failed to signal vCPU: {:?}", e),
            StopVcpus(ref e) => write!(f, "Failed to stop vcpus: {}", e),
            SyncMemory(ref e) => write!(f, "Failed to sync memory to snapshot: {:?}", e),
            VcpuPause => write!(f, "vCPUs pause failed."),
        }
    }
}

/// Errors associated with resuming the microVM.
#[derive(Debug)]
pub enum ResumeMicrovmError {
    /// Sanity checks failed.
    MicroVMInvalidState(StateError),
    /// VM state is missing from the snapshot file.
    #[cfg(target_arch = "x86_64")]
    MissingVmState,
    /// Cannot open the snapshot image file.
    #[cfg(target_arch = "x86_64")]
    OpenSnapshotFile(snapshot::Error),
    /// Failed to restore virtio device state.
    RestoreVirtioDeviceState(SpecificVirtioDeviceStateError, String, String),
    /// Failed to reregister MMIO device.
    ReregisterMmioDevice(device_manager::mmio::Error, String, String),
    /// Failed to restore vCPU state.
    RestoreVcpuState,
    /// Failed to restore VM state.
    RestoreVmState(vstate::Error),
    /// Failed to send event.
    SignalVcpu(vstate::Error),
    /// Setting up microVM for resume failed.
    StartMicroVm(StartMicrovmError),
    /// vCPU resume failed.
    VcpuResume,
}

impl Display for ResumeMicrovmError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        use self::ResumeMicrovmError::*;
        match *self {
            MicroVMInvalidState(ref e) => write!(f, "{}", e),
            #[cfg(target_arch = "x86_64")]
            MissingVmState => write!(f, "Missing VM state"),
            #[cfg(target_arch = "x86_64")]
            OpenSnapshotFile(ref err) => {
                write!(f, "Cannot open the snapshot image file: {:?}", err)
            }
            RestoreVirtioDeviceState(ref err, ref dev_type, ref dev_id) => write!(
                f,
                "Failed to restore the MMIO state for {} device {}: {:?}",
                dev_type, dev_id, err
            ),
            ReregisterMmioDevice(ref err, ref dev_type, ref dev_id) => write!(
                f,
                "Failed to reregister {} device {}: {:?}",
                dev_type, dev_id, err
            ),
            RestoreVcpuState => write!(f, "Failed to restore vCPU state."),
            RestoreVmState(ref e) => write!(f, "Failed to restore VM state: {:?}", e),
            SignalVcpu(ref err) => write!(f, "Failed to signal vCPU: {:?}", err),
            StartMicroVm(ref err) => write!(f, "Failed resume microVM: {}", err),
            VcpuResume => write!(f, "vCPUs resume failed."),
        }
    }
}

/// Errors associated with starting the microVM.
#[derive(Debug)]
pub enum StartMicrovmError {
    /// This error is thrown by the minimal boot loader implementation.
    /// It is related to a faulty memory configuration.
    ConfigureSystem(arch::Error),
    /// Cannot configure the VM.
    ConfigureVm(vstate::Error),
    /// Unable to seek the block device backing file due to invalid permissions or
    /// the file was deleted/corrupted.
    CreateBlockDevice(devices::virtio::block::BlockError),
    /// Splits this at some point.
    CreateBalloon(devices::Error),
    /// Split this at some point.
    /// Internal errors are due to resource exhaustion.
    /// Users errors are due to invalid permissions.
    CreateNetDevice(devices::virtio::Error),
    /// Failed to create a `RateLimiter` object.
    CreateRateLimiter(std::io::Error),
    /// Failed to create the vsock device.
    CreateVsockDevice,
    /// The device manager was not configured.
    DeviceManager,
    /// Cannot read from an Event file descriptor.
    EventFd,
    /// Memory regions are overlapping or mmap fails.
    GuestMemory(GuestMemoryError),
    /// The kernel command line is invalid.
    KernelCmdline(String),
    /// Cannot load kernel due to invalid memory configuration or invalid kernel image.
    KernelLoader(kernel_loader::Error),
    /// Cannot add devices to the Legacy I/O Bus.
    LegacyIOBus(device_manager::legacy::Error),
    /// Cannot load command line string.
    LoadCommandline(kernel::cmdline::Error),
    /// Sanity checks failed.
    MicroVMInvalidState(StateError),
    /// Cannot start the VM because the kernel was not configured.
    MissingKernelConfig,
    /// The net device configuration is missing the tap device.
    NetDeviceNotConfigured,
    /// Cannot open the block device backing file.
    OpenBlockDevice(std::io::Error),
    /// Cannot initialize a MMIO Balloon Device or add a device to the MMIO Bus.
    RegisterBalloonDevice(device_manager::mmio::Error),
    /// Cannot initialize a MMIO Block Device or add a device to the MMIO Bus.
    RegisterBlockDevice(device_manager::mmio::Error),
    /// Cannot add event to Epoll.
    RegisterEvent,
    /// Cannot add a device to the MMIO Bus.
    RegisterMMIODevice(device_manager::mmio::Error),
    /// Cannot initialize a MMIO Network Device or add a device to the MMIO Bus.
    RegisterNetDevice(device_manager::mmio::Error),
    /// Cannot initialize a MMIO Vsock Device or add a device to the MMIO Bus.
    RegisterVsockDevice(device_manager::mmio::Error),
    /// Cannot build seccomp filters.
    SeccompFilters(seccomp::Error),
    /// Failed to signal vCPU.
    SignalVcpu(vstate::Error),
    /// Cannot create a new vCPU file descriptor.
    Vcpu(vstate::Error),
    /// vCPU configuration failed.
    VcpuConfigure(vstate::Error),
    /// vCPUs have already been created.
    VcpusAlreadyPresent,
    /// vCPUs were not configured.
    VcpusNotConfigured,
    /// Cannot spawn a new vCPU thread.
    VcpuSpawn(vstate::Error),
    /// Cannot set mode for terminal.
    StdinHandle(std::io::Error),
}

impl std::convert::From<StateError> for StartMicrovmError {
    fn from(e: StateError) -> Self {
        StartMicrovmError::MicroVMInvalidState(e)
    }
}

/// It's convenient to automatically convert `kernel::cmdline::Error`s
/// to `StartMicrovmError`s.
impl std::convert::From<kernel::cmdline::Error> for StartMicrovmError {
    fn from(e: kernel::cmdline::Error) -> StartMicrovmError {
        StartMicrovmError::KernelCmdline(e.to_string())
    }
}

impl Display for StartMicrovmError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        use self::StartMicrovmError::*;
        match *self {
            ConfigureSystem(ref err) => {
                let mut err_msg = format!("{:?}", err);
                err_msg = err_msg.replace("\"", "");

                write!(f, "Faulty memory configuration. {}", err_msg)
            }
            ConfigureVm(ref err) => {
                let mut err_msg = format!("{:?}", err);
                err_msg = err_msg.replace("\"", "");

                write!(f, "Cannot configure virtual machine. {}", err_msg)
            }
            CreateBlockDevice(ref err) => write!(
                f,
                "Unable to seek the block device backing file due to invalid permissions or \
                 the file was deleted/corrupted. Error number: {:?}",
                err
            ),
            CreateRateLimiter(ref err) => write!(f, "Cannot create RateLimiter. {}", err),
            CreateVsockDevice => write!(f, "Cannot create vsock device."),
            CreateBalloon(ref err) => {
                let mut err_msg = format!("{:?}", err);
                err_msg = err_msg.replace("\"", "");

                write!(f, "Cannot create balloon device. {}", err_msg)
            }
            CreateNetDevice(ref err) => {
                let mut err_msg = format!("{:?}", err);
                err_msg = err_msg.replace("\"", "");

                write!(f, "Cannot create network device. {}", err_msg)
            }
            DeviceManager => write!(f, "The device manager was not configured."),
            EventFd => write!(f, "Cannot read from an Event file descriptor."),
            GuestMemory(ref err) => {
                // Remove imbricated quotes from error message.
                let mut err_msg = format!("{:?}", err);
                err_msg = err_msg.replace("\"", "");
                write!(f, "Invalid Memory Configuration: {}", err_msg)
            }
            KernelCmdline(ref err) => write!(f, "Invalid kernel command line. {}", err),
            KernelLoader(ref err) => {
                let mut err_msg = format!("{}", err);
                err_msg = err_msg.replace("\"", "");
                write!(
                    f,
                    "Cannot load kernel due to invalid memory configuration or invalid kernel \
                     image. {}",
                    err_msg
                )
            }
            LegacyIOBus(ref err) => {
                let mut err_msg = format!("{}", err);
                err_msg = err_msg.replace("\"", "");

                write!(f, "Cannot add devices to the legacy I/O Bus. {}", err_msg)
            }
            LoadCommandline(ref err) => {
                let mut err_msg = format!("{}", err);
                err_msg = err_msg.replace("\"", "");
                write!(f, "Cannot load command line string. {}", err_msg)
            }
            MicroVMInvalidState(ref e) => write!(f, "{}", e),
            MissingKernelConfig => write!(f, "Cannot start microvm without kernel configuration."),
            NetDeviceNotConfigured => {
                write!(f, "The net device configuration is missing the tap device.")
            }
            OpenBlockDevice(ref err) => {
                let mut err_msg = format!("{}", err);
                err_msg = err_msg.replace("\"", "");

                write!(f, "Cannot open the block device backing file. {}", err_msg)
            }
            RegisterBlockDevice(ref err) => write!(
                f,
                "Cannot initialize a MMIO Block Device or add a device to the MMIO Bus. {}",
                err
            ),
            RegisterBalloonDevice(ref err) => {
                let mut err_msg = format!("{}", err);
                err_msg = err_msg.replace("\"", "");
                write!(
                    f,
                    "Cannot initialize a MMIO Balloon Device or add a device to the MMIO Bus. {}",
                    err_msg
                )
            }
            RegisterEvent => write!(f, "Cannot add event to Epoll."),
            RegisterMMIODevice(ref err) => {
                let mut err_msg = format!("{}", err);
                err_msg = err_msg.replace("\"", "");

                write!(f, "Cannot add a device to the MMIO Bus. {}", err_msg)
            }
            RegisterNetDevice(ref err) => {
                let mut err_msg = format!("{}", err);
                err_msg = err_msg.replace("\"", "");

                write!(
                    f,
                    "Cannot initialize a MMIO Network Device or add a device to the MMIO Bus. {}",
                    err_msg
                )
            }
            RegisterVsockDevice(ref err) => {
                let mut err_msg = format!("{}", err);
                err_msg = err_msg.replace("\"", "");

                write!(
                    f,
                    "Cannot initialize a MMIO Vsock Device or add a device to the MMIO Bus. {}",
                    err_msg
                )
            }
            SeccompFilters(ref err) => {
                let mut err_msg = format!("{:?}", err);
                err_msg = err_msg.replace("\"", "");

                write!(f, "Cannot build seccomp filters. {}", err_msg)
            }
            SignalVcpu(ref err) => write!(f, "Failed to signal vCPU. {:?}", err),
            Vcpu(ref err) => {
                let mut err_msg = format!("{:?}", err);
                err_msg = err_msg.replace("\"", "");

                write!(f, "Cannot create a new vCPU. {}", err_msg)
            }
            VcpuConfigure(ref err) => {
                let mut err_msg = format!("{:?}", err);
                err_msg = err_msg.replace("\"", "");

                write!(f, "vCPU configuration failed. {}", err_msg)
            }
            VcpusAlreadyPresent => write!(f, "vCPUs have already been created."),
            VcpusNotConfigured => write!(f, "vCPUs were not configured."),
            VcpuSpawn(ref err) => {
                let mut err_msg = format!("{:?}", err);
                err_msg = err_msg.replace("\"", "");

                write!(f, "Cannot spawn vCPU thread. {}", err_msg)
            }
            StdinHandle(ref err) => write!(f, "Failed to set mode for terminal: {}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kill_vcpus_error_messages() {
        use self::KillVcpusError::*;
        assert_eq!(
            format!("{}", MicroVMInvalidState(StateError::MicroVMIsNotRunning)),
            format!("{}", StateError::MicroVMIsNotRunning)
        );
        assert_eq!(
            format!("{}", SignalVcpu(vstate::Error::NotEnoughMemorySlots)),
            format!(
                "Failed to signal vCPU: {:?}",
                vstate::Error::NotEnoughMemorySlots
            )
        );
    }

    #[test]
    fn test_pause_microvm_error_messages() {
        use self::PauseMicrovmError::*;
        assert_eq!(
            format!("{}", MicroVMInvalidState(StateError::VcpusInvalidState)),
            format!("{}", StateError::VcpusInvalidState)
        );
        assert_eq!(
            format!(
                "{}",
                SaveMmioDeviceState(MmioDeviceStateError::SaveSpecificVirtioDevice(
                    SpecificVirtioDeviceStateError::InvalidDeviceType
                ))
            ),
            "Cannot save a mmio device. SaveSpecificVirtioDevice(InvalidDeviceType)"
        );

        assert_eq!(
            format!("{}", SaveVcpuState(None)),
            "Failed to save vCPU state.".to_string()
        );
        assert_eq!(
            format!(
                "{}",
                SaveVcpuState(Some(vstate::Error::VcpuCountNotInitialized))
            ),
            "Failed to save vCPU state: VcpuCountNotInitialized".to_string()
        );
        assert_eq!(
            format!("{}", SaveVmState(vstate::Error::NotEnoughMemorySlots)),
            format!(
                "Failed to save VM state: {:?}",
                vstate::Error::NotEnoughMemorySlots
            )
        );
        assert_eq!(
            format!("{}", SignalVcpu(vstate::Error::NotEnoughMemorySlots)),
            format!(
                "Failed to signal vCPU: {:?}",
                vstate::Error::NotEnoughMemorySlots
            )
        );

        assert_eq!(
            format!(
                "{}",
                StopVcpus(KillVcpusError::MicroVMInvalidState(
                    StateError::VcpusInvalidState
                ))
            ),
            format!(
                "Failed to stop vcpus: {}",
                KillVcpusError::MicroVMInvalidState(StateError::VcpusInvalidState)
            )
        );
        assert_eq!(
            format!("{}", SyncMemory(GuestMemoryError::MemoryRegionOverlap)),
            format!(
                "Failed to sync memory to snapshot: {:?}",
                GuestMemoryError::MemoryRegionOverlap
            )
        );
        assert_eq!(format!("{}", VcpuPause), "vCPUs pause failed.".to_string());
    }

    #[test]
    fn test_resume_microvm_error_messages() {
        use self::ResumeMicrovmError::*;
        assert_eq!(
            format!(
                "{}",
                MicroVMInvalidState(StateError::MicroVMAlreadyConfigured)
            ),
            format!("{}", StateError::MicroVMAlreadyConfigured)
        );
        assert_eq!(
            format!("{}", MicroVMInvalidState(StateError::MicroVMAlreadyRunning)),
            format!("{}", StateError::MicroVMAlreadyRunning)
        );
        #[cfg(target_arch = "x86_64")]
        assert_eq!(format!("{}", MissingVmState), "Missing VM state");
        #[cfg(target_arch = "x86_64")]
        assert_eq!(
            format!("{}", OpenSnapshotFile(snapshot::Error::MissingMemFile)),
            "Cannot open the snapshot image file: MissingMemFile"
        );
        assert_eq!(
            format!(
                "{}",
                RestoreVirtioDeviceState(
                    SpecificVirtioDeviceStateError::InvalidDeviceType,
                    "block".to_string(),
                    "root".to_string()
                )
            ),
            "Failed to restore the MMIO state for block device root: InvalidDeviceType"
        );
        assert_eq!(
            format!(
                "{}",
                ReregisterMmioDevice(
                    device_manager::mmio::Error::ActivationFailed,
                    "block".to_string(),
                    "root".to_string()
                )
            ),
            "Failed to reregister block device root: ActivationFailed"
        );
        assert_eq!(
            format!("{}", RestoreVcpuState),
            "Failed to restore vCPU state.".to_string()
        );
        assert_eq!(
            format!("{}", RestoreVmState(vstate::Error::NotEnoughMemorySlots)),
            format!(
                "Failed to restore VM state: {:?}",
                vstate::Error::NotEnoughMemorySlots
            )
        );
        assert_eq!(
            format!("{}", SignalVcpu(vstate::Error::NotEnoughMemorySlots)),
            format!(
                "Failed to signal vCPU: {:?}",
                vstate::Error::NotEnoughMemorySlots
            )
        );
        assert_eq!(
            format!("{}", StartMicroVm(StartMicrovmError::EventFd)),
            format!("Failed resume microVM: {}", StartMicrovmError::EventFd)
        );
        assert_eq!(
            format!("{}", VcpuResume),
            "vCPUs resume failed.".to_string()
        );
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn test_start_microvm_error_messages() {
        use self::StartMicrovmError::*;
        assert_eq!(
            format!("{}", ConfigureVm(vstate::Error::NotEnoughMemorySlots)),
            format!(
                "Cannot configure virtual machine. {:?}",
                vstate::Error::NotEnoughMemorySlots
            )
        );
        assert_eq!(
            format!(
                "{}",
                CreateBlockDevice(devices::virtio::block::BlockError::OpenFile(
                    std::io::Error::from_raw_os_error(0)
                ))
            ),
            format!(
                "Unable to seek the block device backing file due to invalid permissions or \
                 the file was deleted/corrupted. Error number: {:?}",
                devices::virtio::block::BlockError::OpenFile(std::io::Error::from_raw_os_error(0))
            )
        );
        assert_eq!(
            format!("{}", CreateBalloon(devices::Error::MalformedDescriptor)),
            format!(
                "Cannot create balloon device. {:?}",
                devices::Error::MalformedDescriptor
            )
        );
        assert_eq!(
            format!(
                "{}",
                CreateRateLimiter(std::io::Error::from_raw_os_error(0))
            ),
            format!(
                "Cannot create RateLimiter. {}",
                std::io::Error::from_raw_os_error(0)
            )
        );
        assert_eq!(
            format!("{}", DeviceManager),
            "The device manager was not configured.".to_string()
        );
        assert_eq!(
            format!("{}", EventFd),
            "Cannot read from an Event file descriptor.".to_string()
        );
        assert_eq!(
            format!("{}", KernelCmdline(".".to_string())),
            "Invalid kernel command line. .".to_string()
        );
        assert_eq!(
            format!(
                "{}",
                KernelLoader(kernel_loader::Error::BigEndianElfOnLittle)
            ),
            format!(
                "Cannot load kernel due to invalid memory configuration or invalid kernel \
                 image. {}",
                kernel_loader::Error::BigEndianElfOnLittle
            )
        );
        assert_eq!(
            format!(
                "{}",
                LegacyIOBus(device_manager::legacy::Error::EventFd(
                    std::io::Error::from_raw_os_error(0)
                ))
            ),
            format!(
                "Cannot add devices to the legacy I/O Bus. {}",
                device_manager::legacy::Error::EventFd(std::io::Error::from_raw_os_error(0))
            )
        );
        assert_eq!(
            format!(
                "{}",
                MicroVMInvalidState(StateError::MicroVMAlreadyConfigured)
            ),
            format!("{}", StateError::MicroVMAlreadyConfigured)
        );
        assert_eq!(
            format!("{}", MicroVMInvalidState(StateError::MicroVMAlreadyRunning)),
            format!("{}", StateError::MicroVMAlreadyRunning)
        );
        assert_eq!(
            format!("{}", MissingKernelConfig),
            "Cannot start microvm without kernel configuration.".to_string()
        );
        assert_eq!(
            format!("{}", NetDeviceNotConfigured),
            "The net device configuration is missing the tap device.".to_string()
        );
        assert_eq!(
            format!("{}", OpenBlockDevice(std::io::Error::from_raw_os_error(0))),
            format!(
                "Cannot open the block device backing file. {}",
                std::io::Error::from_raw_os_error(0)
            )
        );
        assert_eq!(
            format!(
                "{}",
                RegisterBlockDevice(device_manager::mmio::Error::IrqsExhausted)
            ),
            format!(
                "Cannot initialize a MMIO Block Device or add a device to the MMIO Bus. {}",
                device_manager::mmio::Error::IrqsExhausted
            )
        );
        assert_eq!(
            format!(
                "{}",
                RegisterBalloonDevice(device_manager::mmio::Error::IrqsExhausted)
            ),
            format!(
                "Cannot initialize a MMIO Balloon Device or add a device to the MMIO Bus. {}",
                device_manager::mmio::Error::IrqsExhausted
            )
        );
        assert_eq!(
            format!("{}", RegisterEvent),
            "Cannot add event to Epoll.".to_string()
        );
        assert_eq!(
            format!(
                "{}",
                RegisterMMIODevice(device_manager::mmio::Error::IrqsExhausted)
            ),
            "Cannot add a device to the MMIO Bus. no more IRQs are available".to_string()
        );
        assert_eq!(
            format!(
                "{}",
                RegisterNetDevice(device_manager::mmio::Error::IrqsExhausted)
            ),
            format!(
                "Cannot initialize a MMIO Network Device or add a device to the MMIO Bus. {}",
                device_manager::mmio::Error::IrqsExhausted
            )
        );
        assert_eq!(
            format!("{}", SeccompFilters(seccomp::Error::IntoBpf)),
            format!(
                "Cannot build seccomp filters. {:?}",
                seccomp::Error::IntoBpf
            )
        );
        assert_eq!(
            format!("{}", SignalVcpu(vstate::Error::NotEnoughMemorySlots)),
            format!(
                "Failed to signal vCPU. {:?}",
                vstate::Error::NotEnoughMemorySlots
            )
        );
        assert_eq!(
            format!("{}", Vcpu(vstate::Error::NotEnoughMemorySlots)),
            format!(
                "Cannot create a new vCPU. {:?}",
                vstate::Error::NotEnoughMemorySlots
            )
        );
        assert_eq!(
            format!("{}", VcpuConfigure(vstate::Error::NotEnoughMemorySlots)),
            format!(
                "vCPU configuration failed. {:?}",
                vstate::Error::NotEnoughMemorySlots
            )
        );
        assert_eq!(
            format!("{}", VcpusAlreadyPresent),
            "vCPUs have already been created.".to_string()
        );
        assert_eq!(
            format!("{}", VcpusNotConfigured),
            "vCPUs were not configured.".to_string()
        );
        assert_eq!(
            format!("{}", VcpuSpawn(vstate::Error::NotEnoughMemorySlots)),
            format!(
                "Cannot spawn vCPU thread. {:?}",
                vstate::Error::NotEnoughMemorySlots
            )
        );
    }
}
