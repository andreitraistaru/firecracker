// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Currently only supports X86_64.
#![cfg(target_arch = "x86_64")]

// Do not allow warnings. If any of our structures become FFI-unsafe we want to error.
#![deny(warnings)]

extern crate kvm_bindings;

use std::fmt::{self, Display, Formatter};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::Path;

use bincode::Error as SerializationError;

use devices::virtio::MmioDeviceState;
use vmm_config::device_config::DeviceConfigs;
use vstate::{VcpuState, VmState};

/// Magic number, verifies a snapshot file's validity.
pub(super) const SNAPSHOT_MAGIC: u64 = 0xEDA3_25D9_EDA3_25D9;
/// Snapshot format version. Can vary independently from Firecracker's version.
pub(super) const SNAPSHOT_VERSION: Version = Version {
    major: 0,
    minor: 0,
    build: 1,
};

// Helper enum that signals a generic versioning error. Caller translates it to a specific error.
#[derive(Debug)]
pub enum VersionError {
    InvalidFormat,
}

/// Snapshot related errors.
#[derive(Debug)]
pub enum Error {
    BadMagicNumber,
    BadVcpuCount,
    CreateNew(io::Error),
    Deserialize(SerializationError),
    InvalidFileType,
    IO(io::Error),
    MemfileSize,
    MissingMemFile,
    MissingMemSize,
    MissingVcpuNum,
    Mmap(io::Error),
    OpenExisting(io::Error),
    Serialize(SerializationError),
    UnsupportedAppVersion(Version),
    UnsupportedSnapshotVersion(Version),
    Truncate(io::Error),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use self::Error::*;
        match *self {
            BadMagicNumber => write!(f, "Bad snapshot magic number."),
            BadVcpuCount => write!(f, "The vCPU count in the snapshot header does not match the number of vCPUs serialized in the snapshot."),
            CreateNew(ref e) => write!(f, "Failed to create new snapshot file: {}", e),
            Deserialize(ref e) => write!(f, "Failed to deserialize: {}", e),
            InvalidFileType => write!(f, "Invalid snapshot file type."),
            IO(ref e) => write!(f, "Input/output error. {}", e),
            MemfileSize => write!(f, "The memory size defined in the snapshot header does not match the size of the memory file."),
            MissingMemFile => write!(f, "Missing guest memory file."),
            MissingMemSize => write!(f, "Missing guest memory size."),
            MissingVcpuNum => write!(f, "Missing number of vCPUs."),
            Mmap(ref e) => write!(f, "Failed to map memory: {}", e),
            OpenExisting(ref e) => write!(f, "Failed to open snapshot file: {}", e),
            Serialize(ref e) => write!(f, "Failed to serialize snapshot content. {}", e),
            UnsupportedAppVersion(v) => {
                write!(f, "Unsupported app version: {}.", v
                )
            }
            UnsupportedSnapshotVersion(v) => write!(f, "Unsupported snapshot version: {}.", v),
            Truncate(ref e) => write!(f, "Failed to truncate snapshot file: {}", e),
        }
    }
}

type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, PartialEq, PartialOrd)]
pub enum VersionComponent {
    Major,
    Minor,
    Build,
}

/// Version struct composed of major, minor and build numbers.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, PartialOrd, Serialize)]
pub struct Version {
    major: u16,
    minor: u16,
    build: u32,
}

impl Version {
    pub fn new(major: u16, minor: u16, build: u32) -> Version {
        Version {
            major,
            minor,
            build,
        }
    }

    pub fn from_str(version: &str) -> std::result::Result<Version, VersionError> {
        let nums: &Vec<u32> = &version
            .split('.')
            .flat_map(str::parse::<u32>)
            .collect::<Vec<u32>>();
        if nums.len() != 3 {
            return Err(VersionError::InvalidFormat);
        }

        Ok(Version {
            major: nums[0] as u16,
            minor: nums[1] as u16,
            build: nums[2],
        })
    }

    /// Transform each component beginning from `start` to it's max possible value.
    fn ceil(self, start: VersionComponent) -> Version {
        let mut ceil = self;

        if start <= VersionComponent::Major {
            ceil.major = std::u16::MAX;
        }

        if start <= VersionComponent::Minor {
            ceil.minor = std::u16::MAX;
        }

        if start <= VersionComponent::Build {
            ceil.build = std::u32::MAX;
        }

        ceil
    }

    pub fn is_in_range(self, from: Version, to: Version) -> bool {
        self >= from && self <= to
    }
}

impl Display for Version {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.build)
    }
}

/// The header of a Firecracker snapshot image.
/// It strictly contains the most basic info needed to identify a Snapshot
/// and to understand how to read it. This should never change.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SnapshotHdr {
    /** SNAPSHOT_IMAGE_MAGIC */
    magic_id: u64,
    /** SNAPSHOT_VERSION */
    snapshot_version: Version,
    /** Firecracker version **/
    app_version: Version,
}

impl SnapshotHdr {
    pub fn new(app_version: Version) -> Self {
        SnapshotHdr {
            magic_id: SNAPSHOT_MAGIC,
            snapshot_version: SNAPSHOT_VERSION,
            app_version,
        }
    }
}

/// VM info that is not strictly needed inside `SnapshotHdr`
/// but is not part of the VM state either.
#[derive(Clone, Deserialize, Serialize)]
pub struct VmInfo {
    /** Guest memory size */
    mem_size_mib: u64,
}

impl VmInfo {
    pub fn new(mem_size_mib: u64) -> Self {
        VmInfo { mem_size_mib }
    }
}

#[derive(Deserialize, Serialize)]
pub struct MicrovmState {
    pub header: SnapshotHdr,
    pub vm_info: VmInfo,
    pub vm_state: VmState,
    pub vcpu_states: Vec<VcpuState>,
    pub device_states: Vec<MmioDeviceState>,
    pub device_configs: DeviceConfigs,
}

pub struct SnapshotImage {
    file: File,
    microvm_state: Option<MicrovmState>,
}

impl SnapshotImage {
    pub fn create_new<P: AsRef<Path>>(path: P) -> Result<SnapshotImage> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())
            .map_err(Error::CreateNew)?;
        Ok(SnapshotImage {
            file,
            microvm_state: None,
        })
    }

    pub fn open_existing<P: AsRef<Path>>(path: P, app_version: Version) -> Result<SnapshotImage> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(false)
            .open(path.as_ref())
            .map_err(Error::OpenExisting)?;

        let metadata = file.metadata().map_err(Error::OpenExisting)?;
        if !metadata.is_file() {
            return Err(Error::InvalidFileType);
        }

        let bytes = std::fs::read(path).map_err(Error::IO)?;
        let microvm_state = bincode::deserialize(&bytes).map_err(Error::Deserialize)?;

        // Cross-version deserialization is not supported yet.
        Self::validate_header(&microvm_state, app_version)?;

        Ok(SnapshotImage {
            file,
            microvm_state: Some(microvm_state),
        })
    }

    pub fn serialize_microvm(
        &mut self,
        header: SnapshotHdr,
        extra_info: VmInfo,
        vm_state: VmState,
        vcpu_states: Vec<VcpuState>,
        device_configs: DeviceConfigs,
        device_states: Vec<MmioDeviceState>,
    ) -> Result<()> {
        let microvm_state = MicrovmState {
            header,
            vm_info: extra_info,
            vm_state,
            vcpu_states,
            device_configs,
            device_states,
        };
        let bytes = bincode::serialize(&microvm_state).map_err(Error::Serialize)?;
        self.file.write_all(&bytes).map_err(Error::IO)?;
        self.file.flush().map_err(Error::IO)?;

        self.microvm_state = Some(microvm_state);

        Ok(())
    }

    pub fn mem_size_mib(&self) -> Option<usize> {
        self.microvm_state
            .as_ref()
            .map(|state| state.vm_info.mem_size_mib as usize)
    }

    pub fn kvm_vm_state(&self) -> Option<&VmState> {
        self.microvm_state.as_ref().map(|state| &state.vm_state)
    }

    pub fn device_states(&self) -> Option<&Vec<MmioDeviceState>> {
        self.microvm_state
            .as_ref()
            .map(|state| &state.device_states)
    }

    #[allow(dead_code)]
    pub fn vcpu_states(&self) -> Option<&Vec<VcpuState>> {
        self.microvm_state.as_ref().map(|state| &state.vcpu_states)
    }

    pub fn into_vcpu_states(self) -> Vec<VcpuState> {
        self.microvm_state
            .map_or_else(|| vec![], |state| state.vcpu_states)
    }

    pub fn device_configs(&self) -> DeviceConfigs {
        self.microvm_state
            .as_ref()
            .map_or(Default::default(), |state| state.device_configs.clone())
    }

    pub fn validate_mem_file_size(&self, mem_file_path: &str) -> Result<()> {
        let metadata = std::fs::metadata(mem_file_path).map_err(|_| Error::MissingMemFile)?;
        if self
            .microvm_state
            .as_ref()
            .ok_or(Error::MissingMemFile)?
            .vm_info
            .mem_size_mib
            << 20
            != metadata.len()
        {
            Err(Error::MemfileSize)
        } else {
            Ok(())
        }
    }

    fn validate_header(microvm_state: &MicrovmState, current_app_version: Version) -> Result<()> {
        if microvm_state.header.magic_id != SNAPSHOT_MAGIC {
            return Err(Error::BadMagicNumber);
        }

        // We don't validate the Snapshot Version

        // Check if the app version associated with the snapshot is valid.
        // For the moment we accept any app version starting 1.0.0 (first wisp version
        // supporting snapshotting) and ending with the current app version.
        if !microvm_state.header.app_version.is_in_range(
            Version::new(1, 0, 0),
            current_app_version.ceil(VersionComponent::Minor),
        ) {
            return Err(Error::UnsupportedAppVersion(
                microvm_state.header.app_version,
            ));
        }

        Ok(())
    }
}

impl AsRawFd for SnapshotImage {
    fn as_raw_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}

#[cfg(test)]
mod tests {
    extern crate tempfile;

    use self::tempfile::{NamedTempFile, TempPath};

    use std::fmt;
    use std::sync::mpsc::channel;

    use super::super::{KvmContext, Vcpu, Vm};
    use super::*;

    use memory_model::{GuestAddress, GuestMemory};
    use sys_util::EventFd;

    impl fmt::Debug for SnapshotImage {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "SnapshotImage")
        }
    }

    impl fmt::Debug for VcpuState {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "VcpuState")
        }
    }

    impl fmt::Debug for VmState {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "VmState")
        }
    }

    const APP_VER: Version = Version {
        major: 0xDEAD,
        minor: 0xBEAF,
        build: 0xCAFE_BABE,
    };

    impl PartialEq for Error {
        fn eq(&self, other: &Self) -> bool {
            // Guard match with no wildcard to make sure we catch new enum variants.
            match self {
                Error::BadMagicNumber
                | Error::BadVcpuCount
                | Error::CreateNew(_)
                | Error::Deserialize(_)
                | Error::InvalidFileType
                | Error::IO(_)
                | Error::MemfileSize
                | Error::MissingVcpuNum
                | Error::MissingMemFile
                | Error::MissingMemSize
                | Error::Mmap(_)
                | Error::OpenExisting(_)
                | Error::Serialize(_)
                | Error::UnsupportedAppVersion(_)
                | Error::UnsupportedSnapshotVersion(_)
                | Error::Truncate(_) => (),
            };
            match (self, other) {
                (Error::BadMagicNumber, Error::BadMagicNumber) => true,
                (Error::BadVcpuCount, Error::BadVcpuCount) => true,
                (Error::CreateNew(_), Error::CreateNew(_)) => true,
                (Error::Deserialize(_), Error::Deserialize(_)) => true,
                (Error::InvalidFileType, Error::InvalidFileType) => true,
                (Error::IO(_), Error::IO(_)) => true,
                (Error::MissingVcpuNum, Error::MissingVcpuNum) => true,
                (Error::MemfileSize, Error::MemfileSize) => true,
                (Error::MissingMemFile, Error::MissingMemFile) => true,
                (Error::MissingMemSize, Error::MissingMemSize) => true,
                (Error::Mmap(_), Error::Mmap(_)) => true,
                (Error::OpenExisting(_), Error::OpenExisting(_)) => true,
                (Error::Serialize(_), Error::Serialize(_)) => true,
                (Error::UnsupportedAppVersion(_), Error::UnsupportedAppVersion(_)) => true,
                (Error::UnsupportedSnapshotVersion(_), Error::UnsupportedSnapshotVersion(_)) => {
                    true
                }
                (Error::Truncate(_), Error::Truncate(_)) => true,
                _ => false,
            }
        }
    }

    impl PartialEq for SnapshotHdr {
        fn eq(&self, other: &Self) -> bool {
            self.magic_id == other.magic_id
                && self.snapshot_version == other.snapshot_version
                && self.app_version == other.app_version
        }
    }

    impl PartialEq for VmState {
        fn eq(&self, other: &Self) -> bool {
            unsafe {
                libc::memcmp(
                    self as *const VmState as *const libc::c_void,
                    other as *const VmState as *const libc::c_void,
                    std::mem::size_of::<VmState>(),
                ) == 0
            }
        }
    }

    impl Clone for VcpuState {
        fn clone(&self) -> Self {
            VcpuState {
                cpuid: self.cpuid.clone(),
                msrs: self.msrs.clone(),
                debug_regs: self.debug_regs,
                lapic: self.lapic,
                mp_state: self.mp_state,
                regs: self.regs,
                sregs: self.sregs,
                vcpu_events: self.vcpu_events,
                xcrs: self.xcrs,
                xsave: self.xsave,
            }
        }
    }

    impl PartialEq for VcpuState {
        fn eq(&self, other: &Self) -> bool {
            self.cpuid.eq(&other.cpuid)
                && self.msrs.eq(&other.msrs)
                && self.debug_regs.eq(&other.debug_regs)
                && self.lapic.regs[..].eq(&other.lapic.regs[..])
                && self.mp_state.eq(&other.mp_state)
                && self.regs.eq(&other.regs)
                && self.sregs.eq(&other.sregs)
                && self.vcpu_events.eq(&other.vcpu_events)
                && self.xcrs.eq(&other.xcrs)
                && self.xsave.region[..].eq(&other.xsave.region[..])
        }
    }

    // Auxiliary function being used throughout the tests.
    fn setup_vcpu() -> (Vm, Vcpu) {
        let kvm = KvmContext::new().unwrap();
        let gm = GuestMemory::new_anon_from_tuples(&[(GuestAddress(0), 0x10000)]).unwrap();
        let mut vm = Vm::new(kvm.fd()).expect("Cannot create new vm");
        assert!(vm.memory_init(gm, &kvm).is_ok());

        vm.setup_irqchip().unwrap();

        let (_s1, r1) = channel();
        let (s2, _r2) = channel();
        let vcpu = Vcpu::new(
            1,
            &vm,
            devices::Bus::new(),
            EventFd::new().unwrap(),
            r1,
            s2,
            super::super::TimestampUs::default(),
        )
        .expect("Cannot create Vcpu");

        (vm, vcpu)
    }

    fn build_valid_header() -> SnapshotHdr {
        SnapshotHdr {
            magic_id: SNAPSHOT_MAGIC,
            snapshot_version: SNAPSHOT_VERSION,
            app_version: APP_VER,
        }
    }

    #[test]
    fn test_header_serialization() {
        let header = build_valid_header();
        let serialized_header = bincode::serialize(&header).unwrap();
        let deserialized_header =
            bincode::deserialize::<SnapshotHdr>(serialized_header.as_slice()).unwrap();
        assert_eq!(header, deserialized_header);
    }

    #[test]
    fn test_snapshot_getters() {
        let mem_size_mib = 1u64;
        let (_, vcpu) = setup_vcpu();
        let vcpu_states = vec![vcpu.save_state().unwrap()];

        let header = build_valid_header();
        let extra_info = VmInfo::new(mem_size_mib);
        let kvm_vm_state = VmState::default();
        let microvm_state = Some(MicrovmState {
            header,
            vm_info: extra_info,
            vm_state: kvm_vm_state.clone(),
            vcpu_states: vcpu_states.clone(),
            device_states: vec![],
            device_configs: Default::default(),
        });
        let file = NamedTempFile::new().unwrap().into_file();
        let fd = file.as_raw_fd();
        let image = SnapshotImage {
            file,
            microvm_state,
        };

        assert_eq!(image.mem_size_mib().unwrap(), mem_size_mib as usize);
        assert_eq!(image.kvm_vm_state().unwrap(), &kvm_vm_state);
        // Can't compare MmioDeviceState objects, checking for `Some` will suffice.
        assert!(image.device_states().is_some());
        assert_eq!(image.as_raw_fd(), fd);
        assert_eq!(image.vcpu_states().unwrap(), &vcpu_states);
        assert_eq!(image.into_vcpu_states(), vcpu_states);
    }

    fn make_snapshot(
        tmp_snapshot_path: &TempPath,
        vm_state: VmState,
        vcpu_state: VcpuState,
        device_configs: DeviceConfigs,
        header: SnapshotHdr,
        extra_info: VmInfo,
    ) -> String {
        let snapshot_path = tmp_snapshot_path.to_str().unwrap();
        let mut image = SnapshotImage::create_new(snapshot_path).unwrap();
        image
            .serialize_microvm(
                header,
                extra_info,
                vm_state,
                vec![vcpu_state],
                device_configs,
                vec![],
            )
            .unwrap();
        snapshot_path.to_string()
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn test_snapshot_ser_deser() {
        let tmp_snapshot_path = NamedTempFile::new().unwrap().into_temp_path();
        let snapshot_path = tmp_snapshot_path.to_str().unwrap();

        let mem_size_mib: usize = 10;

        let header = build_valid_header();
        let extra_info = VmInfo::new(mem_size_mib as u64);

        let (vm, vcpu) = setup_vcpu();
        let vm_state = vm.save_state().unwrap();
        let vcpu_state = vcpu.save_state().unwrap();

        // Save snapshot.
        {
            let inaccessible_path = "/foo/bar";

            // Test error case: inaccessible snapshot path.
            let ret = SnapshotImage::create_new(inaccessible_path);
            assert!(ret.is_err());
            assert_eq!(
                format!("{}", ret.err().unwrap()),
                "Failed to create new snapshot file: No such file or directory (os error 2)"
            );

            // Good case: snapshot is created.
            let ret = SnapshotImage::create_new(snapshot_path);
            assert!(ret.is_ok());
            let mut image = ret.unwrap();

            assert!(image
                .serialize_microvm(
                    header.clone(),
                    extra_info.clone(),
                    vm_state.clone(),
                    vec![vcpu_state.clone()],
                    Default::default(),
                    vec![]
                )
                .is_ok());
        }

        // Restore snapshot.
        {
            // Test error case: inaccessible path.
            let ret = SnapshotImage::open_existing("/foo/bar", header.app_version);
            assert!(ret.is_err());
            assert_eq!(
                format!("{}", ret.err().unwrap()),
                "Failed to open snapshot file: No such file or directory (os error 2)"
            );

            // Test error case: path points to a directory.
            let ret =
                SnapshotImage::open_existing(std::env::current_dir().unwrap(), header.app_version);
            assert!(ret.is_err());
            assert_eq!(
                format!("{}", ret.err().unwrap()),
                "Failed to open snapshot file: Is a directory (os error 21)"
            );

            // Test error case: path points to an invalid snapshot.
            let bad_snap = NamedTempFile::new().unwrap().into_temp_path();
            let ret = SnapshotImage::open_existing(bad_snap, header.app_version);
            assert!(ret.is_err());
            assert_eq!(
                format!("{}", ret.err().unwrap()),
                "Failed to deserialize: io error: failed to fill whole buffer"
            );

            {
                // Test error case: invalid magic number in header.
                let bad_snap = NamedTempFile::new().unwrap();
                let bad_snap_path = bad_snap.into_temp_path();
                let mut bad_header = header.clone();
                bad_header.magic_id = SNAPSHOT_MAGIC + 1;
                let bad_image_path = make_snapshot(
                    &bad_snap_path,
                    vm_state.clone(),
                    vcpu_state.clone(),
                    Default::default(),
                    bad_header.clone(),
                    extra_info.clone(),
                );
                assert!(std::fs::metadata(bad_image_path.as_str()).is_ok());
                let ret = SnapshotImage::open_existing(bad_image_path.as_str(), header.app_version);
                assert!(ret.is_err());
                assert_eq!(
                    format!("{}", ret.err().unwrap()),
                    "Bad snapshot magic number."
                );
            }

            {
                // Test error case: invalid app version number in header.
                let bad_snap = NamedTempFile::new().unwrap();
                let bad_snap_path = bad_snap.into_temp_path();
                let mut bad_header = header.clone();
                bad_header.app_version = Version::new(0, 0, 0);
                let bad_image_path = make_snapshot(
                    &bad_snap_path,
                    vm_state.clone(),
                    vcpu_state.clone(),
                    Default::default(),
                    bad_header.clone(),
                    extra_info.clone(),
                );
                let ret = SnapshotImage::open_existing(bad_image_path.as_str(), header.app_version);
                assert!(ret.is_err());
                assert_eq!(
                    format!("{}", ret.err().unwrap()),
                    "Unsupported app version: 0.0.0."
                );
            }

            // Good case: valid snapshot.
            let ret = SnapshotImage::open_existing(snapshot_path, header.app_version);
            assert!(ret.is_ok());
            let image = ret.unwrap();

            // Test memory file validation on snapshot loading.
            {
                // Test error case: incorrect memory file path in header.
                let ret = image.validate_mem_file_size("/foo/bar");
                assert_eq!(
                    format!("{}", ret.unwrap_err()),
                    "Missing guest memory file."
                );

                // Test error case: incorrect guest memory size in header.
                let tmp_memfile = NamedTempFile::new().unwrap();
                let tmp_memfile_path = tmp_memfile.path();
                let memfile_path = tmp_memfile_path.to_str().unwrap();

                let ret = image.validate_mem_file_size(memfile_path);
                assert_eq!(
                    format!("{}", ret.unwrap_err()),
                    "The memory size defined in the snapshot header does not match the size of the memory file."
                );

                // Test memory file size validation success.
                tmp_memfile
                    .as_file()
                    .set_len(mem_size_mib as u64 * 1024 * 1024)
                    .unwrap();
                let ret = image.validate_mem_file_size(memfile_path);
                assert!(ret.is_ok());
            }

            // Verify header deserialization.
            assert!(header.eq(&image.microvm_state.as_ref().unwrap().header));

            // Verify VM state deserialization.
            assert!(vm_state.eq(image.kvm_vm_state().unwrap()));

            // Verify vCPU state deserialization.
            assert!(vec![vcpu_state].eq(image.vcpu_states().unwrap()));
        }
    }

    #[test]
    fn test_error_messages() {
        #[cfg(target_env = "musl")]
        let err0_str = "No error information (os error 0)";
        #[cfg(target_env = "gnu")]
        let err0_str = "Success (os error 0)";

        assert_eq!(
            format!("{}", Error::BadMagicNumber),
            "Bad snapshot magic number."
        );
        assert_eq!(format!("{}", Error::BadVcpuCount),
                   "The vCPU count in the snapshot header does not match the number of vCPUs serialized in the snapshot.");
        assert_eq!(
            format!("{}", Error::CreateNew(io::Error::from_raw_os_error(0))),
            format!("Failed to create new snapshot file: {}", err0_str)
        );
        assert_eq!(
            format!(
                "{}",
                Error::Deserialize(SerializationError::from(io::Error::from_raw_os_error(0)))
            ),
            format!("Failed to deserialize: io error: {}", err0_str)
        );
        assert_eq!(
            format!("{}", Error::InvalidFileType),
            "Invalid snapshot file type."
        );
        assert_eq!(
            format!("{}", Error::IO(io::Error::from_raw_os_error(0))),
            format!("Input/output error. {}", err0_str)
        );
        assert_eq!(
            format!("{}", Error::MemfileSize),
            "The memory size defined in the snapshot header does not match the size of the memory file."
        );
        assert_eq!(
            format!("{}", Error::MissingVcpuNum),
            "Missing number of vCPUs."
        );
        assert_eq!(
            format!("{}", Error::MissingMemFile),
            "Missing guest memory file."
        );
        assert_eq!(
            format!("{}", Error::MissingMemSize),
            "Missing guest memory size."
        );
        assert_eq!(
            format!("{}", Error::Mmap(io::Error::from_raw_os_error(0))),
            format!("Failed to map memory: {}", err0_str)
        );
        assert_eq!(
            format!("{}", Error::OpenExisting(io::Error::from_raw_os_error(0))),
            format!("Failed to open snapshot file: {}", err0_str)
        );
        assert_eq!(
            format!(
                "{}",
                Error::Serialize(SerializationError::from(io::Error::from_raw_os_error(0)))
            ),
            format!(
                "Failed to serialize snapshot content. io error: {}",
                err0_str
            )
        );
        assert_eq!(
            format!("{}", Error::UnsupportedAppVersion(Version::new(1, 2, 3))),
            "Unsupported app version: 1.2.3."
        );
        assert_eq!(
            format!(
                "{}",
                Error::UnsupportedSnapshotVersion(Version::new(0, 0, 42))
            ),
            "Unsupported snapshot version: 0.0.42."
        );
        assert_eq!(
            format!("{}", Error::Truncate(io::Error::from_raw_os_error(0))),
            format!("Failed to truncate snapshot file: {}", err0_str)
        );
    }

    #[test]
    fn test_version_new() {
        let v = Version {
            major: 1,
            minor: 2,
            build: 3,
        };

        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.build, 3);
    }

    #[test]
    fn test_version_from_str() {
        let v = Version::from_str("1.2.3").unwrap();
        assert_eq!(v, Version::new(1, 2, 3));

        assert!(Version::from_str("1.2").is_err());
        assert!(Version::from_str("1.2.3.4").is_err());
    }

    #[test]
    fn test_version_ceil() {
        let v = Version::new(1, 2, 3);

        assert_eq!(
            v.ceil(VersionComponent::Build),
            Version::new(1, 2, std::u32::MAX)
        );

        assert_eq!(
            v.ceil(VersionComponent::Minor),
            Version::new(1, std::u16::MAX, std::u32::MAX)
        );

        assert_eq!(
            v.ceil(VersionComponent::Major),
            Version::new(std::u16::MAX, std::u16::MAX, std::u32::MAX)
        );
    }

    #[test]
    fn test_version_is_in_range() {
        let v = Version::new(1, 2, 3);

        assert!(v.is_in_range(Version::new(1, 2, 3), Version::new(1, 2, 3)));
        assert!(v.is_in_range(Version::new(1, 2, 2), Version::new(1, 2, 4)));
        assert!(!v.is_in_range(Version::new(1, 2, 1), Version::new(1, 2, 2)));
        assert!(!v.is_in_range(Version::new(1, 2, 4), Version::new(1, 2, 6)));

        assert!(v.is_in_range(Version::new(1, 1, 0), Version::new(1, 3, 0)));
        assert!(!v.is_in_range(Version::new(1, 0, 0), Version::new(1, 2, 0)));
        assert!(!v.is_in_range(Version::new(1, 3, 0), Version::new(1, 5, 0)));

        assert!(v.is_in_range(Version::new(1, 0, 0), Version::new(2, 0, 0)));
        assert!(!v.is_in_range(Version::new(0, 0, 0), Version::new(1, 0, 0)));
        assert!(!v.is_in_range(Version::new(3, 0, 0), Version::new(5, 0, 0)));
    }
}
