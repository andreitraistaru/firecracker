// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Currently only supports X86_64.
#![cfg(target_arch = "x86_64")]

// Do not allow warnings. If any of our structures become FFI-unsafe we want to error.
#![deny(warnings)]

extern crate kvm_bindings;

use std::fmt::{self, Display, Formatter};
use std::io;
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
    Deserialize(SerializationError),
    IO(io::Error),
    MemfileSize,
    MissingMemFile,
    MissingMemSize,
    Mmap(io::Error),
    Serialize(SerializationError),
    UnsupportedAppVersion(Version),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use self::Error::*;
        match *self {
            BadMagicNumber => write!(f, "Bad snapshot magic number."),
            Deserialize(ref e) => write!(f, "Failed to deserialize: {}", e),
            IO(ref e) => write!(f, "Input/output error. {}", e),
            MemfileSize => write!(f, "The memory size defined in the snapshot header does not match the size of the memory file."),
            MissingMemFile => write!(f, "Missing guest memory file."),
            MissingMemSize => write!(f, "Missing guest memory size."),
            Mmap(ref e) => write!(f, "Failed to map memory: {}", e),
            Serialize(ref e) => write!(f, "Failed to serialize snapshot content. {}", e),
            UnsupportedAppVersion(v) => {
                write!(f, "Unsupported app version: {}.", v
                )
            }
        }
    }
}

type Result<T> = std::result::Result<T, Error>;

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
}

impl Display for Version {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.build)
    }
}

const FIRST_SUPPORTED_APP_VERSION: Version = Version {
    major: 1,
    minor: 0,
    build: 0,
};

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

    pub fn mem_size_mib(&self) -> u64 {
        self.mem_size_mib
    }

    pub fn validate(&self, mem_file_path: &str) -> Result<()> {
        let metadata = std::fs::metadata(mem_file_path).map_err(|_| Error::MissingMemFile)?;
        if self.mem_size_mib << 20 != metadata.len() {
            Err(Error::MemfileSize)
        } else {
            Ok(())
        }
    }
}

#[derive(Deserialize, Serialize)]
pub struct MicrovmState {
    header: SnapshotHdr,
    vm_info: VmInfo,
    vm_state: VmState,
    vcpu_states: Vec<VcpuState>,
    device_configs: DeviceConfigs,
    device_states: Vec<MmioDeviceState>,
}

impl MicrovmState {
    pub fn unpack(
        self,
    ) -> (
        VmInfo,
        VmState,
        Vec<VcpuState>,
        DeviceConfigs,
        Vec<MmioDeviceState>,
    ) {
        (
            self.vm_info,
            self.vm_state,
            self.vcpu_states,
            self.device_configs,
            self.device_states,
        )
    }
}

pub struct SnapshotEngine {}

impl SnapshotEngine {
    fn validate_header(header: &SnapshotHdr, current_app_version: Version) -> Result<()> {
        if header.magic_id != SNAPSHOT_MAGIC {
            return Err(Error::BadMagicNumber);
        }

        // We don't validate the Snapshot Version

        // Check if the app version associated with the snapshot is valid.
        // For the moment we accept any app version starting with 1.0.0 (first wisp version
        // supporting snapshotting) and ending with the current app version.
        if !(header.app_version.major >= FIRST_SUPPORTED_APP_VERSION.major
            && header.app_version.major <= current_app_version.major)
        {
            return Err(Error::UnsupportedAppVersion(header.app_version));
        }

        Ok(())
    }

    pub fn deserialize<P: AsRef<Path>>(
        path: P,
        current_app_version: Version,
    ) -> Result<MicrovmState> {
        let bytes = std::fs::read(path).map_err(Error::IO)?;

        let header: SnapshotHdr = bincode::deserialize(&bytes).map_err(Error::Deserialize)?;
        Self::validate_header(&header, current_app_version)?;

        let microvm_state = bincode::deserialize(&bytes).map_err(Error::Deserialize)?;
        Ok(microvm_state)
    }

    pub fn serialize<P: AsRef<Path>>(
        path: P,
        header: SnapshotHdr,
        vm_info: VmInfo,
        vm_state: VmState,
        vcpu_states: Vec<VcpuState>,
        device_configs: DeviceConfigs,
        device_states: Vec<MmioDeviceState>,
    ) -> Result<()> {
        let microvm_state = MicrovmState {
            header,
            vm_info,
            vm_state,
            vcpu_states,
            device_configs,
            device_states,
        };
        let bytes = bincode::serialize(&microvm_state).map_err(Error::Serialize)?;
        std::fs::write(path, &bytes).map_err(Error::IO)?;

        Ok(())
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

    impl fmt::Debug for SnapshotEngine {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "SnapshotEngine")
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
                | Error::Deserialize(_)
                | Error::IO(_)
                | Error::MemfileSize
                | Error::MissingMemFile
                | Error::MissingMemSize
                | Error::Mmap(_)
                | Error::Serialize(_)
                | Error::UnsupportedAppVersion(_) => (),
            };
            match (self, other) {
                (Error::BadMagicNumber, Error::BadMagicNumber) => true,
                (Error::Deserialize(_), Error::Deserialize(_)) => true,
                (Error::IO(_), Error::IO(_)) => true,
                (Error::MemfileSize, Error::MemfileSize) => true,
                (Error::MissingMemFile, Error::MissingMemFile) => true,
                (Error::MissingMemSize, Error::MissingMemSize) => true,
                (Error::Mmap(_), Error::Mmap(_)) => true,
                (Error::Serialize(_), Error::Serialize(_)) => true,
                (Error::UnsupportedAppVersion(_), Error::UnsupportedAppVersion(_)) => true,
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
    fn test_microvm_state_unpack() {
        let mem_size_mib = 1u64;
        let (_, vcpu) = setup_vcpu();
        let vcpu_states = vec![vcpu.save_state().unwrap()];

        let header = build_valid_header();
        let extra_info = VmInfo::new(mem_size_mib);
        let kvm_vm_state = VmState::default();
        let microvm_state = MicrovmState {
            header,
            vm_info: extra_info,
            vm_state: kvm_vm_state.clone(),
            vcpu_states: vcpu_states.clone(),
            device_configs: Default::default(),
            device_states: vec![],
        };

        let (
            unpacked_vm_info,
            unpacked_kvm_vm_state,
            unpacked_vcpu_states,
            _,
            unpacked_device_states,
        ) = microvm_state.unpack();

        assert_eq!(unpacked_vm_info.mem_size_mib(), mem_size_mib);
        assert_eq!(unpacked_kvm_vm_state, kvm_vm_state);
        assert_eq!(unpacked_vcpu_states, vcpu_states);
        // Can't compare MmioDeviceState objects, checking for `is_empty()` will suffice.
        assert!(unpacked_device_states.is_empty());
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
        SnapshotEngine::serialize(
            snapshot_path,
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
            let ret = SnapshotEngine::serialize(
                inaccessible_path,
                header.clone(),
                extra_info.clone(),
                vm_state.clone(),
                vec![vcpu_state.clone()],
                Default::default(),
                vec![],
            );
            assert!(ret.is_err());
            assert_eq!(
                format!("{}", ret.err().unwrap()),
                "Input/output error. No such file or directory (os error 2)"
            );

            // Good case: snapshot is created.
            assert!(SnapshotEngine::serialize(
                snapshot_path,
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
            let ret = SnapshotEngine::deserialize("/foo/bar", header.app_version);
            assert!(ret.is_err());
            assert_eq!(
                format!("{}", ret.err().unwrap()),
                "Input/output error. No such file or directory (os error 2)"
            );

            // Test error case: path points to a directory.
            let ret =
                SnapshotEngine::deserialize(std::env::current_dir().unwrap(), header.app_version);
            assert!(ret.is_err());
            assert_eq!(
                format!("{}", ret.err().unwrap()),
                "Input/output error. Is a directory (os error 21)"
            );

            // Test error case: path points to an invalid snapshot.
            let bad_snap = NamedTempFile::new().unwrap().into_temp_path();
            let ret = SnapshotEngine::deserialize(bad_snap, header.app_version);
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
                let bad_snapshot_path = make_snapshot(
                    &bad_snap_path,
                    vm_state.clone(),
                    vcpu_state.clone(),
                    Default::default(),
                    bad_header.clone(),
                    extra_info.clone(),
                );
                assert!(std::fs::metadata(bad_snapshot_path.as_str()).is_ok());
                let ret =
                    SnapshotEngine::deserialize(bad_snapshot_path.as_str(), header.app_version);
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
                let bad_snapshot_path = make_snapshot(
                    &bad_snap_path,
                    vm_state.clone(),
                    vcpu_state.clone(),
                    Default::default(),
                    bad_header.clone(),
                    extra_info.clone(),
                );
                let ret =
                    SnapshotEngine::deserialize(bad_snapshot_path.as_str(), header.app_version);
                assert!(ret.is_err());
                assert_eq!(
                    format!("{}", ret.err().unwrap()),
                    "Unsupported app version: 0.0.0."
                );
            }

            // Good case: valid snapshot.
            let ret = SnapshotEngine::deserialize(snapshot_path, header.app_version);
            assert!(ret.is_ok());
            let microvm_state = ret.unwrap();

            // Test memory file validation on snapshot loading.
            {
                // Test error case: incorrect memory file path in header.
                let ret = microvm_state.vm_info.validate("/foo/bar");
                assert_eq!(
                    format!("{}", ret.unwrap_err()),
                    "Missing guest memory file."
                );

                // Test error case: incorrect guest memory size in header.
                let tmp_memfile = NamedTempFile::new().unwrap();
                let tmp_memfile_path = tmp_memfile.path();
                let memfile_path = tmp_memfile_path.to_str().unwrap();

                let ret = microvm_state.vm_info.validate(memfile_path);
                assert_eq!(
                    format!("{}", ret.unwrap_err()),
                    "The memory size defined in the snapshot header does not match the size of the memory file."
                );

                // Test memory file size validation success.
                tmp_memfile
                    .as_file()
                    .set_len(mem_size_mib as u64 * 1024 * 1024)
                    .unwrap();
                let ret = microvm_state.vm_info.validate(memfile_path);
                assert!(ret.is_ok());
            }

            // Verify header deserialization.
            assert!(header.eq(&microvm_state.header));

            // Verify VM state deserialization.
            assert!(vm_state.eq(&microvm_state.vm_state));

            // Verify vCPU state deserialization.
            assert!(vec![vcpu_state].eq(&microvm_state.vcpu_states));
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
        assert_eq!(
            format!(
                "{}",
                Error::Deserialize(SerializationError::from(io::Error::from_raw_os_error(0)))
            ),
            format!("Failed to deserialize: io error: {}", err0_str)
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
}
