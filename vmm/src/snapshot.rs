// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Currently only supports X86_64.
#![cfg(target_arch = "x86_64")]

// Do not allow warnings. If any of our structures become FFI-unsafe we want to error.
#![deny(warnings)]

extern crate kvm_bindings;

use std::cmp;
use std::fmt::{self, Display, Formatter};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::Path;

use bincode::Error as SerializationError;

use devices::virtio::MmioDeviceState;
use serialize::SnapshotReaderWriter;
use vmm_config::machine_config::VmConfig;
use vstate::{VcpuState, VmState};

/// Magic number, verifies a snapshot file's validity.
pub(super) const SNAPSHOT_MAGIC: u64 = 0xEDA3_25D9_EDA3_25D9;
/// Snapshot format version. Can vary independently from Firecracker's version.
pub(super) const SNAPSHOT_VERSION: u64 = 0x0000_0000_0000_0001;

/// How to deal with deserializing snapshots of different versions.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(super) enum SnapshotVersionSupport {
    /// Any version is allowed through.
    Any,
    /// A specific `Major` delta is allowed.
    Major(u16),
    /// The same `Major` and a specific `Minor` delta are allowed.
    Minor(u16),
    /// The same `Major.Minor` and a specific `Build` delta are allowed.
    Build(u32),
    /// Only the same version is allowed.
    Same,
}

// Helper enum that signals a generic versioning error. Caller translates it to a specific error.
enum VersionError {
    UnsupportedVersion,
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
    UnsupportedAppVersion(u64),
    UnsupportedSnapshotVersion(u64),
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
                write!(f, "Unsupported app version: {}.", Version::from(v)
                )
            },
            UnsupportedSnapshotVersion(v) => write!(f, "Unsupported snapshot version: {}.", Version::from(v)),
            Truncate(ref e) => write!(f, "Failed to truncate snapshot file: {}", e),
        }
    }
}

type Result<T> = std::result::Result<T, Error>;

/// Version struct composed of major, minor and build numbers.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct Version {
    major: u16,
    minor: u16,
    build: u32,
}

impl Version {
    fn major_delta(self, rhs: Version) -> u16 {
        cmp::max(self.major, rhs.major) - cmp::min(self.major, rhs.major)
    }

    fn minor_delta(self, rhs: Version) -> u16 {
        cmp::max(self.minor, rhs.minor) - cmp::min(self.minor, rhs.minor)
    }

    fn build_delta(self, rhs: Version) -> u32 {
        cmp::max(self.build, rhs.build) - cmp::min(self.build, rhs.build)
    }
}

impl Into<u64> for Version {
    fn into(self) -> u64 {
        (u64::from(self.major) << 48) | (u64::from(self.minor) << 32) | u64::from(self.build)
    }
}

impl From<u64> for Version {
    fn from(version: u64) -> Self {
        Version {
            major: (version >> 48) as u16,
            minor: ((version >> 32) & 0xFFFF) as u16,
            build: (version & 0xFFFF_FFFF) as u32,
        }
    }
}

impl From<(u16, u16, u32)> for Version {
    fn from(tuple_ver: (u16, u16, u32)) -> Self {
        Version {
            major: tuple_ver.0,
            minor: tuple_ver.1,
            build: tuple_ver.2,
        }
    }
}

impl Display for Version {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.build)
    }
}

/** The header of a Firecracker snapshot image. */
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SnapshotHdr {
    /** SNAPSHOT_IMAGE_MAGIC */
    magic_id: u64,
    /** SNAPSHOT_VERSION */
    snapshot_version: Version,
    /** Firecracker version **/
    app_version: Version,
    /** Guest memory size */
    mem_size_mib: u64,
    /** Number of vCPUs. */
    vcpu_count: u8,
}

impl SnapshotHdr {
    pub fn new(mem_size_mib: u64, vcpu_count: u8, app_version: Version) -> Self {
        SnapshotHdr {
            magic_id: SNAPSHOT_MAGIC,
            snapshot_version: SNAPSHOT_VERSION.into(),
            app_version,
            mem_size_mib,
            vcpu_count,
        }
    }
}

#[derive(Deserialize, Serialize)]
pub struct MicrovmState {
    pub header: SnapshotHdr,
    pub vm_state: Option<VmState>,
    pub vcpu_states: Vec<VcpuState>,
    pub device_states: Vec<MmioDeviceState>,
}

pub struct SnapshotImage {
    file: File,
    pub microvm_state: MicrovmState,
}

impl SnapshotImage {
    pub fn create_new<P: AsRef<Path>>(
        path: P,
        vm_cfg: VmConfig,
        app_version: Version,
    ) -> Result<SnapshotImage> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())
            .map_err(Error::CreateNew)?;
        let header = SnapshotHdr {
            magic_id: SNAPSHOT_MAGIC,
            snapshot_version: SNAPSHOT_VERSION.into(),
            app_version,
            mem_size_mib: vm_cfg.mem_size_mib.ok_or(Error::MissingMemSize)? as u64,
            vcpu_count: vm_cfg.vcpu_count.ok_or(Error::MissingVcpuNum)?,
        };
        let microvm_state = MicrovmState {
            header,
            vm_state: None,
            vcpu_states: vec![],
            device_states: vec![],
        };
        Ok(SnapshotImage {
            file,
            microvm_state,
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

        let reader_writer = SnapshotReaderWriter::new(&file, 0, metadata.len() as u64, false)
            .map_err(Error::OpenExisting)?;
        let microvm_state =
            bincode::deserialize_from::<SnapshotReaderWriter, MicrovmState>(reader_writer)
                .map_err(Error::Deserialize)?;

        // Cross-version deserialization is not supported yet.
        Self::validate_header(&microvm_state, app_version, SnapshotVersionSupport::Same)?;

        Ok(SnapshotImage {
            file,
            microvm_state,
        })
    }

    pub fn serialize_microvm(
        &mut self,
        header: SnapshotHdr,
        vm_state: VmState,
        vcpu_states: Vec<VcpuState>,
        device_states: Vec<MmioDeviceState>,
    ) -> Result<()> {
        self.microvm_state = MicrovmState {
            header,
            vm_state: Some(vm_state),
            vcpu_states,
            device_states,
        };
        let serialized_microvm =
            bincode::serialize(&self.microvm_state).map_err(Error::Serialize)?;
        let microvm_size = serialized_microvm.len() as u64;
        self.file.set_len(microvm_size).map_err(Error::Truncate)?;

        let mut reader_writer = SnapshotReaderWriter::new(&self.file, 0, microvm_size, true)
            .map_err(Error::CreateNew)?;
        reader_writer
            .write_all(serialized_microvm.as_slice())
            .map_err(Error::IO)?;

        Ok(())
    }

    pub fn vcpu_count(&self) -> u8 {
        self.microvm_state.vcpu_states.len() as u8
    }

    pub fn mem_size_mib(&self) -> usize {
        self.microvm_state.header.mem_size_mib as usize
    }

    pub fn validate_mem_file_size(&self, mem_file_path: &str) -> Result<()> {
        let metadata = std::fs::metadata(mem_file_path).map_err(|_| Error::MissingMemFile)?;
        if self.microvm_state.header.mem_size_mib << 20 != metadata.len() {
            Err(Error::MemfileSize)
        } else {
            Ok(())
        }
    }

    /// Checks that two versions are compatible.
    fn validate_version(
        version: Version,
        other_version: Version,
        support: SnapshotVersionSupport,
    ) -> std::result::Result<(), VersionError> {
        use SnapshotVersionSupport::*;

        match support {
            // Anything works.
            Any => Ok(()),

            // The major numbers must not differ by more than `delta`.
            Major(delta) => {
                if version.major_delta(other_version) > delta {
                    Err(VersionError::UnsupportedVersion)
                } else {
                    Ok(())
                }
            }
            // The major numbers must be the same, and the minor numbers must not differ by more
            // than `delta`.
            Minor(delta) => {
                if version.major != other_version.major
                    || version.minor_delta(other_version) > delta
                {
                    Err(VersionError::UnsupportedVersion)
                } else {
                    Ok(())
                }
            }

            // The major and minor numbers must be the same, and the build numbers must not differ
            // by more than `delta`.
            Build(delta) => {
                if version.major != other_version.major
                    || version.minor != other_version.minor
                    || version.build_delta(other_version) > delta
                {
                    Err(VersionError::UnsupportedVersion)
                } else {
                    Ok(())
                }
            }

            // The versions must be identical.
            Same => {
                if version != other_version {
                    Err(VersionError::UnsupportedVersion)
                } else {
                    Ok(())
                }
            }
        }
    }

    fn validate_header(
        microvm_state: &MicrovmState,
        current_app_version: Version,
        _version_support: SnapshotVersionSupport,
    ) -> Result<()> {
        if microvm_state.header.magic_id != SNAPSHOT_MAGIC {
            return Err(Error::BadMagicNumber);
        }
        Self::validate_version(
            SNAPSHOT_VERSION.into(),
            microvm_state.header.snapshot_version,
            // Cross-version deserialization is not supported yet.
            SnapshotVersionSupport::Same,
        )
        .map_err(|_| {
            Error::UnsupportedSnapshotVersion(microvm_state.header.snapshot_version.into())
        })?;
        Self::validate_version(
            current_app_version,
            microvm_state.header.app_version,
            // Cross-version deserialization is not supported yet.
            SnapshotVersionSupport::Same,
        )
        .map_err(|_| Error::UnsupportedAppVersion(microvm_state.header.app_version.into()))?;
        if microvm_state.header.vcpu_count != microvm_state.vcpu_states.len() as u8 {
            return Err(Error::BadVcpuCount);
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

    const APP_VER: u64 = 0xDEAD_BEEF_CAFE_BABE;

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
                && self.mem_size_mib == other.mem_size_mib
                && self.vcpu_count == other.vcpu_count
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

    fn build_valid_header(mem_size_mib: u64, vcpu_count: u8) -> SnapshotHdr {
        SnapshotHdr {
            magic_id: SNAPSHOT_MAGIC,
            snapshot_version: SNAPSHOT_VERSION.into(),
            app_version: APP_VER.into(),
            mem_size_mib,
            vcpu_count,
        }
    }

    #[test]
    fn test_header_serialization() {
        let header = build_valid_header(1, 1);
        let serialized_header = bincode::serialize(&header).unwrap();
        let deserialized_header =
            bincode::deserialize::<SnapshotHdr>(serialized_header.as_slice()).unwrap();
        assert_eq!(header, deserialized_header);
    }

    #[test]
    fn test_snapshot_getters() {
        let vcpu_count = 1u8;
        let mem_size_mib = 1u64;
        let (_, vcpu) = setup_vcpu();
        let vcpu_state = vcpu.save_state().unwrap();

        let header = build_valid_header(mem_size_mib, vcpu_count);
        let microvm_state = MicrovmState {
            header,
            vm_state: None,
            vcpu_states: vec![vcpu_state],
            device_states: vec![],
        };
        let file = NamedTempFile::new().unwrap().into_file();
        let fd = file.as_raw_fd();
        let image = SnapshotImage {
            file,
            microvm_state,
        };

        assert_eq!(image.vcpu_count(), vcpu_count);
        assert_eq!(image.mem_size_mib(), mem_size_mib as usize);
        assert_eq!(image.as_raw_fd(), fd);
    }

    fn make_snapshot(
        tmp_snapshot_path: &TempPath,
        vm_config: VmConfig,
        vm_state: VmState,
        vcpu_state: VcpuState,
        header: SnapshotHdr,
    ) -> String {
        let snapshot_path = tmp_snapshot_path.to_str().unwrap();
        let mut image =
            SnapshotImage::create_new(snapshot_path, vm_config, APP_VER.into()).unwrap();
        image
            .serialize_microvm(header, vm_state, vec![vcpu_state], vec![])
            .unwrap();
        snapshot_path.to_string()
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn test_snapshot_ser_deser() {
        let tmp_snapshot_path = NamedTempFile::new().unwrap().into_temp_path();
        let snapshot_path = tmp_snapshot_path.to_str().unwrap();

        let mem_size_mib: usize = 10;
        let vcpu_count: u8 = 1;

        let header = build_valid_header(mem_size_mib as u64, vcpu_count);

        let mut vm_config = VmConfig {
            vcpu_count: None,
            mem_size_mib: None,
            mem_file_path: None,
            cpu_template: None,
            ht_enabled: None,
            shared_mem: false,
        };

        let (vm, vcpu) = setup_vcpu();
        let vm_state = vm.save_state().unwrap();
        let vcpu_state = vcpu.save_state().unwrap();

        // Save snapshot.
        {
            let inaccessible_path = "/foo/bar";

            // Test error case: inaccessible snapshot path.
            let ret =
                SnapshotImage::create_new(inaccessible_path, vm_config.clone(), APP_VER.into());
            assert!(ret.is_err());
            assert_eq!(
                format!("{}", ret.err().unwrap()),
                "Failed to create new snapshot file: No such file or directory (os error 2)"
            );

            // Test error case: missing guest memory size.
            let ret = SnapshotImage::create_new(snapshot_path, vm_config.clone(), APP_VER.into());
            assert!(ret.is_err());
            assert_eq!(
                format!("{}", ret.err().unwrap()),
                "Missing guest memory size."
            );

            vm_config.mem_size_mib = Some(mem_size_mib);

            // Test error case: missing vCPU count.
            let ret = SnapshotImage::create_new(snapshot_path, vm_config.clone(), APP_VER.into());
            assert!(ret.is_err());
            assert_eq!(
                format!("{}", ret.err().unwrap()),
                "Missing number of vCPUs."
            );

            vm_config.vcpu_count = Some(vcpu_count);

            // Good case: snapshot is created.
            let ret = SnapshotImage::create_new(snapshot_path, vm_config.clone(), APP_VER.into());
            assert!(ret.is_ok());
            let mut image = ret.unwrap();

            assert!(image
                .serialize_microvm(
                    header.clone(),
                    vm_state.clone(),
                    vec![vcpu_state.clone()],
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
                "Failed to open snapshot file: Invalid argument (os error 22)"
            );

            {
                // Test error case: invalid magic number in header.
                let bad_snap = NamedTempFile::new().unwrap();
                let bad_snap_path = bad_snap.into_temp_path();
                let mut bad_header = header.clone();
                bad_header.magic_id = SNAPSHOT_MAGIC + 1;
                let bad_image_path = make_snapshot(
                    &bad_snap_path,
                    vm_config.clone(),
                    vm_state.clone(),
                    vcpu_state.clone(),
                    bad_header.clone(),
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
                // Test error case: invalid snapshot version number in header.
                let bad_snap = NamedTempFile::new().unwrap();
                let bad_snap_path = bad_snap.into_temp_path();
                let mut bad_header = header.clone();
                bad_header.snapshot_version.build += 1;
                let bad_image_path = make_snapshot(
                    &bad_snap_path,
                    vm_config.clone(),
                    vm_state.clone(),
                    vcpu_state.clone(),
                    bad_header.clone(),
                );
                let ret = SnapshotImage::open_existing(bad_image_path.as_str(), header.app_version);
                assert!(ret.is_err());
                assert_eq!(
                    format!("{}", ret.err().unwrap()),
                    format!(
                        "Unsupported snapshot version: {}.",
                        bad_header.snapshot_version
                    )
                );
            }

            {
                // Test error case: invalid app version number in header.
                let bad_snap = NamedTempFile::new().unwrap();
                let bad_snap_path = bad_snap.into_temp_path();
                let mut bad_header = header.clone();
                bad_header.app_version = 0.into();
                let bad_image_path = make_snapshot(
                    &bad_snap_path,
                    vm_config.clone(),
                    vm_state.clone(),
                    vcpu_state.clone(),
                    bad_header.clone(),
                );
                let ret = SnapshotImage::open_existing(bad_image_path.as_str(), header.app_version);
                assert!(ret.is_err());
                assert_eq!(
                    format!("{}", ret.err().unwrap()),
                    "Unsupported app version: 0.0.0."
                );
            }

            {
                // Test error case: incorrect vCPU count in header.
                let bad_snap = NamedTempFile::new().unwrap();
                let bad_snap_path = bad_snap.into_temp_path();
                let mut bad_header = header.clone();
                bad_header.vcpu_count = 2;
                let bad_image_path = make_snapshot(
                    &bad_snap_path,
                    vm_config.clone(),
                    vm_state.clone(),
                    vcpu_state.clone(),
                    bad_header.clone(),
                );
                let ret = SnapshotImage::open_existing(bad_image_path.as_str(), header.app_version);
                assert!(ret.is_err());
                assert_eq!(
                    format!("{}", ret.err().unwrap()),
                    "The vCPU count in the snapshot header does not match the number of vCPUs serialized in the snapshot."
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
            assert!(header.eq(&image.microvm_state.header));

            // Verify VM state deserialization.
            assert!(Some(vm_state).eq(&image.microvm_state.vm_state));

            // Verify vCPU state deserialization.
            assert!(vec![vcpu_state].eq(&image.microvm_state.vcpu_states));
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
            format!("{}", Error::UnsupportedAppVersion(0x0001_0002_0000_0003)),
            "Unsupported app version: 1.2.3."
        );
        assert_eq!(
            format!("{}", Error::UnsupportedSnapshotVersion(42)),
            "Unsupported snapshot version: 0.0.42."
        );
        assert_eq!(
            format!("{}", Error::Truncate(io::Error::from_raw_os_error(0))),
            format!("Failed to truncate snapshot file: {}", err0_str)
        );
    }

    #[test]
    fn test_validate_version() {
        use SnapshotVersionSupport::*;

        assert!(SnapshotImage::validate_version(1u64.into(), 2u64.into(), Any).is_ok());

        assert!(SnapshotImage::validate_version(
            (1u16, 0u16, 0u32).into(),
            (2u16, 0u16, 0u32).into(),
            Major(1)
        )
        .is_ok());
        assert!(SnapshotImage::validate_version(
            (2u16, 0u16, 0u32).into(),
            (1u16, 0u16, 0u32).into(),
            Major(1)
        )
        .is_ok());
        assert!(SnapshotImage::validate_version(
            (1u16, 0u16, 0u32).into(),
            (3u16, 0u16, 0u32).into(),
            Major(1)
        )
        .is_err());

        assert!(SnapshotImage::validate_version(
            (0u16, 1u16, 0u32).into(),
            (0u16, 2u16, 0u32).into(),
            Minor(1)
        )
        .is_ok());
        assert!(SnapshotImage::validate_version(
            (0u16, 2u16, 0u32).into(),
            (0u16, 1u16, 0u32).into(),
            Minor(1)
        )
        .is_ok());
        assert!(SnapshotImage::validate_version(
            (0u16, 1u16, 0u32).into(),
            (0u16, 3u16, 0u32).into(),
            Minor(1)
        )
        .is_err());

        assert!(SnapshotImage::validate_version(1.into(), 2.into(), Build(1)).is_ok());
        assert!(SnapshotImage::validate_version(2.into(), 1.into(), Build(1)).is_ok());
        assert!(SnapshotImage::validate_version(1.into(), 3.into(), Build(1)).is_err());

        assert!(SnapshotImage::validate_version(1.into(), 1.into(), Same).is_ok());
        assert!(SnapshotImage::validate_version(1.into(), 2.into(), Same).is_err());
    }
}
