// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Currently only supports X86_64.
#![cfg(target_arch = "x86_64")]

// Do not allow warnings. If any of our structures become FFI-unsafe we want to error.
#![deny(warnings)]

// Allow casting to a more-strictly-aligned pointer.
#![allow(clippy::cast_ptr_alignment)]

extern crate kvm_bindings;

use std::fmt::{self, Display, Formatter};
use std::fs::{File, OpenOptions};
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::Path;
use std::ptr::null_mut;
use std::slice;
use std::{io, mem};

use libc;

use kvm::{CpuId, KvmMsrs};
use kvm_bindings::{
    kvm_cpuid_entry2, kvm_debugregs, kvm_lapic_state, kvm_mp_state, kvm_msr_entry, kvm_regs,
    kvm_sregs, kvm_vcpu_events, kvm_xcrs, kvm_xsave,
};

use vmm_config::machine_config::VmConfig;
use vstate::{VcpuState, VmState};

pub(super) const SNAPSHOT_MAGIC: u64 = 0xEDA3_25D9_EDA3_25D9;

// Snapshot flags.
const VM_STATE_PRESENT: u64 = 1;
const VCPU_STATE_PRESENT: u64 = 1 << 1;
//const MEM_STATE_PRESENT: u64 = 1 << 2;

// Necessary flags to consider serialization complete.
const SERIALIZATION_COMPLETE: u64 = VM_STATE_PRESENT | VCPU_STATE_PRESENT;

//   Snapshot layout:       Offset
//   +-------------------+  0
//   |  Snapshot header  |
//   +-------------------+  Snapshot header size
//   |    vcpu 0 state   |
//   |        ...        |
//   |    vcpu N state   |
//   +-------------------+  prev + (num_vcpus * sizeof(VcpuRegs))
//   |   devices state   |
//   +-------------------+  prev + sizeof(devices_state) // TODO
//   |      Padding      |
//   +-------------------+  Linux page size alignment (multiple of system page size)
//   |    Guest memory   |
//   +-------------------+  prev + Guest memory size

/// Snapshot related errors.
#[derive(Debug)]
pub enum Error {
    CreateNew(io::Error),
    InvalidVcpuIndex,
    InvalidFileType,
    InvalidSnapshot,
    InvalidSnapshotSize,
    MissingVcpuNum,
    MissingMemSize,
    Mmap(io::Error),
    Munmap(io::Error),
    MsyncHeader(io::Error),
    OpenExisting(io::Error),
    Truncate(io::Error),
    VcpusNotSerialized,
    VmNotSerialized,
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use self::Error::*;
        match *self {
            CreateNew(ref e) => write!(f, "Failed to create new snapshot file: {}", e),
            InvalidVcpuIndex => write!(f, "Invalid vCPU index"),
            InvalidFileType => write!(f, "Invalid snapshot file type"),
            InvalidSnapshot => write!(f, "Invalid snapshot file"),
            InvalidSnapshotSize => write!(f, "Invalid snapshot file size"),
            MissingVcpuNum => write!(f, "Missing number of vCPUs"),
            MissingMemSize => write!(f, "Missing guest memory size"),
            Mmap(ref e) => write!(f, "Failed to map memory: {}", e),
            Munmap(ref e) => write!(f, "Failed to unmap memory: {}", e),
            MsyncHeader(ref e) => write!(f, "Failed to synchronize snapshot header: {}", e),
            OpenExisting(ref e) => write!(f, "Failed to open snapshot file: {}", e),
            Truncate(ref e) => write!(f, "Failed to truncate snapshot file: {}", e),
            VcpusNotSerialized => write!(f, "vCPUs not serialized in the snapshot"),
            VmNotSerialized => write!(f, "VM state not serialized in the snapshot"),
        }
    }
}

type Result<T> = std::result::Result<T, Error>;

/// Minimal `Option` implementation that is FFI-safe.
#[repr(C)]
enum COption<T> {
    /// No value
    CNone,
    /// Some value `T`
    CSome(T),
}

impl<T> COption<T> {
    #[inline]
    pub fn is_some(&self) -> bool {
        match *self {
            COption::CSome(_) => true,
            COption::CNone => false,
        }
    }

    #[inline]
    pub fn is_none(&self) -> bool {
        !self.is_some()
    }

    #[inline]
    pub fn as_ref(&self) -> Option<&T> {
        match *self {
            COption::CSome(ref x) => Some(x),
            COption::CNone => None,
        }
    }
}

// Declare this FFI extern functions that take pointers to `VcpuRegs` and `SnapshotHdr` to
// force the compiler to check whether they are FFI-safe and warn us if their layout is not fixed.
#[allow(dead_code)]
extern "C" {
    fn test_vcpu_regs(ptr: *const VcpuRegs);
    fn test_header(ptr: *const SnapshotHdr);
}

/** The header of VCPU state. */
#[repr(C)]
struct VcpuRegs {
    /** KVM multi-processor state. */
    mp_state: kvm_mp_state,

    /** KVM general purpose register state. */
    regs: kvm_regs,

    /** KVM segment register state. */
    sregs: kvm_sregs,

    /** KVM FPU state. */
    xsave: kvm_xsave,

    /** KVM eXtended Control Register state. */
    xcrs: kvm_xcrs,

    /** KVM debug register state. */
    debugregs: kvm_debugregs,

    /** KVM local APIC state. */
    lapic: kvm_lapic_state,

    /** KVM VCPU events. */
    vcpu_events: kvm_vcpu_events,
}

/** The header of a Firecracker snapshot image. */
#[repr(C)]
pub struct SnapshotHdr {
    /** SNAPSHOT_IMAGE_MAGIC */
    magic_id: u64,

    /** Size of file. */
    file_size: usize,

    mapping_size: usize,

    /** Flags indicating which fields are in this header */
    flags: u64,

    /** Number of vcpu to start. */
    vcpu_count: u8,

    /** Offset from start of file to the serialized vcpus. */
    vcpus_offset: u32,
    nmsrs: u32,
    ncpuids: u32,

    /** The memory size in MiB. */
    mem_size_mib: usize,

    /** Offset from start to first byte of memory. */
    memory_offset: usize,

    kvm_vm_state: COption<VmState>,
    //    /**
    //     * The value of the clock.  This is a monotonic clock that begins at
    //     * zero and increments whenever the image is running.  This clock will
    //     * stop incrementing when the image is paused.  This clock is used to
    //     * drive timers.
    //     */
    //    vm_clock_usec: u64,
    //
    //    /** The number of timers in this image. */
    //    num_timers: u32,
    //
    //    /** Offset from the start of file to the timers array. */
    //    timers_off: u32,
}

pub struct SnapshotImage {
    file: File,
    // Static ref to mapped header. Will live as long as the object, will be unmapped
    // in destructor.
    // Not accessible from outside this object.
    // TODO: wrap it in a struct that mmaps on ctor and munmaps on dtor.
    header: &'static mut SnapshotHdr,
    // Specifies whether this snapshot is read-only or should persist state.
    shared_mapping: bool,
}

impl SnapshotImage {
    fn mmap_region(
        file: &File,
        offset: usize,
        size: usize,
        shared_mapping: bool,
    ) -> std::io::Result<*mut u8> {
        let flags = libc::MAP_NORESERVE
            | if shared_mapping {
                libc::MAP_SHARED
            } else {
                libc::MAP_PRIVATE
            };
        // Safe because we are checking the return value and also unmapping in destructor.
        let addr = unsafe {
            libc::mmap(
                null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                file.as_raw_fd(),
                offset as i64,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        info!(
            "mapped addr {:x} at offset {} of size {}",
            addr as u64, offset, size
        );
        Ok(addr as *mut u8)
    }

    fn msync_region(addr: *mut u8, size: usize) -> std::io::Result<()> {
        // Safe because we check the return value.
        let ret = unsafe { libc::msync(addr as *mut libc::c_void, size, libc::MS_SYNC) };
        if ret == -1 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    fn munmap_region(addr: *mut u8, size: usize) -> io::Result<()> {
        // This is safe because we mmap the area at addr ourselves, and nobody
        // else is holding a reference to it.
        let ret = unsafe { libc::munmap(addr as *mut libc::c_void, size) };
        if ret == -1 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    fn validate_header(
        header: &SnapshotHdr,
        header_size: usize,
        file_size: usize,
        nmsrs: u32,
        ncpuids: u32,
    ) -> Result<()> {
        let computed_mapping_size =
            header_size + (Self::serialized_vcpu_size(nmsrs, ncpuids) * header.vcpu_count as usize);
        if header.magic_id != SNAPSHOT_MAGIC
            || header.vcpus_offset as usize != header_size
            || header.mapping_size < computed_mapping_size
            || header.flags & SERIALIZATION_COMPLETE != SERIALIZATION_COMPLETE
            || header.nmsrs != nmsrs
            || header.ncpuids != ncpuids
        {
            return Err(Error::InvalidSnapshot);
        }
        if header.file_size != file_size {
            return Err(Error::InvalidSnapshotSize);
        }
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        // Memory has to be mapped at a page boundary.
        let computed_memory_offset = (((header.mapping_size - 1) / page_size) + 1) * page_size;
        let computed_file_size = computed_memory_offset + (header.mem_size_mib << 20);
        if header.memory_offset != computed_memory_offset
            || file_size != computed_file_size
            || header.kvm_vm_state.is_none()
        {
            return Err(Error::InvalidSnapshot);
        }
        Ok(())
    }

    pub fn create_new<P: AsRef<Path>>(
        path: P,
        vm_cfg: VmConfig,
        nmsrs: u32,
        ncpuids: u32,
    ) -> Result<SnapshotImage> {
        let vcpu_count = *vm_cfg.vcpu_count.as_ref().ok_or(Error::MissingVcpuNum)?;
        let mem_size_mib = *vm_cfg.mem_size_mib.as_ref().ok_or(Error::MissingMemSize)?;

        let header_size = mem::size_of::<SnapshotHdr>();
        let mapping_size =
            header_size + (Self::serialized_vcpu_size(nmsrs, ncpuids) * vcpu_count as usize);

        // TODO: add validation for same page-size across runs.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        // Memory has to be mapped at a page boundary.
        let memory_offset = (((mapping_size - 1) / page_size) + 1) * page_size;
        let mem_size = mem_size_mib << 20;
        let file_size = memory_offset + mem_size;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())
            .map_err(Error::CreateNew)?;
        file.set_len(file_size as u64).map_err(Error::Truncate)?;

        // Any changes to the mapping should be visible on the snapshot file.
        let shared = true;

        let addr = Self::mmap_region(&file, 0, mapping_size, shared).map_err(Error::Mmap)?;

        let header = unsafe { &mut *(addr as *mut SnapshotHdr) };
        // Create and write the header in the mapping.
        *header = SnapshotHdr {
            magic_id: SNAPSHOT_MAGIC,
            file_size,
            mapping_size,
            flags: 0,
            vcpu_count,
            vcpus_offset: header_size as u32,
            nmsrs,
            ncpuids,
            mem_size_mib,
            memory_offset,
            kvm_vm_state: COption::CNone,
        };

        // VM and VCPU state still needed to make snapshot complete.
        Ok(SnapshotImage {
            file,
            header,
            shared_mapping: shared,
        })
    }

    pub fn open_existing<P: AsRef<Path>>(
        path: P,
        nmsrs: u32,
        ncpuids: u32,
    ) -> Result<SnapshotImage> {
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

        let file_size = metadata.len() as usize;
        let header_size = mem::size_of::<SnapshotHdr>();

        if file_size < header_size {
            return Err(Error::InvalidSnapshotSize);
        }

        // No changes should persist on the snapshot file.
        let shared = false;

        // Map the header region.
        let addr = Self::mmap_region(&file, 0, header_size, shared).map_err(Error::Mmap)?;
        let header = unsafe { &*(addr as *const SnapshotHdr) };

        if let Err(e) = Self::validate_header(header, header_size, file_size, nmsrs, ncpuids) {
            if let Err(err) = Self::munmap_region(addr, header_size) {
                error!("failed to munmap on error: {}", err);
            }
            return Err(e);
        }

        let mapping_size = header.mapping_size;
        // Unmap the header region.
        Self::munmap_region(addr, header_size).map_err(Error::Munmap)?;

        // Map the header + vcpus region.
        let addr = Self::mmap_region(&file, 0, mapping_size, shared).map_err(Error::Mmap)?;
        let header = unsafe { &mut *(addr as *mut SnapshotHdr) };

        Ok(SnapshotImage {
            file,
            header,
            shared_mapping: shared,
        })
    }

    pub fn can_deserialize(&self) -> Result<()> {
        if self.header.flags & VM_STATE_PRESENT == 0 {
            Err(Error::VmNotSerialized)
        } else if self.header.flags & VCPU_STATE_PRESENT == 0 {
            Err(Error::VcpusNotSerialized)
        } else {
            Ok(())
        }
    }

    pub fn kvm_vm_state(&self) -> Option<&VmState> {
        self.header.kvm_vm_state.as_ref()
    }

    pub fn set_kvm_vm_state(&mut self, kvm_vm_state: VmState) {
        self.header.kvm_vm_state = COption::CSome(kvm_vm_state);
        self.header.flags |= VM_STATE_PRESENT;
    }

    // TODO: maybe have this logic in `kvm-ioctls` crate as a `serialize()` function.
    fn serialized_vcpu_size(nmsrs: u32, ncpuids: u32) -> usize {
        // vcpu registers
        mem::size_of::<VcpuRegs>() +
        // sizeof msr entries
        nmsrs as usize * mem::size_of::<kvm_msr_entry>() +
        // sizeof cpuid entries
        ncpuids as usize * mem::size_of::<kvm_cpuid_entry2>()
    }

    // Returns the addresses for (vcpu_regs, msrs, cpuid).
    fn vcpu_offsets(&mut self, vcpu_index: usize) -> (*mut u8, *mut u8, *mut u8) {
        #[allow(clippy::ptr_offset_with_cast)]
        let vcpu_regs_offset = unsafe {
            (self.header as *mut SnapshotHdr as *mut u8)
                .offset(self.header.vcpus_offset as isize)
                .offset(
                    (Self::serialized_vcpu_size(self.header.nmsrs, self.header.ncpuids)
                        * vcpu_index) as isize,
                )
        };
        #[allow(clippy::ptr_offset_with_cast)]
        let msrs_offset = unsafe { vcpu_regs_offset.offset(mem::size_of::<VcpuRegs>() as isize) };
        #[allow(clippy::ptr_offset_with_cast)]
        let cpuid_offset = unsafe {
            msrs_offset
                .offset(self.header.nmsrs as isize * mem::size_of::<kvm_msr_entry>() as isize)
        };

        (vcpu_regs_offset, msrs_offset, cpuid_offset)
    }

    pub fn serialize_vcpu(&mut self, vcpu_index: usize, vcpu_state: Box<VcpuState>) -> Result<()> {
        if vcpu_index >= self.header.vcpu_count as usize {
            return Err(Error::InvalidVcpuIndex);
        }

        let msrs = vcpu_state.msrs.as_entries_slice();
        assert_eq!(self.header.nmsrs, msrs.len() as u32);

        let cpuid = vcpu_state.cpuid.as_entries_slice();
        assert_eq!(self.header.ncpuids, cpuid.len() as u32);

        // Get offsets.
        let (vcpu_regs_offset, msrs_offset, cpuid_offset) = self.vcpu_offsets(vcpu_index);

        // Serialize `repr(C)` VcpuRegs.
        {
            // Safe because we computed the size and `VcpuRegs` is `repr(C)`.
            let vcpu_regs = unsafe { &mut *(vcpu_regs_offset as *mut VcpuRegs) };
            vcpu_regs.debugregs = vcpu_state.debug_regs;
            vcpu_regs.lapic = vcpu_state.lapic;
            vcpu_regs.mp_state = vcpu_state.mp_state;
            vcpu_regs.regs = vcpu_state.regs;
            vcpu_regs.sregs = vcpu_state.sregs;
            vcpu_regs.vcpu_events = vcpu_state.vcpu_events;
            vcpu_regs.xcrs = vcpu_state.xcrs;
            vcpu_regs.xsave = vcpu_state.xsave;
        }
        // Serialize kvm msrs.
        {
            let msrs_addr = msrs_offset as *mut kvm_msr_entry;
            let msrs_dest = unsafe { slice::from_raw_parts_mut(msrs_addr, msrs.len()) };
            msrs_dest.copy_from_slice(msrs);
        }
        // Serialize kvm cpuid.
        {
            let cpuid_addr = cpuid_offset as *mut kvm_cpuid_entry2;
            let cpuid_dest = unsafe { slice::from_raw_parts_mut(cpuid_addr, cpuid.len()) };
            cpuid_dest.copy_from_slice(cpuid);
        }

        // TODO: this doesn't actually guarantee _all_ vcpus have been serialized.
        self.header.flags |= VCPU_STATE_PRESENT;
        Ok(())
    }

    pub fn deser_vcpu(&mut self, vcpu_index: usize) -> Result<VcpuState> {
        if vcpu_index >= self.header.vcpu_count as usize {
            return Err(Error::InvalidVcpuIndex);
        }
        if self.header.flags & VCPU_STATE_PRESENT == 0 {
            return Err(Error::VcpusNotSerialized);
        }

        // Get offsets.
        let (vcpu_regs_offset, msrs_offset, cpuid_offset) = self.vcpu_offsets(vcpu_index);

        // Deserialize `repr(C)` components.
        // Safe because we computed the size and `VcpuRegs` is `repr(C)`.
        let vcpu_regs = unsafe { &mut *(vcpu_regs_offset as *mut VcpuRegs) };

        // Deserialize kvm msrs.
        let msrs = {
            let msrs_addr = msrs_offset as *mut kvm_msr_entry;
            let msr_entries =
                unsafe { slice::from_raw_parts(msrs_addr, self.header.nmsrs as usize) };
            KvmMsrs::from_entries(msr_entries)
        };
        // Deserialize kvm cpuid.
        let cpuid = {
            let cpuid_addr = cpuid_offset as *mut kvm_cpuid_entry2;
            let cpuid = unsafe { slice::from_raw_parts(cpuid_addr, self.header.ncpuids as usize) };
            CpuId::from_entries(cpuid)
        };

        Ok(VcpuState {
            cpuid,
            msrs,
            debug_regs: vcpu_regs.debugregs,
            lapic: vcpu_regs.lapic,
            mp_state: vcpu_regs.mp_state,
            regs: vcpu_regs.regs,
            sregs: vcpu_regs.sregs,
            vcpu_events: vcpu_regs.vcpu_events,
            xcrs: vcpu_regs.xcrs,
            xsave: vcpu_regs.xsave,
        })
    }

    pub fn sync_header(&mut self) -> Result<()> {
        // Sync header + vcpus.
        Self::msync_region(
            self.header as *mut SnapshotHdr as *mut u8,
            self.header.mapping_size,
        )
        .map_err(Error::MsyncHeader)
    }

    pub fn vcpu_count(&self) -> u8 {
        self.header.vcpu_count
    }

    pub fn mem_size_mib(&self) -> usize {
        self.header.mem_size_mib
    }

    pub fn memory_offset(&self) -> usize {
        self.header.memory_offset
    }

    pub fn is_shared_mapping(&self) -> bool {
        self.shared_mapping
    }
}

impl AsRawFd for SnapshotImage {
    fn as_raw_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}

impl Drop for SnapshotImage {
    fn drop(&mut self) {
        let addr = self.header as *mut SnapshotHdr as *mut u8;
        match Self::munmap_region(addr, self.header.mapping_size) {
            Ok(()) => (),
            Err(e) => error!(
                "failed to munmap(addr: {:?}, len: {}) : {}",
                addr, self.header.mapping_size, e
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate tempfile;

    use self::tempfile::tempfile;

    use std::fmt;
    use std::io::Write;

    use super::*;

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

    impl PartialEq for Error {
        fn eq(&self, other: &Self) -> bool {
            // Guard match with no wildcard to make sure we catch new enum variants.
            match self {
                Error::CreateNew(_)
                | Error::InvalidVcpuIndex
                | Error::InvalidFileType
                | Error::InvalidSnapshot
                | Error::InvalidSnapshotSize
                | Error::MissingVcpuNum
                | Error::MissingMemSize
                | Error::Mmap(_)
                | Error::Munmap(_)
                | Error::MsyncHeader(_)
                | Error::OpenExisting(_)
                | Error::Truncate(_)
                | Error::VcpusNotSerialized
                | Error::VmNotSerialized => (),
            };
            match (self, other) {
                (Error::CreateNew(_), Error::CreateNew(_)) => true,
                (Error::InvalidVcpuIndex, Error::InvalidVcpuIndex) => true,
                (Error::InvalidFileType, Error::InvalidFileType) => true,
                (Error::InvalidSnapshot, Error::InvalidSnapshot) => true,
                (Error::InvalidSnapshotSize, Error::InvalidSnapshotSize) => true,
                (Error::MissingVcpuNum, Error::MissingVcpuNum) => true,
                (Error::MissingMemSize, Error::MissingMemSize) => true,
                (Error::Mmap(_), Error::Mmap(_)) => true,
                (Error::Munmap(_), Error::Munmap(_)) => true,
                (Error::MsyncHeader(_), Error::MsyncHeader(_)) => true,
                (Error::OpenExisting(_), Error::OpenExisting(_)) => true,
                (Error::Truncate(_), Error::Truncate(_)) => true,
                (Error::VcpusNotSerialized, Error::VcpusNotSerialized) => true,
                (Error::VmNotSerialized, Error::VmNotSerialized) => true,
                _ => false,
            }
        }
    }

    #[test]
    fn test_coption() {
        let none: COption<bool> = COption::CNone;
        assert!(none.is_none());
        assert_eq!(none.as_ref(), None);

        let some = COption::CSome(true);
        assert!(some.is_some());
        assert_eq!(some.as_ref(), Some(&true));
    }

    #[test]
    fn test_mmap_msync_munmap() {
        let mut f = tempfile().expect("failed to create temp file");

        // Verify mmap errors.
        SnapshotImage::mmap_region(&f, 0, 0, true)
            .expect_err("mmap_region should have failed because size is 0");

        let control_slice = [1, 2, 3, 4];
        // Write some data in the file.
        f.write_all(&control_slice)
            .expect("failed to write to temp file");

        // Do the correct mapping this time.
        let mapping: *mut u8 = SnapshotImage::mmap_region(&f, 0, control_slice.len(), false)
            .expect("failed to mmap_region");
        // Build a slice from the mmap'ed memory.
        let slice = unsafe { std::slice::from_raw_parts_mut(mapping, control_slice.len()) };
        // Verify the mmap'ed contents match the contents written in the file.
        assert_eq!(slice, control_slice);

        // Verify msync works on the mapped memory.
        SnapshotImage::msync_region(mapping, control_slice.len()).expect("failed to msync_region");
        // Verify msync errors.
        SnapshotImage::msync_region((mapping as u64 + 1) as *mut u8, control_slice.len())
            .expect_err("msync_region should have failed");

        // Verify memory unmapping.
        SnapshotImage::munmap_region(mapping, control_slice.len())
            .expect("failed to munmap_region");
        // Verify munmap errors.
        SnapshotImage::munmap_region(mapping, 0)
            .expect_err("mmap_region should have failed because size is 0");
    }

    fn build_valid_header(nmsrs: u32, ncpuids: u32) -> (SnapshotHdr, usize, usize) {
        let vcpu_count = 1;
        let mem_size_mib = 1;

        let header_size = mem::size_of::<SnapshotHdr>();
        let mapping_size = header_size
            + (SnapshotImage::serialized_vcpu_size(nmsrs, ncpuids) * vcpu_count as usize);

        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        // Memory has to be mapped at a page boundary.
        let memory_offset = (((mapping_size - 1) / page_size) + 1) * page_size;
        let mem_size = mem_size_mib << 20;
        let file_size = memory_offset + mem_size;

        (
            SnapshotHdr {
                magic_id: SNAPSHOT_MAGIC,
                file_size,
                mapping_size,
                flags: SERIALIZATION_COMPLETE,
                vcpu_count,
                vcpus_offset: header_size as u32,
                nmsrs,
                ncpuids,
                mem_size_mib,
                memory_offset,
                kvm_vm_state: COption::CSome(Default::default()),
            },
            header_size,
            file_size,
        )
    }

    #[test]
    fn test_validate_header() {
        let nmsrs = 10;
        let ncpuids = 10;

        let (mut header, header_size, file_size) = build_valid_header(nmsrs, ncpuids);
        SnapshotImage::validate_header(&header, header_size, file_size, nmsrs, ncpuids)
            .expect("invalid header");

        // Validate magic id.
        header.magic_id = 0;
        assert_eq!(
            SnapshotImage::validate_header(&header, header_size, file_size, nmsrs, ncpuids),
            Err(Error::InvalidSnapshot)
        );

        // Validate vcpus_offset.
        let (mut header, header_size, file_size) = build_valid_header(nmsrs, ncpuids);
        header.vcpus_offset -= 1;
        assert_eq!(
            SnapshotImage::validate_header(&header, header_size, file_size, nmsrs, ncpuids),
            Err(Error::InvalidSnapshot)
        );

        // Validate mapping_size.
        let (mut header, header_size, file_size) = build_valid_header(nmsrs, ncpuids);
        header.mapping_size -= 1;
        assert_eq!(
            SnapshotImage::validate_header(&header, header_size, file_size, nmsrs, ncpuids),
            Err(Error::InvalidSnapshot)
        );

        // Validate flags.
        let (mut header, header_size, file_size) = build_valid_header(nmsrs, ncpuids);
        header.flags = 0;
        assert_eq!(
            SnapshotImage::validate_header(&header, header_size, file_size, nmsrs, ncpuids),
            Err(Error::InvalidSnapshot)
        );

        // Validate nmsrs.
        let (mut header, header_size, file_size) = build_valid_header(nmsrs, ncpuids);
        header.nmsrs -= 1;
        assert_eq!(
            SnapshotImage::validate_header(&header, header_size, file_size, nmsrs, ncpuids),
            Err(Error::InvalidSnapshot)
        );

        // Validate ncpuids.
        let (mut header, header_size, file_size) = build_valid_header(nmsrs, ncpuids);
        header.ncpuids -= 1;
        assert_eq!(
            SnapshotImage::validate_header(&header, header_size, file_size, nmsrs, ncpuids),
            Err(Error::InvalidSnapshot)
        );

        // Validate file_size.
        let (mut header, header_size, file_size) = build_valid_header(nmsrs, ncpuids);
        header.file_size -= 1;
        assert_eq!(
            SnapshotImage::validate_header(&header, header_size, file_size, nmsrs, ncpuids),
            Err(Error::InvalidSnapshotSize)
        );

        // Validate memory_offset.
        let (mut header, header_size, file_size) = build_valid_header(nmsrs, ncpuids);
        header.memory_offset -= 1;
        assert_eq!(
            SnapshotImage::validate_header(&header, header_size, file_size, nmsrs, ncpuids),
            Err(Error::InvalidSnapshot)
        );

        // Validate all offsets add up to file size.
        let (mut header, header_size, file_size) = build_valid_header(nmsrs, ncpuids);
        header.mem_size_mib += 1;
        assert_eq!(
            SnapshotImage::validate_header(&header, header_size, file_size, nmsrs, ncpuids),
            Err(Error::InvalidSnapshot)
        );

        // Validate kvm_vm_state.
        let (mut header, header_size, file_size) = build_valid_header(nmsrs, ncpuids);
        header.kvm_vm_state = COption::CNone;
        assert_eq!(
            SnapshotImage::validate_header(&header, header_size, file_size, nmsrs, ncpuids),
            Err(Error::InvalidSnapshot)
        );
    }

    #[test]
    fn test_snapshot_getters() {
        let nmsrs = 10;
        let ncpuids = 10;

        let vcpu_count = 1;
        let mem_size_mib = 1;

        let header_size = mem::size_of::<SnapshotHdr>();
        let mapping_size = header_size
            + (SnapshotImage::serialized_vcpu_size(nmsrs, ncpuids) * vcpu_count as usize);

        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        // Memory has to be mapped at a page boundary.
        let memory_offset = (((mapping_size - 1) / page_size) + 1) * page_size;
        let mem_size = mem_size_mib << 20;
        let file_size = memory_offset + mem_size;

        let mut header_ = SnapshotHdr {
            magic_id: SNAPSHOT_MAGIC,
            file_size,
            mapping_size,
            flags: SERIALIZATION_COMPLETE,
            vcpu_count,
            vcpus_offset: header_size as u32,
            nmsrs,
            ncpuids,
            mem_size_mib,
            memory_offset,
            kvm_vm_state: COption::CSome(Default::default()),
        };

        // Turn it into pointer then back to reference to get static ref.
        let header_ptr: *mut SnapshotHdr = &mut header_;
        let header: &mut SnapshotHdr = unsafe { &mut (*header_ptr) };

        let file = tempfile().expect("failed to create temp file");
        let file_rawfd = file.as_raw_fd();
        let image = SnapshotImage {
            file,
            header,
            shared_mapping: true,
        };

        assert!(image.is_shared_mapping());
        assert_eq!(image.vcpu_count(), vcpu_count);
        assert_eq!(image.mem_size_mib(), mem_size_mib);
        assert_eq!(image.memory_offset(), memory_offset);
        assert_eq!(image.as_raw_fd(), file_rawfd);
    }

    #[test]
    fn test_snapshot_ser_deser() {
        let nmsrs = 10;
        let ncpuids = 10;
        let vcpu_count = 2;
        let mem_size_mib = 128;
        let vcpu_state = || VcpuState {
            cpuid: CpuId::new(ncpuids as usize),
            msrs: KvmMsrs::new(nmsrs as usize),
            debug_regs: Default::default(),
            lapic: Default::default(),
            mp_state: Default::default(),
            regs: Default::default(),
            sregs: Default::default(),
            vcpu_events: Default::default(),
            xcrs: Default::default(),
            xsave: Default::default(),
        };
        let snapshot_path = "foo.image";

        ////////////////////////////////////////////
        // Test serialization
        ////////////////////////////////////////////
        {
            // Test create errors.
            match SnapshotImage::create_new(
                snapshot_path,
                VmConfig {
                    vcpu_count: None, // Error case.
                    mem_size_mib: Some(mem_size_mib),
                    ht_enabled: None,
                    cpu_template: None,
                },
                nmsrs,
                ncpuids,
            ) {
                Err(Error::MissingVcpuNum) => (),
                _ => assert!(false),
            };
            match SnapshotImage::create_new(
                snapshot_path,
                VmConfig {
                    vcpu_count: Some(vcpu_count),
                    mem_size_mib: None, // Error case.
                    ht_enabled: None,
                    cpu_template: None,
                },
                nmsrs,
                ncpuids,
            ) {
                Err(Error::MissingMemSize) => (),
                _ => assert!(false),
            };

            // Test successful create.
            let mut image = SnapshotImage::create_new(
                snapshot_path,
                VmConfig {
                    vcpu_count: Some(vcpu_count),
                    mem_size_mib: Some(mem_size_mib),
                    ht_enabled: None,
                    cpu_template: None,
                },
                nmsrs,
                ncpuids,
            )
            .expect("failed to create new snapshot image");

            // Test complete serialization checks.
            assert_eq!(image.can_deserialize(), Err(Error::VmNotSerialized));
            image.set_kvm_vm_state(VmState::default());

            assert_eq!(image.can_deserialize(), Err(Error::VcpusNotSerialized));

            // Test vcpu serialization.
            // Invalid vcpu index.
            assert_eq!(
                image.serialize_vcpu((vcpu_count + 1) as usize, Box::new(vcpu_state())),
                Err(Error::InvalidVcpuIndex)
            );
            // Success.
            for i in 0..vcpu_count {
                assert!(image
                    .serialize_vcpu(i as usize, Box::new(vcpu_state()))
                    .is_ok());
            }

            // Serialization should be complete.
            assert!(image.can_deserialize().is_ok());

            // Syncing to snapshot file should work.
            assert!(image.sync_header().is_ok());
        }
        ////////////////////////////////////////////
        // Test deserialization
        ////////////////////////////////////////////
        {
            let invalid_path = "invalid_path";
            // Test invalid path.
            assert_eq!(
                SnapshotImage::open_existing(invalid_path, nmsrs, ncpuids).unwrap_err(),
                Error::OpenExisting(io::Error::from_raw_os_error(libc::ENOENT))
            );
            {
                let _file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(invalid_path)
                    .expect("failed to create file");
            }
            // Test invalid snapshot size.
            assert_eq!(
                SnapshotImage::open_existing(invalid_path, nmsrs, ncpuids).unwrap_err(),
                Error::InvalidSnapshotSize
            );
            std::fs::remove_file(invalid_path).expect("failed to delete invalid file");

            // Opening correct snapshot file should work.
            let mut image = SnapshotImage::open_existing(snapshot_path, nmsrs, ncpuids)
                .expect("failed to open existing snapshot");

            // Deserialization should be possible.
            assert!(image.can_deserialize().is_ok());

            // Check vm state deserialization. Does not impl PartialEq to be able to compare.
            assert!(image.kvm_vm_state().is_some());

            // Check vcpu deserialization.

            // Correct deser.
            for i in 0..vcpu_count {
                let state = image.deser_vcpu(i as usize).expect("failed to deser vcpu");
                let orig = vcpu_state();
                assert!(vcpu_states_eq(&orig, &state));
            }

            // Errors.
            {
                // Test invalid vcpu index.
                assert_eq!(
                    image.deser_vcpu((vcpu_count + 1) as usize).unwrap_err(),
                    Error::InvalidVcpuIndex
                );
                let flags = image.header.flags;
                image.header.flags = 0;
                // Test invalid vcpu index.
                assert_eq!(image.deser_vcpu(0).unwrap_err(), Error::VcpusNotSerialized);
                image.header.flags = flags;
            }
        }

        // Remove snapshot file.
        std::fs::remove_file(snapshot_path).expect("failed to delete snapshot");
    }

    fn vcpu_states_eq(one: &VcpuState, other: &VcpuState) -> bool {
        one.cpuid == other.cpuid
            && one.msrs == other.msrs
            && one.debug_regs == other.debug_regs
            && one.sregs == other.sregs
            && one.mp_state == other.mp_state
            && one.regs == other.regs
            && one.vcpu_events == other.vcpu_events
            && one.xcrs == other.xcrs
    }

    #[test]
    fn test_error_messages() {
        #[cfg(target_env = "musl")]
        let err0_str = "No error information (os error 0)";
        #[cfg(target_env = "gnu")]
        let err0_str = "Success (os error 0)";

        assert_eq!(
            format!("{}", Error::CreateNew(io::Error::from_raw_os_error(0))),
            format!("Failed to create new snapshot file: {}", err0_str)
        );
        assert_eq!(format!("{}", Error::InvalidVcpuIndex), "Invalid vCPU index");
        assert_eq!(
            format!("{}", Error::InvalidFileType),
            "Invalid snapshot file type"
        );
        assert_eq!(
            format!("{}", Error::InvalidSnapshot),
            "Invalid snapshot file"
        );
        assert_eq!(
            format!("{}", Error::InvalidSnapshotSize),
            "Invalid snapshot file size"
        );
        assert_eq!(
            format!("{}", Error::MissingVcpuNum),
            "Missing number of vCPUs"
        );
        assert_eq!(
            format!("{}", Error::MissingMemSize),
            "Missing guest memory size"
        );
        assert_eq!(
            format!("{}", Error::Mmap(io::Error::from_raw_os_error(0))),
            format!("Failed to map memory: {}", err0_str)
        );
        assert_eq!(
            format!("{}", Error::Munmap(io::Error::from_raw_os_error(0))),
            format!("Failed to unmap memory: {}", err0_str)
        );
        assert_eq!(
            format!("{}", Error::MsyncHeader(io::Error::from_raw_os_error(0))),
            format!("Failed to synchronize snapshot header: {}", err0_str)
        );
        assert_eq!(
            format!("{}", Error::OpenExisting(io::Error::from_raw_os_error(0))),
            format!("Failed to open snapshot file: {}", err0_str)
        );
        assert_eq!(
            format!("{}", Error::Truncate(io::Error::from_raw_os_error(0))),
            format!("Failed to truncate snapshot file: {}", err0_str)
        );
        assert_eq!(
            format!("{}", Error::VcpusNotSerialized),
            "vCPUs not serialized in the snapshot"
        );
        assert_eq!(
            format!("{}", Error::VmNotSerialized),
            "VM state not serialized in the snapshot"
        );
    }
}
