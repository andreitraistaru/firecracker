// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

#![deny(missing_docs)]

//! A safe wrapper around the kernel's KVM interface.

extern crate libc;
extern crate serde;

extern crate kvm_bindings;
#[macro_use]
extern crate sys_util;

mod cap;
mod device;
mod ioctl_defs;
/// Helper for dealing with KVM api structures
mod kvm_vec;

use std::fs::File;
use std::io;
use std::os::raw::*;
use std::os::unix::io::{AsRawFd, FromRawFd, RawFd};
use std::ptr::null_mut;
use std::result;

use libc::{open, EINVAL, O_CLOEXEC, O_RDWR};

use kvm_bindings::*;
use sys_util::EventFd;
use sys_util::{
    ioctl, ioctl_with_mut_ptr, ioctl_with_mut_ref, ioctl_with_ptr, ioctl_with_ref, ioctl_with_val,
};

/// Wrapper over possible Kvm method Result.
pub type Result<T> = result::Result<T, io::Error>;

pub use self::kvm_vec::*;
pub use cap::*;
use device::new_device;
pub use device::DeviceFd;
use ioctl_defs::*;
pub use kvm_bindings::KVM_API_VERSION;

// TODO: These should've been generated in x86-specific kvm_bindings.

/// Taken from Linux Kernel v4.14.13 (arch/x86/include/asm/kvm_host.h).
pub const MAX_KVM_CPUID_ENTRIES: usize = 80;
/// Maximum number of MSRs KVM supports (See arch/x86/kvm/x86.c).
pub const MAX_KVM_MSR_ENTRIES: usize = 256;

/// Taken from Linux Kernel v4.14.13 (arch/x86/include/uapi/asm/kvm.h).
pub const KVM_IRQCHIP_PIC_MASTER: u32 = 0;
/// Taken from Linux Kernel v4.14.13 (arch/x86/include/uapi/asm/kvm.h).
pub const KVM_IRQCHIP_PIC_SLAVE: u32 = 1;
/// Taken from Linux Kernel v4.14.13 (arch/x86/include/uapi/asm/kvm.h).
pub const KVM_IRQCHIP_IOAPIC: u32 = 2;
/// Taken from Linux Kernel v4.14.13 (arch/x86/include/uapi/asm/kvm.h).
pub const KVM_NR_IRQCHIPS: u32 = 3;

/// A wrapper around opening and using `/dev/kvm`.
///
/// The handle is used to issue system ioctls.
pub struct Kvm {
    kvm: File,
}

impl Kvm {
    /// Opens `/dev/kvm/` and returns a `Kvm` object on success.
    pub fn new() -> Result<Self> {
        // Open `/dev/kvm` using `O_CLOEXEC` flag.
        let fd = Self::open_with_cloexec(true)?;
        // Safe because we verify that ret is valid and we own the fd.
        Ok(unsafe { Self::new_with_fd_number(fd) })
    }

    /// Creates a new Kvm object assuming `fd` represents an existing open file descriptor
    /// associated with `/dev/kvm`.
    ///
    /// # Arguments
    ///
    /// * `fd` - File descriptor for `/dev/kvm`.
    ///
    pub unsafe fn new_with_fd_number(fd: RawFd) -> Self {
        Kvm {
            kvm: File::from_raw_fd(fd),
        }
    }

    /// Opens `/dev/kvm` and returns the fd number on success.
    ///
    /// # Arguments
    ///
    /// * `close_on_exec`: If true opens `/dev/kvm` using the `O_CLOEXEC` flag.
    ///
    pub fn open_with_cloexec(close_on_exec: bool) -> Result<RawFd> {
        let open_flags = O_RDWR | if close_on_exec { O_CLOEXEC } else { 0 };
        // Safe because we give a constant nul-terminated string and verify the result.
        let ret = unsafe { open("/dev/kvm\0".as_ptr() as *const c_char, open_flags) };
        if ret < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(ret)
        }
    }

    /// Returns the KVM API version.
    pub fn get_api_version(&self) -> i32 {
        // Safe because we know that our file is a KVM fd and that the request is one of the ones
        // defined by kernel.
        unsafe { ioctl(self, KVM_GET_API_VERSION()) }
    }

    /// Query the availability of a particular kvm capability.
    /// Returns 0 if the capability is not available and > 0 otherwise.
    ///
    fn check_extension_int(&self, c: Cap) -> i32 {
        // Safe because we know that our file is a KVM fd and that the extension is one of the ones
        // defined by kernel.
        unsafe { ioctl_with_val(self, KVM_CHECK_EXTENSION(), c as c_ulong) }
    }

    /// Checks if a particular `Cap` is available.
    ///
    /// According to the KVM API doc, KVM_CHECK_EXTENSION returns "0 if unsupported; 1 (or some
    /// other positive integer) if supported".
    ///
    /// # Arguments
    ///
    /// * `c` - KVM capability.
    ///
    pub fn check_extension(&self, c: Cap) -> bool {
        self.check_extension_int(c) >= 1
    }

    /// Gets the size of the mmap required to use vcpu's `kvm_run` structure.
    pub fn get_vcpu_mmap_size(&self) -> Result<usize> {
        // Safe because we know that our file is a KVM fd and we verify the return result.
        let res = unsafe { ioctl(self, KVM_GET_VCPU_MMAP_SIZE()) };
        if res > 0 {
            Ok(res as usize)
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Gets the recommended number of VCPUs per VM.
    ///
    pub fn get_nr_vcpus(&self) -> usize {
        let x = self.check_extension_int(Cap::NrVcpus);
        if x > 0 {
            x as usize
        } else {
            4
        }
    }

    /// Gets the maximum allowed memory slots per VM.
    ///
    /// KVM reports the number of available memory slots (`KVM_CAP_NR_MEMSLOTS`)
    /// using the extension interface.  Both x86 and s390 implement this, ARM
    /// and powerpc do not yet enable it.
    /// Default to 32 when `KVM_CAP_NR_MEMSLOTS` is not implemented.
    ///
    pub fn get_nr_memslots(&self) -> usize {
        let x = self.check_extension_int(Cap::NrMemslots);
        if x > 0 {
            x as usize
        } else {
            32
        }
    }

    /// Gets the recommended maximum number of VCPUs per VM.
    ///
    pub fn get_max_vcpus(&self) -> usize {
        match self.check_extension_int(Cap::MaxVcpus) {
            0 => self.get_nr_vcpus(),
            x => x as usize,
        }
    }

    /// X86 specific call to get the system supported CPUID values.
    ///
    /// # Arguments
    ///
    /// * `max_entries_count` - Maximum number of CPUID entries. This function can return less than
    ///                         this when the hardware does not support so many CPUID entries.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn get_supported_cpuid(&self, max_entries_count: usize) -> Result<CpuId> {
        let mut cpuid = CpuId::new(max_entries_count);

        let ret = unsafe {
            // ioctl is unsafe. The kernel is trusted not to write beyond the bounds of the memory
            // allocated for the struct. The limit is read from nent, which is set to the allocated
            // size(max_entries_count) above.
            ioctl_with_mut_ptr(self, KVM_GET_SUPPORTED_CPUID(), cpuid.as_mut_ptr())
        };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }

        // TODO HACK! #86 until FAMstruct is integrated
        Ok(CpuId::from_entries(cpuid.as_entries_slice()))
    }

    /// X86 specific call to read list of MSRs available for VMs.
    ///
    /// See the documentation for `KVM_GET_MSR_INDEX_LIST`.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn get_msr_index_list(&self) -> Result<MsrList> {
        let mut indexes = MsrList::new(MAX_KVM_MSR_ENTRIES);
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_msr_list struct.
            ioctl_with_mut_ptr(self, KVM_GET_MSR_INDEX_LIST(), indexes.as_mut_ptr())
        };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(indexes)
    }

    /// X86 specific call to read list of MSRs available on this host.
    ///
    /// See the documentation for `KVM_GET_MSR_FEATURE_INDEX_LIST`.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn get_msr_feature_index_list(&self) -> Result<MsrList> {
        let mut indexes = MsrList::new(MAX_KVM_MSR_ENTRIES);
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_msr_list struct.
            ioctl_with_mut_ptr(self, KVM_GET_MSR_FEATURE_INDEX_LIST(), indexes.as_mut_ptr())
        };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(indexes)
    }

    /// Creates a VM fd using the KVM fd (`KVM_CREATE_VM`).
    ///
    /// A call to this function will also initialize the supported cpuid (`KVM_GET_SUPPORTED_CPUID`)
    /// and the size of the vcpu mmap area (`KVM_GET_VCPU_MMAP_SIZE`).
    ///
    pub fn create_vm(&self) -> Result<VmFd> {
        // Safe because we know kvm is a real kvm fd as this module is the only one that can make
        // Kvm objects.
        let ret = unsafe { ioctl(&self.kvm, KVM_CREATE_VM()) };
        if ret >= 0 {
            // Safe because we verify the value of ret and we are the owners of the fd.
            let vm_file = unsafe { File::from_raw_fd(ret) };
            let run_mmap_size = self.get_vcpu_mmap_size()?;
            Ok(VmFd {
                vm: vm_file,
                run_size: run_mmap_size,
            })
        } else {
            Err(io::Error::last_os_error())
        }
    }
}

impl AsRawFd for Kvm {
    fn as_raw_fd(&self) -> RawFd {
        self.kvm.as_raw_fd()
    }
}

/// An address either in programmable I/O space or in memory mapped I/O space.
pub enum IoeventAddress {
    /// Representation of an programmable I/O address.
    Pio(u64),
    /// Representation of an memory mapped I/O address.
    Mmio(u64),
}

/// Used in `VmFd::register_ioevent` to indicate that no datamatch is requested.
pub struct NoDatamatch;
impl Into<u64> for NoDatamatch {
    fn into(self) -> u64 {
        0
    }
}

/// A safe wrapper over the `kvm_run` struct.
///
/// The wrapper is needed for sending the pointer to `kvm_run` between
/// threads as raw pointers do not implement `Send` and `Sync`.
pub struct KvmRunWrapper {
    kvm_run_ptr: *mut u8,
    size: usize,
}

// Send and Sync aren't automatically inherited for the raw address pointer.
// Accessing that pointer is only done through the stateless interface which
// allows the object to be shared by multiple threads without a decrease in
// safety.
unsafe impl Send for KvmRunWrapper {}
unsafe impl Sync for KvmRunWrapper {}

impl KvmRunWrapper {
    /// Maps the first `size` bytes of the given `fd`.
    ///
    /// # Arguments
    /// * `fd` - File descriptor to mmap from.
    /// * `size` - Size of memory region in bytes.
    pub fn from_fd(fd: &AsRawFd, size: usize) -> Result<KvmRunWrapper> {
        // This is safe because we are creating a mapping in a place not already used by any other
        // area in this process.
        let addr = unsafe {
            libc::mmap(
                null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        Ok(KvmRunWrapper {
            kvm_run_ptr: addr as *mut u8,
            size,
        })
    }

    /// Returns a mutable reference to `kvm_run`.
    ///
    #[allow(clippy::mut_from_ref)]
    pub fn as_mut_ref(&self) -> &mut kvm_run {
        // Safe because we know we mapped enough memory to hold the `kvm_run` struct because the
        // kernel told us how large it was.
        #[allow(clippy::cast_ptr_alignment)]
        unsafe {
            &mut *(self.kvm_run_ptr as *mut kvm_run)
        }
    }
}

impl Drop for KvmRunWrapper {
    fn drop(&mut self) {
        // This is safe because we mmap the area at addr ourselves, and nobody
        // else is holding a reference to it.
        let _ = unsafe { libc::munmap(self.kvm_run_ptr as *mut libc::c_void, self.size) };
    }
}

/// A wrapper around creating and using a VM.
pub struct VmFd {
    vm: File,
    run_size: usize,
}

impl VmFd {
    /// Creates/modifies a guest physical memory slot using `KVM_SET_USER_MEMORY_REGION`.
    ///
    /// See the documentation on the `KVM_SET_USER_MEMORY_REGION` ioctl.
    ///
    pub fn set_user_memory_region(
        &self,
        user_memory_region: kvm_userspace_memory_region,
    ) -> Result<()> {
        let ret =
            unsafe { ioctl_with_ref(self, KVM_SET_USER_MEMORY_REGION(), &user_memory_region) };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Sets the address of the three-page region in the VM's address space.
    ///
    /// See the documentation on the `KVM_SET_TSS_ADDR` ioctl.
    ///
    /// # Arguments
    ///
    /// * `offset` - Physical address of a three-page region in the guest's physical address space.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn set_tss_address(&self, offset: usize) -> Result<()> {
        // Safe because we know that our file is a VM fd and we verify the return result.
        let ret = unsafe { ioctl_with_val(self, KVM_SET_TSS_ADDR(), offset as c_ulong) };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Creates an in-kernel interrupt controller.
    ///
    /// See the documentation on the `KVM_CREATE_IRQCHIP` ioctl.
    ///
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64"
    ))]
    pub fn create_irq_chip(&self) -> Result<()> {
        // Safe because we know that our file is a VM fd and we verify the return result.
        let ret = unsafe { ioctl(self, KVM_CREATE_IRQCHIP()) };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Sets the level on the given irq to 1 if `active` is true, and 0 otherwise.
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64"
    ))]
    pub fn set_irq_line(&self, irq: u32, active: bool) -> Result<()> {
        let mut irq_level = kvm_irq_level::default();
        irq_level.__bindgen_anon_1.irq = irq;
        irq_level.level = if active { 1 } else { 0 };

        // Safe because we know that our file is a VM fd, we know the kernel will only read the
        // correct amount of memory from our pointer, and we verify the return result.
        let ret = unsafe { ioctl_with_ref(self, KVM_IRQ_LINE(), &irq_level) };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Creates a PIT as per the `KVM_CREATE_PIT2` ioctl.
    ///
    /// Note that this call can only succeed after a call to `Vm::create_irq_chip`.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn create_pit2(&self, pit_config: kvm_pit_config) -> Result<()> {
        // Safe because we know that our file is a VM fd, we know the kernel will only read the
        // correct amount of memory from our pointer, and we verify the return result.
        let ret = unsafe { ioctl_with_ref(self, KVM_CREATE_PIT2(), &pit_config) };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Registers an event to be signaled whenever a certain address is written to.
    ///
    /// # Arguments
    ///
    /// * `evt` - EventFd which will be signaled. When signaling, the usual `vmexit` to userspace
    ///           is prevented.
    /// * `addr` - Address being written to.
    /// * `datamatch` - Limits signaling `evt` to only the cases where the value being written is
    ///                 equal to this parameter. The size of `datamatch` is important and it must
    ///                 match the expected size of the guest's write.
    ///
    pub fn register_ioevent<T: Into<u64>>(
        &self,
        evt: &EventFd,
        addr: &IoeventAddress,
        datamatch: T,
    ) -> Result<()> {
        let mut flags = 0;
        if std::mem::size_of::<T>() > 0 {
            flags |= 1 << kvm_ioeventfd_flag_nr_datamatch
        }
        if let IoeventAddress::Pio(_) = *addr {
            flags |= 1 << kvm_ioeventfd_flag_nr_pio
        }

        let ioeventfd = kvm_ioeventfd {
            datamatch: datamatch.into(),
            len: std::mem::size_of::<T>() as u32,
            addr: match addr {
                IoeventAddress::Pio(ref p) => *p as u64,
                IoeventAddress::Mmio(ref m) => *m,
            },
            fd: evt.as_raw_fd(),
            flags,
            ..Default::default()
        };
        // Safe because we know that our file is a VM fd, we know the kernel will only read the
        // correct amount of memory from our pointer, and we verify the return result.
        let ret = unsafe { ioctl_with_ref(self, KVM_IOEVENTFD(), &ioeventfd) };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Gets the bitmap of pages dirtied since the last call of this function.
    ///
    /// Leverages the dirty page logging feature in KVM. As a side-effect, this also resets the
    /// bitmap inside the kernel. Because of this, this function is only usable from one place in
    /// the code. Right now, that's just the dirty page count metrics, but if it's needed in other
    /// places, then some scheme for buffering the results is needed.
    ///
    /// # Arguments
    ///
    /// * `slot` - Guest memory slot identifier.
    /// * `memory_size` - Size of the memory region.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn get_dirty_log(&self, slot: u32, memory_size: usize) -> Result<Vec<u64>> {
        // Compute the length of the bitmap needed for all dirty pages in one memory slot.
        // One memory page is 4KiB (4096 bits) and KVM_GET_DIRTY_LOG returns one dirty bit for
        // each page.
        let page_size = 4 << 10;

        let div_round_up = |dividend, divisor| (dividend + divisor - 1) / divisor;
        // For ease of access we are saving the bitmap in a u64 vector. If `mem_size` is not a
        // multiple of `page_size * 64`, we use a ceil function to compute the capacity of the
        // bitmap. This way we make sure that all dirty pages are added to the bitmap.
        let bitmap_size = div_round_up(memory_size, page_size * 64);
        let mut bitmap = vec![0; bitmap_size];
        let b_data = bitmap.as_mut_ptr() as *mut c_void;
        let dirtylog = kvm_dirty_log {
            slot,
            padding1: 0,
            __bindgen_anon_1: kvm_dirty_log__bindgen_ty_1 {
                dirty_bitmap: b_data,
            },
        };
        // Safe because we know that our file is a VM fd, and we know that the amount of memory
        // we allocated for the bitmap is at least one bit per page.
        let ret = unsafe { ioctl_with_ref(self, KVM_GET_DIRTY_LOG(), &dirtylog) };
        if ret == 0 {
            Ok(bitmap)
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Registers an event that will, when signaled, trigger the `gsi` IRQ.
    ///
    /// # Arguments
    ///
    /// * `evt` - Event to be signaled.
    /// * `gsi` - IRQ to be triggered.
    ///
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64"
    ))]
    pub fn register_irqfd(&self, evt: &EventFd, gsi: u32) -> Result<()> {
        let irqfd = kvm_irqfd {
            fd: evt.as_raw_fd() as u32,
            gsi,
            ..Default::default()
        };
        // Safe because we know that our file is a VM fd, we know the kernel will only read the
        // correct amount of memory from our pointer, and we verify the return result.
        let ret = unsafe { ioctl_with_ref(self, KVM_IRQFD(), &irqfd) };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Deregisters the event `evt` that would trigger the `gsi` IRQ.
    ///
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64"
    ))]
    pub fn deregister_irqfd(&self, evt: &EventFd, gsi: u32) -> Result<()> {
        let irqfd = kvm_irqfd {
            fd: evt.as_raw_fd() as u32,
            gsi,
            flags: KVM_IRQFD_FLAG_DEASSIGN,
            ..Default::default()
        };
        // Safe because we know that our file is a VM fd, we know the kernel will only read the
        // correct amount of memory from our pointer, and we verify the return result.
        let ret = unsafe { ioctl_with_ref(self, KVM_IRQFD(), &irqfd) };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Constructs a new kvm VCPU fd.
    ///
    /// # Arguments
    ///
    /// * `id` - The CPU number between [0, max vcpus).
    ///
    /// # Errors
    /// Returns an error when the VM fd is invalid or the VCPU memory cannot be mapped correctly.
    ///
    pub fn create_vcpu(&self, id: u8) -> Result<VcpuFd> {
        // Safe because we know that vm is a VM fd and we verify the return result.
        #[allow(clippy::cast_lossless)]
        let vcpu_fd = unsafe { ioctl_with_val(&self.vm, KVM_CREATE_VCPU(), id as c_ulong) };
        if vcpu_fd < 0 {
            return Err(io::Error::last_os_error());
        }

        // Wrap the vcpu now in case the following ? returns early. This is safe because we verified
        // the value of the fd and we own the fd.
        let vcpu = unsafe { File::from_raw_fd(vcpu_fd) };

        let kvm_run_ptr = KvmRunWrapper::from_fd(&vcpu, self.run_size)?;

        Ok(VcpuFd { vcpu, kvm_run_ptr })
    }

    /// X86 specific call to retrieve the state of a kernel interrupt controller.
    ///
    /// See the documentation for `KVM_GET_IRQCHIP`.
    ///
    /// # Arguments
    ///
    /// * `irqchip` - `kvm_irqchip` to be read.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn get_irqchip(&self, irqchip: &mut kvm_irqchip) -> Result<()> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_irqchip struct.
            ioctl_with_mut_ref(self, KVM_GET_IRQCHIP(), irqchip)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call to set the state of a kernel interrupt controller.
    ///
    /// See the documentation for `KVM_SET_IRQCHIP`.
    ///
    /// # Arguments
    ///
    /// * `irqchip` - `kvm_irqchip` to be written.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn set_irqchip(&self, irqchip: &kvm_irqchip) -> Result<()> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_irqchip struct.
            ioctl_with_ref(self, KVM_SET_IRQCHIP(), irqchip)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call to retrieve the current timestamp of kvmclock.
    ///
    /// See the documentation for `KVM_GET_CLOCK`.
    ///
    /// # Arguments
    ///
    /// * `clock` - `kvm_clock_data` to be read.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn get_clock(&self, clock: &mut kvm_clock_data) -> Result<()> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_clock_data struct.
            ioctl_with_mut_ref(self, KVM_GET_CLOCK(), clock)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call to set the current timestamp of kvmclock.
    ///
    /// See the documentation for `KVM_SET_CLOCK`.
    ///
    /// # Arguments
    ///
    /// * `clock` - `kvm_clock_data` to be written.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn set_clock(&self, clock: &kvm_clock_data) -> Result<()> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_clock_data struct.
            ioctl_with_ref(self, KVM_SET_CLOCK(), clock)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call to retrieve the state of the in-kernel PIT model.
    ///
    /// See the documentation for `KVM_GET_PIT2`.
    ///
    /// # Arguments
    ///
    /// * `pitstate` - `kvm_pit_state2` to be read.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn get_pit2(&self, pitstate: &mut kvm_pit_state2) -> Result<()> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_pit_state2 struct.
            ioctl_with_mut_ref(self, KVM_GET_PIT2(), pitstate)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call to set the state of the in-kernel PIT model.
    ///
    /// See the documentation for `KVM_SET_PIT2`.
    ///
    /// # Arguments
    ///
    /// * `pitstate` - `kvm_pit_state2` to be written.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn set_pit2(&self, pitstate: &kvm_pit_state2) -> Result<()> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_pit_state2 struct.
            ioctl_with_ref(self, KVM_SET_PIT2(), pitstate)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Creates an emulated device in the kernel.
    ///
    /// See the documentation for `KVM_CREATE_DEVICE`.
    ///
    /// # Arguments
    ///
    /// * `device`: device configuration. For details check the `kvm_create_device` structure in the
    ///                [KVM API doc](https://www.kernel.org/doc/Documentation/virtual/kvm/api.txt).
    ///
    /// # Example
    ///
    /// ```rust
    /// # extern crate kvm;
    /// # extern crate kvm_bindings;
    /// # use kvm::{Kvm, VmFd, VcpuFd};
    /// use kvm_bindings::{
    ///     kvm_device_type_KVM_DEV_TYPE_VFIO,
    ///     kvm_device_type_KVM_DEV_TYPE_ARM_VGIC_V3,
    ///     KVM_CREATE_DEVICE_TEST,
    /// };
    /// let kvm = Kvm::new().unwrap();
    /// let vm = kvm.create_vm().unwrap();
    ///
    /// // Creating a device with the KVM_CREATE_DEVICE_TEST flag to check
    /// // whether the device type is supported. This will not create the device.
    /// // To create the device the flag needs to be removed.
    /// let mut device = kvm_bindings::kvm_create_device {
    ///     #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    ///     type_: kvm_device_type_KVM_DEV_TYPE_VFIO,
    ///     #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    ///     type_: kvm_device_type_KVM_DEV_TYPE_ARM_VGIC_V3,
    ///     fd: 0,
    ///     flags: KVM_CREATE_DEVICE_TEST,
    /// };
    /// let device_fd = vm
    ///     .create_device(&mut device).unwrap();
    /// ```
    ///
    pub fn create_device(&self, device: &mut kvm_create_device) -> Result<DeviceFd> {
        let ret = unsafe { ioctl_with_ref(self, KVM_CREATE_DEVICE(), device) };
        if ret == 0 {
            Ok(new_device(unsafe { File::from_raw_fd(device.fd as i32) }))
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Returns the preferred CPU target type which can be emulated by KVM on underlying host.
    ///
    /// The preferred CPU target is returned in the `kvi` parameter.
    /// See documentation for `KVM_ARM_PREFERRED_TARGET`.
    ///
    /// # Arguments
    /// * `kvi` - CPU target configuration (out). For details check the `kvm_vcpu_init`
    ///           structure in the
    ///           [KVM API doc](https://www.kernel.org/doc/Documentation/virtual/kvm/api.txt).
    ///
    /// # Example
    ///
    /// ```rust
    /// # extern crate kvm;
    /// # extern crate kvm_bindings;
    /// # use kvm::{Kvm, VmFd, VcpuFd};
    /// use kvm_bindings::kvm_vcpu_init;
    /// let kvm = Kvm::new().unwrap();
    /// let vm = kvm.create_vm().unwrap();
    /// let mut kvi = kvm_vcpu_init::default();
    /// vm.get_preferred_target(&mut kvi).unwrap();
    /// ```
    ///
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    pub fn get_preferred_target(&self, kvi: &mut kvm_vcpu_init) -> Result<()> {
        // The ioctl is safe because we allocated the struct and we know the
        // kernel will write exactly the size of the struct.
        let ret = unsafe { ioctl_with_mut_ref(self, KVM_ARM_PREFERRED_TARGET(), kvi) };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
}

impl AsRawFd for VmFd {
    fn as_raw_fd(&self) -> RawFd {
        self.vm.as_raw_fd()
    }
}

/// Reasons for vcpu exits. The exit reasons are mapped to the `KVM_EXIT_*` defines
/// from `include/uapi/linux/kvm.h`.
#[derive(Debug)]
pub enum VcpuExit<'a> {
    /// An out port instruction was run on the given port with the given data.
    IoOut(u16 /* port */, &'a [u8] /* data */),
    /// An in port instruction was run on the given port.
    ///
    /// The given slice should be filled in before `Vcpu::run` is called again.
    IoIn(u16 /* port */, &'a mut [u8] /* data */),
    /// A read instruction was run against the given MMIO address.
    ///
    /// The given slice should be filled in before `Vcpu::run` is called again.
    MmioRead(u64 /* address */, &'a mut [u8]),
    /// A write instruction was run against the given MMIO address with the given data.
    MmioWrite(u64 /* address */, &'a [u8]),
    /// Corresponds to KVM_EXIT_UNKNOWN.
    Unknown,
    /// Corresponds to KVM_EXIT_EXCEPTION.
    Exception,
    /// Corresponds to KVM_EXIT_HYPERCALL.
    Hypercall,
    /// Corresponds to KVM_EXIT_DEBUG.
    Debug,
    /// Corresponds to KVM_EXIT_HLT.
    Hlt,
    /// Corresponds to KVM_EXIT_IRQ_WINDOW_OPEN.
    IrqWindowOpen,
    /// Corresponds to KVM_EXIT_SHUTDOWN.
    Shutdown,
    /// Corresponds to KVM_EXIT_FAIL_ENTRY.
    FailEntry,
    /// Corresponds to KVM_EXIT_INTR.
    Intr,
    /// Corresponds to KVM_EXIT_SET_TPR.
    SetTpr,
    /// Corresponds to KVM_EXIT_TPR_ACCESS.
    TprAccess,
    /// Corresponds to KVM_EXIT_S390_SIEIC.
    S390Sieic,
    /// Corresponds to KVM_EXIT_S390_RESET.
    S390Reset,
    /// Corresponds to KVM_EXIT_DCR.
    Dcr,
    /// Corresponds to KVM_EXIT_NMI.
    Nmi,
    /// Corresponds to KVM_EXIT_INTERNAL_ERROR.
    InternalError,
    /// Corresponds to KVM_EXIT_OSI.
    Osi,
    /// Corresponds to KVM_EXIT_PAPR_HCALL.
    PaprHcall,
    /// Corresponds to KVM_EXIT_S390_UCONTROL.
    S390Ucontrol,
    /// Corresponds to KVM_EXIT_WATCHDOG.
    Watchdog,
    /// Corresponds to KVM_EXIT_S390_TSCH.
    S390Tsch,
    /// Corresponds to KVM_EXIT_EPR.
    Epr,
    /// Corresponds to KVM_EXIT_SYSTEM_EVENT.
    SystemEvent,
    /// Corresponds to KVM_EXIT_S390_STSI.
    S390Stsi,
    /// Corresponds to KVM_EXIT_IOAPIC_EOI.
    IoapicEoi,
    /// Corresponds to KVM_EXIT_HYPERV.
    Hyperv,
}

/// A wrapper around creating and using a kvm related VCPU fd
pub struct VcpuFd {
    vcpu: File,
    kvm_run_ptr: KvmRunWrapper,
}

impl VcpuFd {
    /// Gets the VCPU registers using the `KVM_GET_REGS` ioctl.
    ///
    #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
    pub fn get_regs(&self) -> Result<kvm_regs> {
        // Safe because we know that our file is a VCPU fd, we know the kernel will only read the
        // correct amount of memory from our pointer, and we verify the return result.
        let mut regs = unsafe { std::mem::zeroed() };
        let ret = unsafe { ioctl_with_mut_ref(self, KVM_GET_REGS(), &mut regs) };
        if ret == 0 {
            Ok(regs)
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Sets the VCPU registers using `KVM_SET_REGS` ioctl.
    ///
    /// # Arguments
    ///
    /// * `regs` - Registers being set.
    ///
    #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
    pub fn set_regs(&self, regs: &kvm_regs) -> Result<()> {
        // Safe because we know that our file is a VCPU fd, we know the kernel will only read the
        // correct amount of memory from our pointer, and we verify the return result.
        let ret = unsafe { ioctl_with_ref(self, KVM_SET_REGS(), regs) };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Gets the VCPU special registers using `KVM_GET_SREGS` ioctl.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn get_sregs(&self) -> Result<kvm_sregs> {
        // Safe because we know that our file is a VCPU fd, we know the kernel will only write the
        // correct amount of memory to our pointer, and we verify the return result.
        let mut regs = kvm_sregs::default();

        let ret = unsafe { ioctl_with_mut_ref(self, KVM_GET_SREGS(), &mut regs) };
        if ret == 0 {
            Ok(regs)
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Sets the VCPU special registers using `KVM_SET_SREGS` ioctl.
    ///
    /// # Arguments
    ///
    /// * `sregs` - Special registers to be set.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn set_sregs(&self, sregs: &kvm_sregs) -> Result<()> {
        // Safe because we know that our file is a VCPU fd, we know the kernel will only read the
        // correct amount of memory from our pointer, and we verify the return result.
        let ret = unsafe { ioctl_with_ref(self, KVM_SET_SREGS(), sregs) };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call that gets the FPU-related structure.
    ///
    /// See the documentation for `KVM_GET_FPU`.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn get_fpu(&self) -> Result<kvm_fpu> {
        let mut fpu = kvm_fpu::default();

        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_fpu struct.
            ioctl_with_mut_ref(self, KVM_GET_FPU(), &mut fpu)
        };
        if ret == 0 {
            Ok(fpu)
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call to setup the FPU.
    ///
    /// See the documentation for `KVM_SET_FPU`.
    ///
    /// # Arguments
    ///
    /// * `fpu` - FPU configurations struct.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn set_fpu(&self, fpu: &kvm_fpu) -> Result<()> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_fpu struct.
            ioctl_with_ref(self, KVM_SET_FPU(), fpu)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call to retrieve the CPUID registers.
    ///
    /// It requires knowledge of how many `kvm_cpuid_entry2` entries there are to get.
    /// See the documentation for `KVM_GET_CPUID2`.
    ///
    /// # Arguments
    ///
    /// * `num_entries` - Number of CPUID entries to be read.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn get_cpuid2(&self, num_entries: usize) -> Result<CpuId> {
        let mut cpuid = CpuId::new(num_entries);
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_cpuid2 struct.
            ioctl_with_mut_ptr(self, KVM_GET_CPUID2(), cpuid.as_mut_ptr())
        };
        if ret == 0 {
            Ok(cpuid)
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call to setup the CPUID registers.
    ///
    /// See the documentation for `KVM_SET_CPUID2`.
    ///
    /// # Arguments
    ///
    /// * `cpuid` - CPUID registers to be written.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn set_cpuid2(&self, cpuid: &CpuId) -> Result<()> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_cpuid2 struct.
            ioctl_with_ptr(self, KVM_SET_CPUID2(), cpuid.as_ptr())
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call to get the state of the LAPIC (Local Advanced Programmable Interrupt
    /// Controller).
    ///
    /// See the documentation for `KVM_GET_LAPIC`.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn get_lapic(&self) -> Result<kvm_lapic_state> {
        let mut klapic = kvm_lapic_state::default();

        let ret = unsafe {
            // The ioctl is unsafe unless you trust the kernel not to write past the end of the
            // local_apic struct.
            ioctl_with_mut_ref(self, KVM_GET_LAPIC(), &mut klapic)
        };
        if ret == 0 {
            Ok(klapic)
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call to set the state of the LAPIC (Local Advanced Programmable Interrupt
    /// Controller).
    ///
    /// See the documentation for `KVM_SET_LAPIC`.
    ///
    /// # Arguments
    ///
    /// * `klapic` - LAPIC state registers.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn set_lapic(&self, klapic: &kvm_lapic_state) -> Result<()> {
        let ret = unsafe {
            // The ioctl is safe because the kernel will only read from the klapic struct.
            ioctl_with_ref(self, KVM_SET_LAPIC(), klapic)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call to read model-specific registers for this VCPU.
    ///
    /// It emulates `KVM_GET_MSRS` ioctl's behavior by returning the number of MSRs
    /// successfully read upon success or the last error number in case of failure.
    ///
    /// # Arguments
    ///
    /// * `msrs` - MSRs to be read.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn get_msrs(&self, msrs: &mut KvmMsrs) -> Result<usize> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_msrs struct.
            ioctl_with_mut_ptr(self, KVM_GET_MSRS(), msrs.as_mut_ptr())
        };
        // KVM_GET_MSRS returns the number of msr entries read.
        if ret >= 0 {
            Ok(ret as usize)
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call to setup the MSRS.
    ///
    /// See the documentation for `KVM_SET_MSRS`.
    ///
    /// # Arguments
    ///
    /// * `msrs` - MSRs to be written.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn set_msrs(&self, msrs: &KvmMsrs) -> Result<usize> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_msrs struct.
            ioctl_with_ptr(self, KVM_SET_MSRS(), msrs.as_ptr())
        };
        // KVM_SET_MSRS actually returns the number of msr entries written.
        if ret >= 0 {
            Ok(ret as usize)
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Returns the vcpu's current "multiprocessing state".
    ///
    /// See the documentation for `KVM_GET_MP_STATE`.
    ///
    /// # Arguments
    ///
    /// * `kvm_mp_state` - multiprocessing state to be read.
    ///
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "s390"
    ))]
    pub fn get_mp_state(&self, mp_state: &mut kvm_mp_state) -> Result<()> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_mp_state struct.
            ioctl_with_mut_ref(self, KVM_GET_MP_STATE(), mp_state)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Sets the vcpu's current "multiprocessing state".
    ///
    /// See the documentation for `KVM_SET_MP_STATE`.
    ///
    /// # Arguments
    ///
    /// * `kvm_mp_state` - multiprocessing state to be written.
    ///
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "s390"
    ))]
    pub fn set_mp_state(&self, mp_state: kvm_mp_state) -> Result<()> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_mp_state struct.
            ioctl_with_ref(self, KVM_SET_MP_STATE(), &mp_state)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call that returns the vcpu's current "xsave struct".
    ///
    /// See the documentation for `KVM_GET_XSAVE`.
    ///
    /// # Arguments
    ///
    /// * `kvm_xsave` - xsave struct to be read.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn get_xsave(&self, xsave: &mut kvm_xsave) -> Result<()> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_xsave struct.
            ioctl_with_mut_ref(self, KVM_GET_XSAVE(), xsave)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call that sets the vcpu's current "xsave struct".
    ///
    /// See the documentation for `KVM_SET_XSAVE`.
    ///
    /// # Arguments
    ///
    /// * `kvm_xsave` - xsave struct to be written.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn set_xsave(&self, xsave: &kvm_xsave) -> Result<()> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_xsave struct.
            ioctl_with_ref(self, KVM_SET_XSAVE(), xsave)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call that returns the vcpu's current "xcrs".
    ///
    /// See the documentation for `KVM_GET_XCRS`.
    ///
    /// # Arguments
    ///
    /// * `kvm_xcrs` - xcrs to be read.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn get_xcrs(&self, xcrs: &mut kvm_xcrs) -> Result<()> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_xcrs struct.
            ioctl_with_mut_ref(self, KVM_GET_XCRS(), xcrs)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call that sets the vcpu's current "xcrs".
    ///
    /// See the documentation for `KVM_SET_XCRS`.
    ///
    /// # Arguments
    ///
    /// * `kvm_xcrs` - xcrs to be written.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn set_xcrs(&self, xcrs: &kvm_xcrs) -> Result<()> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_xcrs struct.
            ioctl_with_ref(self, KVM_SET_XCRS(), xcrs)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call that returns the vcpu's current "debug registers".
    ///
    /// See the documentation for `KVM_GET_DEBUGREGS`.
    ///
    /// # Arguments
    ///
    /// * `kvm_debugregs` - debug registers to be read.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn get_debug_regs(&self, debug_regs: &mut kvm_debugregs) -> Result<()> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_debugregs struct.
            ioctl_with_mut_ref(self, KVM_GET_DEBUGREGS(), debug_regs)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// X86 specific call that sets the vcpu's current "debug registers".
    ///
    /// See the documentation for `KVM_SET_DEBUGREGS`.
    ///
    /// # Arguments
    ///
    /// * `kvm_debugregs` - debug registers to be written.
    ///
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn set_debug_regs(&self, debug_regs: &kvm_debugregs) -> Result<()> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_debugregs struct.
            ioctl_with_ref(self, KVM_SET_DEBUGREGS(), debug_regs)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Returns currently pending exceptions, interrupts, and NMIs as well as related
    /// states of the vcpu.
    ///
    /// See the documentation for `KVM_GET_VCPU_EVENTS`.
    ///
    /// # Arguments
    ///
    /// * `kvm_vcpu_events` - vcpu events to be read.
    ///
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64"
    ))]
    pub fn get_vcpu_events(&self, vcpu_events: &mut kvm_vcpu_events) -> Result<()> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_vcpu_events struct.
            ioctl_with_mut_ref(self, KVM_GET_VCPU_EVENTS(), vcpu_events)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Sets pending exceptions, interrupts, and NMIs as well as related states of the vcpu.
    ///
    /// See the documentation for `KVM_SET_VCPU_EVENTS`.
    ///
    /// # Arguments
    ///
    /// * `kvm_vcpu_events` - vcpu events to be written.
    ///
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64"
    ))]
    pub fn set_vcpu_events(&self, vcpu_events: &kvm_vcpu_events) -> Result<()> {
        let ret = unsafe {
            // Here we trust the kernel not to read past the end of the kvm_vcpu_events struct.
            ioctl_with_ref(self, KVM_SET_VCPU_EVENTS(), vcpu_events)
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Sets the type of CPU to be exposed to the guest and optional features.
    ///
    /// This initializes an ARM vCPU to the specified type with the specified features
    /// and resets the values of all of its registers to defaults. See the documentation for
    /// `KVM_ARM_VCPU_INIT`.
    ///
    /// # Arguments
    ///
    /// * `kvi` - information about preferred CPU target type and recommended features for it.
    ///           For details check the `kvm_vcpu_init` structure in the
    ///           [KVM API doc](https://www.kernel.org/doc/Documentation/virtual/kvm/api.txt).
    ///
    /// # Example
    /// ```rust
    /// # extern crate kvm;
    /// # extern crate kvm_bindings;
    /// # use kvm::{Kvm, VmFd, VcpuFd};
    /// use kvm_bindings::kvm_vcpu_init;
    /// let kvm = Kvm::new().unwrap();
    /// let vm = kvm.create_vm().unwrap();
    /// let vcpu = vm.create_vcpu(0).unwrap();
    ///
    /// let mut kvi = kvm_vcpu_init::default();
    /// vm.get_preferred_target(&mut kvi).unwrap();
    /// vcpu.vcpu_init(&kvi).unwrap();
    /// ```
    ///
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    pub fn vcpu_init(&self, kvi: &kvm_vcpu_init) -> Result<()> {
        // This is safe because we allocated the struct and we know the kernel will read
        // exactly the size of the struct.
        let ret = unsafe { ioctl_with_ref(self, KVM_ARM_VCPU_INIT(), kvi) };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Sets the value of one register for this vCPU.
    ///
    /// The id of the register is encoded as specified in the kernel documentation
    /// for `KVM_SET_ONE_REG`.
    ///
    /// # Arguments
    ///
    /// * `reg_id` - ID of the register for which we are setting the value.
    /// * `data` - value for the specified register.
    ///
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    pub fn set_one_reg(&self, reg_id: u64, data: u64) -> Result<()> {
        let data_ref = &data as *const u64;
        let onereg = kvm_one_reg {
            id: reg_id,
            addr: data_ref as u64,
        };
        // This is safe because we allocated the struct and we know the kernel will read
        // exactly the size of the struct.
        let ret = unsafe { ioctl_with_ref(self, KVM_SET_ONE_REG(), &onereg) };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Sets the `immediate_exit` flag on the `kvm_run` struct associated with this object to `val`.
    pub fn set_kvm_immediate_exit(&self, val: u8) {
        let kvm_run = self.kvm_run_ptr.as_mut_ref();
        kvm_run.immediate_exit = val;
    }

    /// Triggers the running of the current virtual CPU returning an exit reason.
    ///
    pub fn run(&self) -> Result<VcpuExit> {
        // Safe because we know that our file is a VCPU fd and we verify the return result.
        let ret = unsafe { ioctl(self, KVM_RUN()) };
        if ret == 0 {
            let run = self.kvm_run_ptr.as_mut_ref();
            match run.exit_reason {
                // make sure you treat all possible exit reasons from include/uapi/linux/kvm.h corresponding
                // when upgrading to a different kernel version
                KVM_EXIT_UNKNOWN => Ok(VcpuExit::Unknown),
                KVM_EXIT_EXCEPTION => Ok(VcpuExit::Exception),
                KVM_EXIT_IO => {
                    let run_start = run as *mut kvm_run as *mut u8;
                    // Safe because the exit_reason (which comes from the kernel) told us which
                    // union field to use.
                    let io = unsafe { run.__bindgen_anon_1.io };
                    let port = io.port;
                    let data_size = io.count as usize * io.size as usize;
                    // The data_offset is defined by the kernel to be some number of bytes into the
                    // kvm_run stucture, which we have fully mmap'd.
                    let data_ptr = unsafe { run_start.offset(io.data_offset as isize) };
                    // The slice's lifetime is limited to the lifetime of this Vcpu, which is equal
                    // to the mmap of the kvm_run struct that this is slicing from
                    let data_slice = unsafe {
                        std::slice::from_raw_parts_mut::<u8>(data_ptr as *mut u8, data_size)
                    };
                    match u32::from(io.direction) {
                        KVM_EXIT_IO_IN => Ok(VcpuExit::IoIn(port, data_slice)),
                        KVM_EXIT_IO_OUT => Ok(VcpuExit::IoOut(port, data_slice)),
                        _ => Err(io::Error::from_raw_os_error(EINVAL)),
                    }
                }
                KVM_EXIT_HYPERCALL => Ok(VcpuExit::Hypercall),
                KVM_EXIT_DEBUG => Ok(VcpuExit::Debug),
                KVM_EXIT_HLT => Ok(VcpuExit::Hlt),
                KVM_EXIT_MMIO => {
                    // Safe because the exit_reason (which comes from the kernel) told us which
                    // union field to use.
                    let mmio = unsafe { &mut run.__bindgen_anon_1.mmio };
                    let addr = mmio.phys_addr;
                    let len = mmio.len as usize;
                    let data_slice = &mut mmio.data[..len];
                    if mmio.is_write != 0 {
                        Ok(VcpuExit::MmioWrite(addr, data_slice))
                    } else {
                        Ok(VcpuExit::MmioRead(addr, data_slice))
                    }
                }
                KVM_EXIT_IRQ_WINDOW_OPEN => Ok(VcpuExit::IrqWindowOpen),
                KVM_EXIT_SHUTDOWN => Ok(VcpuExit::Shutdown),
                KVM_EXIT_FAIL_ENTRY => Ok(VcpuExit::FailEntry),
                KVM_EXIT_INTR => Ok(VcpuExit::Intr),
                KVM_EXIT_SET_TPR => Ok(VcpuExit::SetTpr),
                KVM_EXIT_TPR_ACCESS => Ok(VcpuExit::TprAccess),
                KVM_EXIT_S390_SIEIC => Ok(VcpuExit::S390Sieic),
                KVM_EXIT_S390_RESET => Ok(VcpuExit::S390Reset),
                KVM_EXIT_DCR => Ok(VcpuExit::Dcr),
                KVM_EXIT_NMI => Ok(VcpuExit::Nmi),
                KVM_EXIT_INTERNAL_ERROR => Ok(VcpuExit::InternalError),
                KVM_EXIT_OSI => Ok(VcpuExit::Osi),
                KVM_EXIT_PAPR_HCALL => Ok(VcpuExit::PaprHcall),
                KVM_EXIT_S390_UCONTROL => Ok(VcpuExit::S390Ucontrol),
                KVM_EXIT_WATCHDOG => Ok(VcpuExit::Watchdog),
                KVM_EXIT_S390_TSCH => Ok(VcpuExit::S390Tsch),
                KVM_EXIT_EPR => Ok(VcpuExit::Epr),
                KVM_EXIT_SYSTEM_EVENT => Ok(VcpuExit::SystemEvent),
                KVM_EXIT_S390_STSI => Ok(VcpuExit::S390Stsi),
                KVM_EXIT_IOAPIC_EOI => Ok(VcpuExit::IoapicEoi),
                KVM_EXIT_HYPERV => Ok(VcpuExit::Hyperv),
                r => panic!("unknown kvm exit reason: {}", r),
            }
        } else {
            Err(io::Error::last_os_error())
        }
    }
}

impl AsRawFd for VcpuFd {
    fn as_raw_fd(&self) -> RawFd {
        self.vcpu.as_raw_fd()
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl KvmArray for kvm_cpuid2 {
    type Entry = kvm_cpuid_entry2;

    fn len(&self) -> usize {
        self.nent as usize
    }

    fn set_len(&mut self, len: usize) {
        self.nent = len as u32;
    }

    fn max_len() -> usize {
        MAX_KVM_CPUID_ENTRIES
    }

    fn entries(&self) -> &__IncompleteArrayField<kvm_cpuid_entry2> {
        &self.entries
    }

    fn entries_mut(&mut self) -> &mut __IncompleteArrayField<kvm_cpuid_entry2> {
        &mut self.entries
    }
}

/// Wrapper for `kvm_cpuid2`.
///
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub type CpuId = KvmVec<kvm_cpuid2>;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl KvmArray for kvm_msr_list {
    type Entry = u32;

    fn len(&self) -> usize {
        self.nmsrs as usize
    }

    fn set_len(&mut self, len: usize) {
        self.nmsrs = len as u32;
    }

    fn max_len() -> usize {
        MAX_KVM_MSR_ENTRIES as usize
    }

    fn entries(&self) -> &__IncompleteArrayField<u32> {
        &self.indices
    }

    fn entries_mut(&mut self) -> &mut __IncompleteArrayField<u32> {
        &mut self.indices
    }
}

/// Wrapper for `kvm_msr_list`.
///
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub type MsrList = KvmVec<kvm_msr_list>;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl KvmArray for kvm_msrs {
    /// The entries in the zero-sized array are of type `kvm_msr_entry`.
    type Entry = kvm_msr_entry;

    fn len(&self) -> usize {
        self.nmsrs as usize
    }

    fn set_len(&mut self, len: usize) {
        self.nmsrs = len as u32;
    }

    fn max_len() -> usize {
        MAX_KVM_MSR_ENTRIES as usize
    }

    fn entries(&self) -> &__IncompleteArrayField<Self::Entry> {
        &self.entries
    }

    fn entries_mut(&mut self) -> &mut __IncompleteArrayField<Self::Entry> {
        &mut self.entries
    }
}

/// Wrapper for `kvm_msrs` which has a zero length array at the end.
///
/// Hides the zero-sized array behind a bounds check.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub type KvmMsrs = KvmVec<kvm_msrs>;

#[cfg(test)]
mod tests {
    extern crate byteorder;

    use super::*;

    use kvm_cpuid_entry2;
    use CpuId;

    // as per https://github.com/torvalds/linux/blob/master/arch/x86/include/asm/fpu/internal.h
    pub const KVM_FPU_CWD: usize = 0x37f;
    pub const KVM_FPU_MXCSR: usize = 0x1f80;

    impl VmFd {
        fn get_run_size(&self) -> usize {
            self.run_size
        }
    }

    // Helper function for mmap an anonymous memory of `size`.
    // Panics if the mmap fails.
    fn mmap_anonymous(size: usize) -> *mut u8 {
        let addr = unsafe {
            libc::mmap(
                null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_SHARED | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            panic!("mmap failed.");
        }

        addr as *mut u8
    }

    impl KvmRunWrapper {
        fn new(size: usize) -> Self {
            KvmRunWrapper {
                kvm_run_ptr: mmap_anonymous(size),
                size,
            }
        }
    }

    // kvm system related function tests
    #[test]
    fn new() {
        Kvm::new().unwrap();
    }

    #[test]
    fn check_extension() {
        let kvm = Kvm::new().unwrap();
        assert!(kvm.check_extension(Cap::UserMemory));
        // I assume nobody is testing this on s390
        assert!(!kvm.check_extension(Cap::S390UserSigp));
    }

    #[test]
    fn vcpu_mmap_size() {
        let kvm = Kvm::new().unwrap();
        let mmap_size = kvm.get_vcpu_mmap_size().unwrap();
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        assert!(mmap_size >= page_size);
        assert_eq!(mmap_size % page_size, 0);
    }

    #[test]
    fn get_nr_vcpus() {
        let kvm = Kvm::new().unwrap();
        let nr_vcpus = kvm.get_nr_vcpus();
        assert!(nr_vcpus >= 4);
    }

    #[test]
    fn get_max_memslots() {
        let kvm = Kvm::new().unwrap();
        let max_mem_slots = kvm.get_nr_memslots();
        assert!(max_mem_slots >= 32);
    }

    #[test]
    fn test_faulty_memory_region_slot() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let max_mem_slots = kvm.get_nr_memslots();

        let mem_size = 1 << 20;
        let userspace_addr = mmap_anonymous(mem_size) as u64;
        let region_addr: usize = 0x0;

        // KVM is checking that the memory region slot is less than KVM_CAP_NR_MEMSLOTS.
        // Valid slots are in the interval [0, KVM_CAP_NR_MEMSLOTS).
        let mem_region = kvm_userspace_memory_region {
            slot: max_mem_slots as u32,
            guest_phys_addr: region_addr as u64,
            memory_size: mem_size as u64,
            userspace_addr: userspace_addr as u64,
            flags: 0,
        };

        // KVM returns -EINVAL (22) when the slot > KVM_CAP_NR_MEMSLOTS.
        assert_eq!(
            vm.set_user_memory_region(mem_region)
                .unwrap_err()
                .raw_os_error()
                .unwrap(),
            22
        );
    }

    #[test]
    fn set_kvm_immediate_exit() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let vcpu = vm.create_vcpu(0).unwrap();
        assert_eq!(vcpu.kvm_run_ptr.as_mut_ref().immediate_exit, 0);
        vcpu.set_kvm_immediate_exit(1);
        assert_eq!(vcpu.kvm_run_ptr.as_mut_ref().immediate_exit, 1);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn cpuid_test() {
        let kvm = Kvm::new().unwrap();
        if kvm.check_extension(Cap::ExtCpuid) {
            let vm = kvm.create_vm().unwrap();
            let mut cpuid = kvm.get_supported_cpuid(MAX_KVM_CPUID_ENTRIES).unwrap();
            let ncpuids = cpuid.as_mut_entries_slice().len();
            assert!(ncpuids <= MAX_KVM_CPUID_ENTRIES);
            let nr_vcpus = kvm.get_nr_vcpus();
            for cpu_idx in 0..nr_vcpus {
                let vcpu = vm.create_vcpu(cpu_idx as u8).unwrap();
                vcpu.set_cpuid2(&cpuid).unwrap();
                let mut retrieved_cpuid = vcpu.get_cpuid2(ncpuids).unwrap();
                // Only check the first few leafs as some (e.g. 13) are reserved.
                assert_eq!(
                    cpuid.as_entries_slice()[..3],
                    retrieved_cpuid.as_entries_slice()[..3]
                );
            }
        }
    }

    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "s390"
    ))]
    #[test]
    fn mpstate_test() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let vcpu = vm.create_vcpu(0).unwrap();
        let mut mp_state = kvm_mp_state::default();
        vcpu.get_mp_state(&mut mp_state).unwrap();
        vcpu.set_mp_state(mp_state).unwrap();
        let mut other_mp_state = kvm_mp_state::default();
        vcpu.get_mp_state(&mut other_mp_state).unwrap();
        assert_eq!(mp_state, other_mp_state);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn xsave_test() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let vcpu = vm.create_vcpu(0).unwrap();
        let mut xsave = kvm_xsave::default();
        vcpu.get_xsave(&mut xsave).unwrap();
        vcpu.set_xsave(&xsave).unwrap();
        let mut other_xsave = kvm_xsave::default();
        vcpu.get_xsave(&mut other_xsave).unwrap();
        assert_eq!(&xsave.region[..], &other_xsave.region[..]);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn xcrs_test() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let vcpu = vm.create_vcpu(0).unwrap();
        let mut xcrs = kvm_xcrs::default();
        vcpu.get_xcrs(&mut xcrs).unwrap();
        vcpu.set_xcrs(&xcrs).unwrap();
        let mut other_xcrs = kvm_xcrs::default();
        vcpu.get_xcrs(&mut other_xcrs).unwrap();
        assert_eq!(xcrs, other_xcrs);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn debugregs_test() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let vcpu = vm.create_vcpu(0).unwrap();
        let mut debugregs = kvm_debugregs::default();
        vcpu.get_debug_regs(&mut debugregs).unwrap();
        vcpu.set_debug_regs(&debugregs).unwrap();
        let mut other_debugregs = kvm_debugregs::default();
        vcpu.get_debug_regs(&mut other_debugregs).unwrap();
        assert_eq!(debugregs, other_debugregs);
    }

    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64"
    ))]
    #[test]
    fn vcpu_events_test() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let vcpu = vm.create_vcpu(0).unwrap();
        let mut vcpu_events = kvm_vcpu_events::default();
        vcpu.get_vcpu_events(&mut vcpu_events).unwrap();
        vcpu.set_vcpu_events(&vcpu_events).unwrap();
        let mut other_vcpu_events = kvm_vcpu_events::default();
        vcpu.get_vcpu_events(&mut other_vcpu_events).unwrap();
        assert_eq!(vcpu_events, other_vcpu_events);
    }

    // kvm vm related function tests
    #[test]
    fn create_vm() {
        let kvm = Kvm::new().unwrap();
        kvm.create_vm().unwrap();
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn msr_index_list() {
        let kvm = Kvm::new().unwrap();
        if kvm.check_extension(Cap::MsrFeatures) {
            let msr_list = kvm.get_msr_index_list().unwrap();
            assert!(msr_list.as_original_struct().len() <= MAX_KVM_MSR_ENTRIES);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn msr_feature_index_list() {
        let kvm = Kvm::new().unwrap();
        if kvm.check_extension(Cap::MsrFeatures) {
            let msr_list = kvm.get_msr_feature_index_list().unwrap();
            assert!(msr_list.as_original_struct().len() <= MAX_KVM_MSR_ENTRIES);
        }
    }

    #[test]
    fn get_vm_run_size() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        assert_eq!(kvm.get_vcpu_mmap_size().unwrap(), vm.get_run_size());
    }

    #[test]
    fn set_invalid_memory_test() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let invalid_mem_region = kvm_userspace_memory_region {
            slot: 0,
            guest_phys_addr: 0,
            memory_size: 0,
            userspace_addr: 0,
            flags: 0,
        };
        assert!(vm.set_user_memory_region(invalid_mem_region).is_err());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn set_tss_address() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        assert!(vm.set_tss_address(0xfffb_d000).is_ok());
    }

    #[test]
    fn irq_chip() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        assert!(vm.create_irq_chip().is_ok());

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let mut irqchip = kvm_irqchip::default();
            irqchip.chip_id = KVM_IRQCHIP_PIC_MASTER;
            vm.get_irqchip(&mut irqchip).unwrap();
            vm.set_irqchip(&irqchip).unwrap();
            let mut other_irqchip = kvm_irqchip::default();
            other_irqchip.chip_id = KVM_IRQCHIP_PIC_MASTER;
            vm.get_irqchip(&mut other_irqchip).unwrap();
            unsafe { assert_eq!(irqchip.chip.dummy[..], other_irqchip.chip.dummy[..]) };
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn pit2() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        assert!(vm.create_pit2(kvm_pit_config::default()).is_ok());
        if kvm.check_extension(Cap::PitState2) {
            let mut pit2 = kvm_pit_state2::default();
            vm.get_pit2(&mut pit2).unwrap();
            vm.set_pit2(&pit2).unwrap();
            let mut other_pit2 = kvm_pit_state2::default();
            vm.get_pit2(&mut other_pit2).unwrap();
            // Load time will differ, let's overwrite it so we can test equality.
            other_pit2.channels[0].count_load_time = pit2.channels[0].count_load_time;
            other_pit2.channels[1].count_load_time = pit2.channels[1].count_load_time;
            other_pit2.channels[2].count_load_time = pit2.channels[2].count_load_time;
            assert_eq!(pit2, other_pit2);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn clock() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let mut orig = kvm_clock_data::default();
        let mut fudged = kvm_clock_data::default();
        let mut new = kvm_clock_data::default();
        // Get current time.
        vm.get_clock(&mut orig).unwrap();
        // Reset time.
        fudged.clock = 10;
        vm.set_clock(&fudged).unwrap();
        // Get new time.
        vm.get_clock(&mut new).unwrap();
        // Verify new time has progressed but is smaller than orig time.
        assert!(fudged.clock < new.clock);
        assert!(new.clock < orig.clock);
    }

    #[test]
    fn register_ioevent() {
        assert_eq!(std::mem::size_of::<NoDatamatch>(), 0);

        let kvm = Kvm::new().unwrap();
        let vm_fd = kvm.create_vm().unwrap();
        let evtfd = EventFd::new().unwrap();
        vm_fd
            .register_ioevent(&evtfd, &IoeventAddress::Pio(0xf4), NoDatamatch)
            .unwrap();
        vm_fd
            .register_ioevent(&evtfd, &IoeventAddress::Mmio(0x1000), NoDatamatch)
            .unwrap();
        vm_fd
            .register_ioevent(&evtfd, &IoeventAddress::Pio(0xc1), 0x7fu8)
            .unwrap();
        vm_fd
            .register_ioevent(&evtfd, &IoeventAddress::Pio(0xc2), 0x1337u16)
            .unwrap();
        vm_fd
            .register_ioevent(&evtfd, &IoeventAddress::Pio(0xc4), 0xdead_beefu32)
            .unwrap();
        vm_fd
            .register_ioevent(&evtfd, &IoeventAddress::Pio(0xc8), 0xdead_beef_dead_beefu64)
            .unwrap();
    }

    #[test]
    fn irqfd() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        vm.create_irq_chip().unwrap();

        let evtfd1 = EventFd::new().unwrap();
        let evtfd2 = EventFd::new().unwrap();
        let evtfd3 = EventFd::new().unwrap();
        vm.register_irqfd(&evtfd1, 4).unwrap();
        vm.register_irqfd(&evtfd2, 8).unwrap();
        vm.register_irqfd(&evtfd3, 4).unwrap();
        vm.register_irqfd(&evtfd3, 4).unwrap_err();
        vm.register_irqfd(&evtfd3, 5).unwrap_err();

        vm.set_irq_line(4, true).unwrap();
        vm.set_irq_line(4, false).unwrap();

        // Deregister evt3 from IRQ 4.
        vm.deregister_irqfd(&evtfd3, 4).unwrap();
        // Re-registering evt3 for IRQ 4 should now work.
        vm.register_irqfd(&evtfd3, 4).unwrap();
    }

    #[test]
    fn create_vcpu() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        vm.create_vcpu(0).unwrap();
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn reg_test() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let vcpu = vm.create_vcpu(0).unwrap();
        let mut regs = vcpu.get_regs().unwrap();
        regs.rax = 0x1;
        vcpu.set_regs(&regs).unwrap();
        assert!(vcpu.get_regs().unwrap().rax == 0x1);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn sreg_test() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let vcpu = vm.create_vcpu(0).unwrap();
        let mut sregs = vcpu.get_sregs().unwrap();
        sregs.cr0 = 0x1;
        sregs.efer = 0x2;

        vcpu.set_sregs(&sregs).unwrap();
        assert_eq!(vcpu.get_sregs().unwrap().cr0, 0x1);
        assert_eq!(vcpu.get_sregs().unwrap().efer, 0x2);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn fpu_test() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let vcpu = vm.create_vcpu(0).unwrap();
        let mut fpu: kvm_fpu = kvm_fpu {
            fcw: KVM_FPU_CWD as u16,
            mxcsr: KVM_FPU_MXCSR as u32,
            ..Default::default()
        };

        fpu.fcw = KVM_FPU_CWD as u16;
        fpu.mxcsr = KVM_FPU_MXCSR as u32;

        vcpu.set_fpu(&fpu).unwrap();
        assert_eq!(vcpu.get_fpu().unwrap().fcw, KVM_FPU_CWD as u16);
        //The following will fail; kvm related bug; uncomment when bug solved
        //assert_eq!(vcpu.get_fpu().unwrap().mxcsr, KVM_FPU_MXCSR as u32);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn lapic_test() {
        use std::io::Cursor;
        //we might get read of byteorder if we replace 5h3 mem::transmute with something safer
        use self::byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
        //as per https://github.com/torvalds/linux/arch/x86/kvm/lapic.c
        //try to write and read the APIC_ICR (0x300) register which is non-read only and
        //one can simply write to it
        let kvm = Kvm::new().unwrap();
        assert!(kvm.check_extension(Cap::Irqchip));
        let vm = kvm.create_vm().unwrap();
        //the get_lapic ioctl will fail if there is no irqchip created beforehand
        assert!(vm.create_irq_chip().is_ok());
        let vcpu = vm.create_vcpu(0).unwrap();
        let mut klapic: kvm_lapic_state = vcpu.get_lapic().unwrap();

        let reg_offset = 0x300;
        let value = 2 as u32;
        //try to write and read the APIC_ICR	0x300
        let write_slice =
            unsafe { &mut *(&mut klapic.regs[reg_offset..] as *mut [i8] as *mut [u8]) };
        let mut writer = Cursor::new(write_slice);
        writer.write_u32::<LittleEndian>(value).unwrap();
        vcpu.set_lapic(&klapic).unwrap();
        klapic = vcpu.get_lapic().unwrap();
        let read_slice = unsafe { &*(&klapic.regs[reg_offset..] as *const [i8] as *const [u8]) };
        let mut reader = Cursor::new(read_slice);
        assert_eq!(reader.read_u32::<LittleEndian>().unwrap(), value);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[allow(clippy::cast_ptr_alignment)]
    fn msrs_test() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let vcpu = vm.create_vcpu(0).unwrap();
        let mut configured_entry_vec = Vec::<kvm_msr_entry>::new();

        configured_entry_vec.push(kvm_msr_entry {
            index: 0x0000_0174,
            data: 0x0,
            ..Default::default()
        });
        configured_entry_vec.push(kvm_msr_entry {
            index: 0x0000_0175,
            data: 0x1,
            ..Default::default()
        });

        let mut kvm_msrs_wrapper = KvmMsrs::new(configured_entry_vec.len());
        {
            let entries = kvm_msrs_wrapper.as_mut_entries_slice();
            entries.copy_from_slice(&configured_entry_vec);
        }
        vcpu.set_msrs(&kvm_msrs_wrapper).unwrap();

        // Now test that GET_MSRS returns the same.
        let mut returned_kvm_msrs = KvmMsrs::new(configured_entry_vec.len());
        {
            // Configure the struct to say which entries we want.
            let entries = returned_kvm_msrs.as_mut_entries_slice();
            let wanted_kvm_msrs_entries = [
                kvm_msr_entry {
                    index: 0x0000_0174,
                    ..Default::default()
                },
                kvm_msr_entry {
                    index: 0x0000_0175,
                    ..Default::default()
                },
            ];
            entries.copy_from_slice(&wanted_kvm_msrs_entries);
        }
        let nmsrs = vcpu.get_msrs(&mut returned_kvm_msrs).unwrap();
        assert_eq!(nmsrs, configured_entry_vec.len());
        assert_eq!(nmsrs, returned_kvm_msrs.as_original_struct().nmsrs as usize);

        let returned_kvm_msr_entries = returned_kvm_msrs.as_mut_entries_slice();

        for (i, entry) in returned_kvm_msr_entries.iter_mut().enumerate() {
            assert_eq!(entry, &mut configured_entry_vec[i]);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_run_code() {
        use std::io::Write;

        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        // This example based on https://lwn.net/Articles/658511/
        #[rustfmt::skip]
        let code = [
            0xba, 0xf8, 0x03, /* mov $0x3f8, %dx */
            0x00, 0xd8, /* add %bl, %al */
            0x04, b'0', /* add $'0', %al */
            0xee, /* out %al, %dx */
            0xec, /* in %dx, %al */
            0xc6, 0x06, 0x00, 0x80, 0x00, /* movl $0, (0x8000); This generates a MMIO Write.*/
            0x8a, 0x16, 0x00, 0x80, /* movl (0x8000), %dl; This generates a MMIO Read.*/
            0xc6, 0x06, 0x00, 0x20, 0x00, /* movl $0, (0x2000); Dirty one page in guest mem. */
            0xf4, /* hlt */
        ];

        let mem_size = 0x4000;
        let load_addr = mmap_anonymous(mem_size);
        let guest_addr: u64 = 0x1000;
        let slot: u32 = 0;
        let mem_region = kvm_userspace_memory_region {
            slot,
            guest_phys_addr: guest_addr,
            memory_size: mem_size as u64,
            userspace_addr: load_addr as u64,
            flags: KVM_MEM_LOG_DIRTY_PAGES,
        };
        vm.set_user_memory_region(mem_region).unwrap();

        unsafe {
            // Get a mutable slice of `mem_size` from `load_addr`.
            // This is safe because we mapped it before.
            let mut slice = std::slice::from_raw_parts_mut(load_addr, mem_size);
            slice.write_all(&code).unwrap();
        }

        let vcpu_fd = vm.create_vcpu(0).expect("new VcpuFd failed");

        let mut vcpu_sregs = vcpu_fd.get_sregs().expect("get sregs failed");
        assert_ne!(vcpu_sregs.cs.base, 0);
        assert_ne!(vcpu_sregs.cs.selector, 0);
        vcpu_sregs.cs.base = 0;
        vcpu_sregs.cs.selector = 0;
        vcpu_fd.set_sregs(&vcpu_sregs).expect("set sregs failed");

        let mut vcpu_regs = vcpu_fd.get_regs().expect("get regs failed");
        // Set the Instruction Pointer to the guest address where we loaded the code.
        vcpu_regs.rip = guest_addr;
        vcpu_regs.rax = 2;
        vcpu_regs.rbx = 3;
        vcpu_regs.rflags = 2;
        vcpu_fd.set_regs(&vcpu_regs).expect("set regs failed");

        // Variables used for checking if MMIO Read & Write were performed.
        let mut mmio_read = false;
        let mut mmio_write = false;
        loop {
            match vcpu_fd.run().expect("run failed") {
                VcpuExit::IoIn(addr, data) => {
                    assert_eq!(addr, 0x3f8);
                    assert_eq!(data.len(), 1);
                }
                VcpuExit::IoOut(addr, data) => {
                    assert_eq!(addr, 0x3f8);
                    assert_eq!(data.len(), 1);
                    assert_eq!(data[0], b'5');
                }
                VcpuExit::MmioRead(addr, data) => {
                    assert_eq!(addr, 0x8000);
                    assert_eq!(data.len(), 1);
                    mmio_read = true;
                }
                VcpuExit::MmioWrite(addr, data) => {
                    assert_eq!(addr, 0x8000);
                    assert_eq!(data.len(), 1);
                    assert_eq!(data[0], 0);
                    mmio_write = true;
                }
                VcpuExit::Hlt => {
                    // The code snippet dirties 2 pages:
                    // * one when the code itself is loaded in memory;
                    // * and one more from the `movl` that writes to address 0x2000
                    let dirty_pages_bitmap = vm.get_dirty_log(slot, mem_size).unwrap();
                    let dirty_pages: u32 = dirty_pages_bitmap
                        .into_iter()
                        .map(|page| page.count_ones())
                        .sum();
                    assert_eq!(dirty_pages, 2);
                    break;
                }
                r => panic!("unexpected exit reason: {:?}", r),
            }
        }
        assert!(mmio_read);
        assert!(mmio_write);
    }

    fn get_raw_errno<T>(result: super::Result<T>) -> i32 {
        result.err().unwrap().raw_os_error().unwrap()
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[allow(clippy::cognitive_complexity)]
    #[test]
    fn faulty_kvm_fds_test() {
        let badf_errno = libc::EBADF;

        let faulty_kvm = Kvm {
            kvm: unsafe { File::from_raw_fd(-1) },
        };
        assert_eq!(get_raw_errno(faulty_kvm.get_vcpu_mmap_size()), badf_errno);
        let max_cpus = faulty_kvm.get_nr_vcpus();
        assert_eq!(max_cpus, 4);
        assert_eq!(
            get_raw_errno(faulty_kvm.get_supported_cpuid(max_cpus)),
            badf_errno
        );
        assert_eq!(get_raw_errno(faulty_kvm.get_msr_index_list()), badf_errno);
        assert_eq!(
            get_raw_errno(faulty_kvm.get_msr_feature_index_list()),
            badf_errno
        );

        assert_eq!(get_raw_errno(faulty_kvm.create_vm()), badf_errno);
        let faulty_vm_fd = VmFd {
            vm: unsafe { File::from_raw_fd(-1) },
            run_size: 0,
        };
        let invalid_mem_region = kvm_userspace_memory_region {
            slot: 0,
            guest_phys_addr: 0,
            memory_size: 0,
            userspace_addr: 0,
            flags: 0,
        };
        assert_eq!(
            get_raw_errno(faulty_vm_fd.set_user_memory_region(invalid_mem_region)),
            badf_errno
        );
        assert_eq!(get_raw_errno(faulty_vm_fd.set_tss_address(0)), badf_errno);
        assert_eq!(get_raw_errno(faulty_vm_fd.create_irq_chip()), badf_errno);
        assert_eq!(
            get_raw_errno(faulty_vm_fd.create_pit2(kvm_pit_config::default())),
            badf_errno
        );
        let event_fd = EventFd::new().unwrap();
        assert_eq!(
            get_raw_errno(faulty_vm_fd.register_ioevent(&event_fd, &IoeventAddress::Pio(0), 0u64)),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vm_fd.get_irqchip(&mut kvm_irqchip::default())),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vm_fd.set_irqchip(&kvm_irqchip::default())),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vm_fd.get_clock(&mut kvm_clock_data::default())),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vm_fd.set_clock(&kvm_clock_data::default())),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vm_fd.get_pit2(&mut kvm_pit_state2::default())),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vm_fd.set_pit2(&kvm_pit_state2::default())),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vm_fd.register_irqfd(&event_fd, 0)),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vm_fd.deregister_irqfd(&event_fd, 0)),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vm_fd.set_irq_line(1, true)),
            badf_errno
        );
        assert_eq!(get_raw_errno(faulty_vm_fd.create_vcpu(0)), badf_errno);
        let faulty_vcpu_fd = VcpuFd {
            vcpu: unsafe { File::from_raw_fd(-1) },
            kvm_run_ptr: KvmRunWrapper::new(10),
        };
        assert_eq!(get_raw_errno(faulty_vcpu_fd.get_regs()), badf_errno);
        assert_eq!(
            get_raw_errno(faulty_vcpu_fd.set_regs(&unsafe { std::mem::zeroed() })),
            badf_errno
        );
        assert_eq!(get_raw_errno(faulty_vcpu_fd.get_sregs()), badf_errno);
        assert_eq!(
            get_raw_errno(faulty_vcpu_fd.set_sregs(&unsafe { std::mem::zeroed() })),
            badf_errno
        );
        assert_eq!(get_raw_errno(faulty_vcpu_fd.get_fpu()), badf_errno);
        assert_eq!(
            get_raw_errno(faulty_vcpu_fd.set_fpu(&unsafe { std::mem::zeroed() })),
            badf_errno
        );
        assert_eq!(get_raw_errno(faulty_vcpu_fd.get_cpuid2(1)), badf_errno);
        assert_eq!(
            get_raw_errno(
                faulty_vcpu_fd.set_cpuid2(
                    &Kvm::new()
                        .unwrap()
                        .get_supported_cpuid(MAX_KVM_CPUID_ENTRIES)
                        .unwrap()
                )
            ),
            badf_errno
        );
        // kvm_lapic_state does not implement debug by default so we cannot
        // use unwrap_err here.
        assert!(faulty_vcpu_fd.get_lapic().is_err());
        assert_eq!(
            get_raw_errno(faulty_vcpu_fd.set_lapic(&unsafe { std::mem::zeroed() })),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vcpu_fd.get_msrs(&mut KvmMsrs::new(1))),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vcpu_fd.set_msrs(&KvmMsrs::new(1))),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vcpu_fd.get_mp_state(&mut kvm_mp_state::default())),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vcpu_fd.set_mp_state(kvm_mp_state::default())),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vcpu_fd.get_xsave(&mut kvm_xsave::default())),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vcpu_fd.set_xsave(&kvm_xsave::default())),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vcpu_fd.get_xcrs(&mut kvm_xcrs::default())),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vcpu_fd.set_xcrs(&kvm_xcrs::default())),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vcpu_fd.get_debug_regs(&mut kvm_debugregs::default())),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vcpu_fd.set_debug_regs(&kvm_debugregs::default())),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vcpu_fd.get_vcpu_events(&mut kvm_vcpu_events::default())),
            badf_errno
        );
        assert_eq!(
            get_raw_errno(faulty_vcpu_fd.set_vcpu_events(&kvm_vcpu_events::default())),
            badf_errno
        );
        assert_eq!(get_raw_errno(faulty_vcpu_fd.run()), badf_errno);
    }

    #[test]
    fn test_kvm_api_version() {
        let kvm = Kvm::new().unwrap();
        assert_eq!(kvm.get_api_version(), KVM_API_VERSION as i32);
    }

    #[test]
    fn test_cpuid_clone() {
        let kvm = Kvm::new().unwrap();
        let cpuid_1 = kvm.get_supported_cpuid(MAX_KVM_CPUID_ENTRIES).unwrap();
        let mut cpuid_2 = cpuid_1.clone();
        assert!(cpuid_1 == cpuid_2);
        cpuid_2 = unsafe { std::mem::zeroed() };
        assert!(cpuid_1 != cpuid_2);
    }

    #[test]
    fn test_cpu_id_new() {
        let num_entries = 5;
        let mut cpuid = CpuId::new(num_entries);

        // check that the cpuid contains `num_entries` empty entries
        assert!(unsafe { &*(cpuid.as_ptr()) }.nent == num_entries as u32);
        for entry in cpuid.as_mut_entries_slice() {
            assert!(
                *entry
                    == kvm_cpuid_entry2 {
                        function: 0,
                        index: 0,
                        flags: 0,
                        eax: 0,
                        ebx: 0,
                        ecx: 0,
                        edx: 0,
                        padding: [0, 0, 0],
                    }
            );
        }
    }

    #[test]
    fn test_cpu_id_from_entries() {
        let num_entries = 4;
        let mut cpuid = CpuId::new(num_entries);

        // add entry
        let mut entries = cpuid.as_mut_entries_slice().to_vec();
        let new_entry = kvm_cpuid_entry2 {
            function: 0x4,
            index: 0,
            flags: 1,
            eax: 0b110_0000,
            ebx: 0,
            ecx: 0,
            edx: 0,
            padding: [0, 0, 0],
        };
        entries.insert(0, new_entry);
        cpuid = CpuId::from_entries(&entries);

        // check that the cpuid contains the new entry
        assert!(unsafe { &*(cpuid.as_ptr()) }.nent == (num_entries + 1) as u32);
        assert!(cpuid.as_mut_entries_slice().len() == num_entries + 1);
        assert!(cpuid.as_mut_entries_slice()[0] == new_entry);
    }

    #[test]
    fn test_cpu_id_mut_entries_slice() {
        let num_entries = 5;
        let mut cpuid = CpuId::new(num_entries);

        {
            // check that the CpuId has been initialized correctly:
            // there should be `num_entries` empty entries
            assert!(unsafe { &*(cpuid.as_ptr()) }.nent == num_entries as u32);

            let entries = cpuid.as_mut_entries_slice();
            assert!(entries.len() == num_entries);
            for entry in entries.iter() {
                assert!(
                    *entry
                        == kvm_cpuid_entry2 {
                            function: 0,
                            index: 0,
                            flags: 0,
                            eax: 0,
                            ebx: 0,
                            ecx: 0,
                            edx: 0,
                            padding: [0, 0, 0],
                        }
                );
            }
        }

        let new_entry = kvm_cpuid_entry2 {
            function: 0x4,
            index: 0,
            flags: 1,
            eax: 0b110_0000,
            ebx: 0,
            ecx: 0,
            edx: 0,
            padding: [0, 0, 0],
        };
        // modify the first entry
        cpuid.as_mut_entries_slice()[0] = new_entry;
        // test that the first entry has been modified in the underlying structure
        assert!(cpuid.as_entries_slice()[0] == new_entry);
    }
}
