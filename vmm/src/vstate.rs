// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

extern crate kvm_bindings;

use libc::{c_int, c_void, siginfo_t, EINVAL};
use std::cell::Cell;
use std::io;
use std::ops::Deref;
use std::result;
use std::sync::atomic::{fence, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::thread;

use super::{KvmContext, TimestampUs};
use arch;
#[cfg(target_arch = "x86_64")]
use cpuid::{c3, filter_cpuid, t2};
use default_syscalls;
use kvm::*;
use kvm_bindings::{
    kvm_clock_data, kvm_debugregs, kvm_irqchip, kvm_lapic_state, kvm_mp_state, kvm_pit_config,
    kvm_pit_state2, kvm_regs, kvm_sregs, kvm_userspace_memory_region, kvm_vcpu_events, kvm_xcrs,
    kvm_xsave, KVM_PIT_SPEAKER_DUMMY,
};
use logger::{LogOption, Metric, LOGGER, METRICS};
use memory_model::{GuestAddress, GuestMemory, GuestMemoryError};
use sys_util::{register_vcpu_signal_handler, EventFd, Killable};
#[cfg(target_arch = "x86_64")]
use vmm_config::machine_config::CpuFeaturesTemplate;
use vmm_config::machine_config::VmConfig;

const KVM_MEM_LOG_DIRTY_PAGES: u32 = 0x1;

#[cfg(target_arch = "x86_64")]
const MAGIC_IOPORT_SIGNAL_GUEST_BOOT_COMPLETE: u64 = 0x03f0;
#[cfg(target_arch = "aarch64")]
const MAGIC_IOPORT_SIGNAL_GUEST_BOOT_COMPLETE: u64 = 0x40000000;
const MAGIC_VALUE_SIGNAL_GUEST_BOOT_COMPLETE: u8 = 123;

pub(crate) const VCPU_RTSIG_OFFSET: i32 = 0;

/// Errors associated with the wrappers over KVM ioctls.
#[derive(Debug)]
pub enum Error {
    #[cfg(target_arch = "x86_64")]
    /// A call to cpuid instruction failed.
    CpuId(cpuid::Error),
    /// Invalid guest memory configuration.
    GuestMemory(GuestMemoryError),
    /// Retrieving supported guest MSRs fails.
    GuestMSRs(std::io::Error),
    /// Hyperthreading flag is not initialized.
    HTNotInitialized,
    /// vCPU count is not initialized.
    VcpuCountNotInitialized,
    /// Cannot open the Vcpu file descriptor.
    VcpuFd(io::Error),
    /// Failed to get KVM vcpu debug regs.
    VcpuGetDebugRegs(io::Error),
    /// Failed to get KVM vcpu lapic.
    VcpuGetLapic(io::Error),
    /// Failed to get KVM vcpu mp state.
    VcpuGetMpState(io::Error),
    /// Failed to get KVM vcpu msrs.
    VcpuGetMsrs(io::Error),
    /// Failed to get KVM vcpu regs.
    VcpuGetRegs(io::Error),
    /// Failed to get KVM vcpu sregs.
    VcpuGetSregs(io::Error),
    /// Failed to get KVM vcpu evenct.
    VcpuGetVcpuEvents(io::Error),
    /// Failed to get KVM vcpu xcrs.
    VcpuGetXcrs(io::Error),
    /// Failed to get KVM vcpu xsave.
    VcpuGetXsave(io::Error),
    /// Vcpu is in an unexpected state.
    VcpuInvalidState,
    /// Vcpu response timed out.
    VcpuResponseTimeout,
    /// Failed to set KVM vcpu cpuid.
    VcpuSetCpuid(io::Error),
    /// Failed to set KVM vcpu debug regs.
    VcpuSetDebugRegs(io::Error),
    /// Failed to set KVM vcpu lapic.
    VcpuSetLapic(io::Error),
    /// Failed to set KVM vcpu mp state.
    VcpuSetMpState(io::Error),
    /// Failed to set KVM vcpu msrs.
    VcpuSetMsrs(io::Error),
    /// Failed to set KVM vcpu regs.
    VcpuSetRegs(io::Error),
    /// Failed to set KVM vcpu sregs.
    VcpuSetSregs(io::Error),
    /// Failed to set KVM vcpu evenct.
    VcpuSetVcpuEvents(io::Error),
    /// Failed to set KVM vcpu xcrs.
    VcpuSetXcrs(io::Error),
    /// Failed to set KVM vcpu xsave.
    VcpuSetXsave(io::Error),
    /// Cannot spawn a new vcpu thread.
    VcpuSpawn(io::Error),
    /// Failed to get KVM vm pit state.
    VmGetPit2(io::Error),
    /// Failed to get KVM vm clock.
    VmGetClock(io::Error),
    /// Failed to get KVM vm irqchip.
    VmGetIrqChip(io::Error),
    /// Cannot open the VM file descriptor.
    VmFd(io::Error),
    /// Failed to set KVM vm pit state.
    VmSetPit2(io::Error),
    /// Failed to set KVM vm clock.
    VmSetClock(io::Error),
    /// Failed to set KVM vm irqchip.
    VmSetIrqChip(io::Error),
    /// Cannot configure the microvm.
    VmSetup(io::Error),
    /// The number of configured slots is bigger than the maximum reported by KVM.
    NotEnoughMemorySlots,
    /// Failed to signal Vcpu.
    SignalVcpu(std::io::Error),
    #[cfg(target_arch = "x86_64")]
    /// Cannot set the local interruption due to bad configuration.
    LocalIntConfiguration(arch::x86_64::interrupts::Error),
    /// Cannot set the memory regions.
    SetUserMemoryRegion(io::Error),
    #[cfg(target_arch = "x86_64")]
    /// Error configuring the MSR registers
    MSRSConfiguration(arch::x86_64::msr::Error),
    #[cfg(target_arch = "aarch64")]
    /// Error configuring the general purpose aarch64 registers.
    REGSConfiguration(arch::aarch64::regs::Error),
    #[cfg(target_arch = "x86_64")]
    /// Error configuring the general purpose registers
    REGSConfiguration(arch::x86_64::regs::Error),
    #[cfg(target_arch = "x86_64")]
    /// Error configuring the special registers
    SREGSConfiguration(arch::x86_64::regs::Error),
    #[cfg(target_arch = "x86_64")]
    /// Error configuring the floating point related registers
    FPUConfiguration(arch::x86_64::regs::Error),
    /// Unexpected KVM_RUN exit reason
    VcpuUnhandledKvmExit,
    #[cfg(target_arch = "aarch64")]
    /// Error setting up the global interrupt controller.
    SetupGIC(arch::aarch64::gic::Error),
    #[cfg(target_arch = "aarch64")]
    /// Error getting the Vcpu preferred target on Arm.
    VcpuArmPreferredTarget(io::Error),
    #[cfg(target_arch = "aarch64")]
    /// Error doing Vcpu Init on Arm.
    VcpuArmInit(io::Error),
}

pub type Result<T> = result::Result<T, Error>;

/// A wrapper around creating and using a VM.
pub struct Vm {
    fd: VmFd,
    guest_mem: Option<GuestMemory>,

    // X86 specific fields.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    supported_cpuid: CpuId,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    supported_msrs: MsrList,

    // Arm specific fields.
    // On aarch64 we need to keep around the fd obtained by creating the VGIC device.
    #[cfg(target_arch = "aarch64")]
    irqchip_handle: Option<DeviceFd>,
}

impl Vm {
    /// Constructs a new `Vm` using the given `Kvm` instance.
    pub fn new(kvm: &Kvm) -> Result<Self> {
        //create fd for interacting with kvm-vm specific functions
        let vm_fd = kvm.create_vm().map_err(Error::VmFd)?;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let supported_cpuid = kvm
            .get_supported_cpuid(MAX_KVM_CPUID_ENTRIES)
            .map_err(Error::VmFd)?;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let supported_msrs =
            arch::x86_64::msr::supported_guest_msrs(kvm).map_err(Error::GuestMSRs)?;

        Ok(Vm {
            fd: vm_fd,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            supported_cpuid,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            supported_msrs,
            guest_mem: None,
            #[cfg(target_arch = "aarch64")]
            irqchip_handle: None,
        })
    }

    /// Returns a ref to the supported `CpuId` for this Vm.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn supported_cpuid(&self) -> &CpuId {
        &self.supported_cpuid
    }

    /// Returns a ref to the supported `MsrList` for this Vm.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn supported_msrs(&self) -> &MsrList {
        &self.supported_msrs
    }

    /// Initializes the guest memory.
    pub fn memory_init(&mut self, guest_mem: GuestMemory, kvm_context: &KvmContext) -> Result<()> {
        if guest_mem.num_regions() > kvm_context.max_memslots() {
            return Err(Error::NotEnoughMemorySlots);
        }
        guest_mem
            .with_regions(|index, guest_addr, size, host_addr| {
                info!("Guest memory starts at {:x?}", host_addr);

                let flags = if LOGGER.flags() & LogOption::LogDirtyPages as usize > 0 {
                    KVM_MEM_LOG_DIRTY_PAGES
                } else {
                    0
                };

                let memory_region = kvm_userspace_memory_region {
                    slot: index as u32,
                    guest_phys_addr: guest_addr.offset() as u64,
                    memory_size: size as u64,
                    userspace_addr: host_addr as u64,
                    flags,
                };
                self.fd.set_user_memory_region(memory_region)
            })
            .map_err(Error::SetUserMemoryRegion)?;
        self.guest_mem = Some(guest_mem);

        #[cfg(target_arch = "x86_64")]
        self.fd
            .set_tss_address(GuestAddress(arch::x86_64::layout::KVM_TSS_ADDRESS).offset())
            .map_err(Error::VmSetup)?;

        Ok(())
    }

    /// Creates the irq chip and an in-kernel device model for the PIT.
    #[cfg(target_arch = "x86_64")]
    pub fn setup_irqchip(&self) -> Result<()> {
        self.fd.create_irq_chip().map_err(Error::VmSetup)?;
        let mut pit_config = kvm_pit_config::default();
        // We need to enable the emulation of a dummy speaker port stub so that writing to port 0x61
        // (i.e. KVM_SPEAKER_BASE_ADDRESS) does not trigger an exit to user space.
        pit_config.flags = KVM_PIT_SPEAKER_DUMMY;
        self.fd.create_pit2(pit_config).map_err(Error::VmSetup)
    }

    /// Creates the GIC (Global Interrupt Controller).
    #[cfg(target_arch = "aarch64")]
    pub fn setup_irqchip(&mut self, vcpu_count: u8) -> Result<()> {
        self.irqchip_handle =
            Some(arch::aarch64::gic::create_gicv3(&self.fd, vcpu_count).map_err(Error::SetupGIC)?);
        Ok(())
    }

    /// Gets a reference to the guest memory owned by this VM.
    ///
    /// Note that `GuestMemory` does not include any device memory that may have been added after
    /// this VM was constructed.
    pub fn get_memory(&self) -> Option<&GuestMemory> {
        self.guest_mem.as_ref()
    }

    /// Gets a reference to the kvm file descriptor owned by this VM.
    ///
    pub fn get_fd(&self) -> &VmFd {
        &self.fd
    }

    pub fn save_state(&self) -> Result<VmState> {
        let mut pitstate = kvm_pit_state2::default();
        self.fd.get_pit2(&mut pitstate).map_err(Error::VmGetPit2)?;

        let mut clock = kvm_clock_data::default();
        self.fd.get_clock(&mut clock).map_err(Error::VmGetClock)?;
        // This bit is not accepted in SET_CLOCK, clear it.
        clock.flags &= !KVM_CLOCK_TSC_STABLE;

        let mut pic_master = kvm_irqchip::default();
        pic_master.chip_id = KVM_IRQCHIP_PIC_MASTER;
        self.fd
            .get_irqchip(&mut pic_master)
            .map_err(Error::VmGetIrqChip)?;

        let mut pic_slave = kvm_irqchip::default();
        pic_slave.chip_id = KVM_IRQCHIP_PIC_SLAVE;
        self.fd
            .get_irqchip(&mut pic_slave)
            .map_err(Error::VmGetIrqChip)?;

        let mut ioapic = kvm_irqchip::default();
        ioapic.chip_id = KVM_IRQCHIP_IOAPIC;
        self.fd
            .get_irqchip(&mut ioapic)
            .map_err(Error::VmGetIrqChip)?;

        Ok(VmState {
            pitstate,
            clock,
            pic_master,
            pic_slave,
            ioapic,
        })
    }

    pub fn restore_state(&self, state: &VmState) -> Result<()> {
        self.fd
            .set_pit2(&state.pitstate)
            .map_err(Error::VmSetPit2)?;
        self.fd.set_clock(&state.clock).map_err(Error::VmSetClock)?;
        self.fd
            .set_irqchip(&state.pic_master)
            .map_err(Error::VmSetIrqChip)?;
        self.fd
            .set_irqchip(&state.pic_slave)
            .map_err(Error::VmSetIrqChip)?;
        self.fd
            .set_irqchip(&state.ioapic)
            .map_err(Error::VmSetIrqChip)?;
        Ok(())
    }
}

/// Structure holding VM kvm state.
#[derive(Default)]
#[repr(C)]
pub struct VmState {
    pitstate: kvm_pit_state2,
    clock: kvm_clock_data,
    pic_master: kvm_irqchip,
    pic_slave: kvm_irqchip,
    ioapic: kvm_irqchip,
}

/// List of events that the Vcpu can receive.
pub enum VcpuEvent {
    /// Event that should pause the Vcpu.
    Pause,
    /// Event that should resume the Vcpu.
    Resume,
}

/// List of responses that the Vcpu reports.
#[derive(Debug, PartialEq)]
pub enum VcpuResponse {
    /// Vcpu is paused.
    Paused,
    /// Vcpu is resumed.
    Resumed,
}

// Rustacean implementation of a state machine.
//
// `StateStruct<T>` is a wrapper over `T` that also encodes state information for `T`.
//
// Each state for `T` is represented by a `StateFn<T>` which is a function that acts as
// the state handler for that particular state of `T`.
//
// `StateFn<T>` returns another `StateStruct<T>` thus each state gets clearly defined
// transitions to other states.
type StateFn<T> = fn(&mut T) -> StateStruct<T>;
struct StateStruct<T> {
    function: StateFn<T>,
    end_state: bool,
}

impl<T> StateStruct<T> {
    // Creates a new state wrapper.
    fn new(function: StateFn<T>, end_state: bool) -> StateStruct<T> {
        StateStruct {
            function,
            end_state,
        }
    }
    // Creates a new state wrapper that has further possible transitions.
    fn next(function: StateFn<T>) -> StateStruct<T> {
        StateStruct::new(function, false)
    }
    // Creates a new state wrapper that has no further transitions. The state machine
    // will finish after running this handler.
    fn finish(function: StateFn<T>) -> StateStruct<T> {
        StateStruct::new(function, true)
    }
}

// Implement Deref of `StateStruct<T>` so that we can directly call its underlying state handler.
impl<T> Deref for StateStruct<T> {
    type Target = StateFn<T>;
    fn deref(&self) -> &Self::Target {
        &self.function
    }
}

enum VcpuHandleState {
    Empty,
    Inactive(Box<Vcpu>),
    Active(thread::JoinHandle<()>),
}

/// Wrapper over Vcpu that hides the underlying interactions with the Vcpu thread.
pub struct VcpuHandle {
    event_sender: Sender<VcpuEvent>,
    response_receiver: Receiver<VcpuResponse>,
    state: VcpuHandleState,
}

impl VcpuHandle {
    /// Constructs a new vcpu handle for `vm`. This will also create the underlying `Vcpu`.
    ///
    /// # Arguments
    ///
    /// * `id` - Represents the CPU number between [0, max vcpus).
    /// * `vm` - The virtual machine this vcpu will get attached to.
    /// * `io_bus` - The io-bus used to access port-io devices.
    /// * `exit_evt` - An `EventFd` that will be written into when this vcpu exits.
    /// * `create_ts` - A timestamp used by the vcpu to calculate its lifetime.
    pub fn new(
        id: u8,
        vm: &Vm,
        io_bus: devices::Bus,
        exit_evt: EventFd,
        create_ts: TimestampUs,
    ) -> Result<VcpuHandle> {
        let (event_sender, event_receiver) = channel();
        let (response_sender, response_receiver) = channel();

        let vcpu = Vcpu::new(
            id,
            vm,
            io_bus,
            exit_evt,
            event_receiver,
            response_sender,
            create_ts,
        )?;

        Ok(VcpuHandle {
            event_sender,
            response_receiver,
            state: VcpuHandleState::Inactive(Box::new(vcpu)),
        })
    }

    pub fn validate_active(&self) -> Result<()> {
        match self.state {
            VcpuHandleState::Active(_) => (),
            _ => Err(Error::VcpuInvalidState)?,
        };
        Ok(())
    }

    /// Configures a vcpu for a fresh boot of the guest.
    ///
    /// # Arguments
    ///
    /// * `machine_config` - Specifies necessary info used for the CPUID configuration.
    /// * `kernel_load_addr` - Offset from `guest_mem` at which the kernel is loaded.
    /// * `vm` - The virtual machine this vcpu will get attached to.
    pub fn configure_vcpu(
        &mut self,
        machine_config: &VmConfig,
        kernel_load_addr: GuestAddress,
        vm: &Vm,
    ) -> Result<()> {
        match self.state {
            VcpuHandleState::Inactive(ref mut vcpu) => {
                vcpu.configure(machine_config, kernel_load_addr, vm)?;
            }
            _ => Err(Error::VcpuInvalidState)?,
        };
        Ok(())
    }

    /// Starts the vcpu on its own thread.
    ///
    /// # Arguments
    ///
    /// * `seccomp_level` - Seccomp level applied to the vcpu.
    /// * `mmio_bus` - The mmio-bus used to access memory-mapped devices.
    pub fn start_vcpu(&mut self, seccomp_level: u32, mmio_bus: Option<devices::Bus>) -> Result<()> {
        // Need to swap to be able to extract vcpu and re-assign inside match.
        let mut state = VcpuHandleState::Empty;
        std::mem::swap(&mut self.state, &mut state);
        match state {
            VcpuHandleState::Inactive(mut vcpu) => {
                if let Some(mmio_bus) = mmio_bus {
                    vcpu.set_mmio_bus(mmio_bus);
                }
                let cpu_id = vcpu.cpu_index();
                let thread = thread::Builder::new()
                    .name(format!("fc_vcpu{}", cpu_id))
                    .spawn(move || {
                        vcpu.run(seccomp_level);
                    })
                    .map_err(Error::VcpuSpawn)?;
                self.state = VcpuHandleState::Active(thread);
            }
            _ => {
                // Revert the swap.
                std::mem::swap(&mut self.state, &mut state);
                Err(Error::VcpuInvalidState)?;
            }
        };

        Ok(())
    }

    pub fn send_event(&self, event: VcpuEvent) -> Result<()> {
        let thread = match self.state {
            VcpuHandleState::Active(ref thread) => thread,
            _ => Err(Error::VcpuInvalidState)?,
        };
        // Use expect() to crash if the other thread closed this channel.
        self.event_sender
            .send(event)
            .expect("event sender channel closed on vcpu end.");
        thread.kill(VCPU_RTSIG_OFFSET).map_err(Error::SignalVcpu)?;
        Ok(())
    }

    pub fn response_receiver(&self) -> &Receiver<VcpuResponse> {
        &self.response_receiver
    }

    /// Will block until thread is joined, make sure the vcpu has exited.
    #[allow(dead_code)]
    pub fn join_vcpu_thread(self) -> Result<()> {
        match self.state {
            // Use expect() to crash the other thread had panicked.
            VcpuHandleState::Active(thread) => thread.join().expect("vcpu thread panicked"),
            _ => Err(Error::VcpuInvalidState)?,
        };
        Ok(())
    }
}

/// A wrapper around creating and using a kvm-based VCPU.
pub struct Vcpu {
    #[cfg(target_arch = "x86_64")]
    cpuid: CpuId,
    fd: VcpuFd,
    id: u8,
    io_bus: devices::Bus,
    mmio_bus: Option<devices::Bus>,
    exit_evt: EventFd,
    event_receiver: Receiver<VcpuEvent>,
    response_sender: Sender<VcpuResponse>,
    create_ts: TimestampUs,
    msr_list: MsrList,
}

// Using this for easier explicit type-casting to help IDEs interpret the code.
type VcpuCell = Cell<Option<*mut Vcpu>>;

impl Vcpu {
    thread_local!(static TLS_VCPU_PTR: VcpuCell = Cell::new(None));

    /// Associates `self` with the current thread.
    ///
    /// Is a prerequisite to succesfully run `run_on_thread_local_vcpu()` on the current thread.
    fn init_thread_local_data(&mut self) {
        Self::TLS_VCPU_PTR.with(|cell: &VcpuCell| {
            if cell.get().is_some() {
                panic!("Thread already has a copy of a `Vcpu` pointer in its TLS.");
            }
            cell.set(Some(self as *mut Vcpu));
        });
    }

    /// Deassociates `self` from the current thread.
    ///
    /// Should be called if the current `self` had called `init_thread_local_data()` and
    /// now needs to move to a different thread.
    ///
    /// Returns EINVAL if `self` was not previously associated with the current thread.
    fn reset_thread_local_data(&mut self) -> io::Result<()> {
        // Best-effort to clean up TLS. If the `Vcpu` was moved to another thread
        // _before_ running this, then there is nothing we can do.
        let mut tls_contains_self = false;
        Self::TLS_VCPU_PTR.with(|cell: &VcpuCell| {
            if let Some(vcpu_ptr) = cell.get() {
                if vcpu_ptr == self as *mut Vcpu {
                    tls_contains_self = true;
                }
            }
        });

        if tls_contains_self {
            Self::TLS_VCPU_PTR.with(|cell: &VcpuCell| cell.take());
            Ok(())
        } else {
            Err(io::Error::from_raw_os_error(EINVAL))
        }
    }

    /// Runs `func` for the `Vcpu` associated with the current thread.
    ///
    /// It requires that `init_thread_local_data()` was run on this thread.
    ///
    /// Returns EINVAL if there is no `Vcpu` associated with the current thread.
    fn run_on_thread_local_vcpu<F>(func: F) -> io::Result<()>
    where
        F: FnOnce(&mut Vcpu),
    {
        Self::TLS_VCPU_PTR.with(|cell: &VcpuCell| {
            if let Some(vcpu_ptr) = cell.get() {
                let vcpu_ref: &mut Vcpu = unsafe { &mut *vcpu_ptr };
                func(vcpu_ref);
                Ok(())
            } else {
                Err(io::Error::from_raw_os_error(EINVAL))
            }
        })
    }

    /// Constructs a new VCPU for `vm`.
    ///
    /// # Arguments
    ///
    /// * `id` - Represents the CPU number between [0, max vcpus).
    /// * `vm` - The virtual machine this vcpu will get attached to.
    /// * `io_bus` - The io-bus used to access port-io devices.
    /// * `exit_evt` - An `EventFd` that will be written into when this vcpu exits.
    /// * `event_receiver` - A `Receiver<VcpuEvent>` that the vpcu will listen on for commands.
    /// * `response_sender` - A `Sender<VcpuResponse>` for the vcpu to send back commands responses.
    /// * `create_ts` - A timestamp used by the vcpu to calculate its lifetime.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u8,
        vm: &Vm,
        io_bus: devices::Bus,
        exit_evt: EventFd,
        event_receiver: Receiver<VcpuEvent>,
        response_sender: Sender<VcpuResponse>,
        create_ts: TimestampUs,
    ) -> Result<Self> {
        let kvm_vcpu = vm.fd.create_vcpu(id).map_err(Error::VcpuFd)?;

        // Initially the cpuid per vCPU is the one supported by this VM.
        Ok(Vcpu {
            #[cfg(target_arch = "x86_64")]
            cpuid: vm.supported_cpuid().clone(),
            fd: kvm_vcpu,
            id,
            io_bus,
            mmio_bus: None,
            exit_evt,
            event_receiver,
            response_sender,
            create_ts,
            msr_list: vm.supported_msrs().clone(),
        })
    }

    /// Returns the cpu index as seen by the guest OS.
    pub fn cpu_index(&self) -> u8 {
        self.id
    }

    /// Sets a MMIO bus for this vcpu.
    pub fn set_mmio_bus(&mut self, mmio_bus: devices::Bus) {
        self.mmio_bus = Some(mmio_bus);
    }

    #[cfg(target_arch = "x86_64")]
    /// Configures a x86_64 specific vcpu and should be called once per vcpu.
    ///
    /// # Arguments
    ///
    /// * `machine_config` - Specifies necessary info used for the CPUID configuration.
    /// * `kernel_start_addr` - Offset from `guest_mem` at which the kernel starts.
    /// * `vm` - The virtual machine this vcpu will get attached to.
    pub fn configure(
        &mut self,
        machine_config: &VmConfig,
        kernel_start_addr: GuestAddress,
        vm: &Vm,
    ) -> Result<()> {
        filter_cpuid(
            self.id,
            machine_config
                .vcpu_count
                .ok_or(Error::VcpuCountNotInitialized)?,
            machine_config.ht_enabled.ok_or(Error::HTNotInitialized)?,
            &mut self.cpuid,
        )
        .map_err(Error::CpuId)?;

        if let Some(template) = machine_config.cpu_template {
            match template {
                CpuFeaturesTemplate::T2 => t2::set_cpuid_entries(self.cpuid.as_mut_entries_slice()),
                CpuFeaturesTemplate::C3 => c3::set_cpuid_entries(self.cpuid.as_mut_entries_slice()),
            }
        }

        self.fd
            .set_cpuid2(&self.cpuid)
            .map_err(Error::VcpuSetCpuid)?;

        arch::x86_64::msr::setup_msrs_for_boot(&self.fd).map_err(Error::MSRSConfiguration)?;
        // Safe to unwrap because this method is called after the VM is configured
        let vm_memory = vm
            .get_memory()
            .ok_or(Error::GuestMemory(GuestMemoryError::MemoryNotInitialized))?;
        arch::x86_64::regs::setup_regs(&self.fd, kernel_start_addr.offset() as u64)
            .map_err(Error::REGSConfiguration)?;
        arch::x86_64::regs::setup_fpu(&self.fd).map_err(Error::FPUConfiguration)?;
        arch::x86_64::regs::setup_sregs(vm_memory, &self.fd).map_err(Error::SREGSConfiguration)?;
        arch::x86_64::interrupts::set_lint(&self.fd).map_err(Error::LocalIntConfiguration)?;
        Ok(())
    }

    #[cfg(target_arch = "aarch64")]
    /// Configures an aarch64 specific vcpu.
    ///
    /// # Arguments
    ///
    /// * `_machine_config` - Specifies necessary info used for the CPUID configuration.
    /// * `kernel_load_addr` - Offset from `guest_mem` at which the kernel is loaded.
    /// * `vm` - The virtual machine this vcpu will get attached to.
    pub fn configure(
        &mut self,
        _machine_config: &VmConfig,
        kernel_load_addr: GuestAddress,
        vm: &Vm,
    ) -> Result<()> {
        let vm_memory = vm
            .get_memory()
            .ok_or(Error::GuestMemory(GuestMemoryError::MemoryNotInitialized))?;

        let mut kvi: kvm_bindings::kvm_vcpu_init = kvm_bindings::kvm_vcpu_init::default();

        // This reads back the kernel's preferred target type.
        vm.fd
            .get_preferred_target(&mut kvi)
            .map_err(Error::VcpuArmPreferredTarget)?;
        // We already checked that the capability is supported.
        kvi.features[0] |= 1 << kvm_bindings::KVM_ARM_VCPU_PSCI_0_2;
        // Non-boot cpus are powered off initially.
        if self.id > 0 {
            kvi.features[0] |= 1 << kvm_bindings::KVM_ARM_VCPU_POWER_OFF;
        }

        self.fd.vcpu_init(&kvi).map_err(Error::VcpuArmInit)?;
        arch::aarch64::regs::setup_regs(&self.fd, self.id, kernel_load_addr.offset(), vm_memory)
            .map_err(Error::REGSConfiguration)?;
        Ok(())
    }

    fn check_boot_complete_signal(&self, addr: u64, data: &[u8]) {
        if addr == MAGIC_IOPORT_SIGNAL_GUEST_BOOT_COMPLETE
            && data[0] == MAGIC_VALUE_SIGNAL_GUEST_BOOT_COMPLETE
        {
            super::Vmm::log_boot_time(&self.create_ts);
        }
    }

    /// Runs the vCPU in KVM context and handles the kvm exit reason.
    ///
    /// Returns Result<bool> where the bool signifies whether KVM_RUN returned EINTR.
    fn run_emulation(&mut self) -> Result<bool> {
        match self.fd.run() {
            Ok(run) => match run {
                VcpuExit::IoIn(addr, data) => {
                    self.io_bus.read(u64::from(addr), data);
                    METRICS.vcpu.exit_io_in.inc();
                    Ok(false)
                }
                VcpuExit::IoOut(addr, data) => {
                    #[cfg(target_arch = "x86_64")]
                    self.check_boot_complete_signal(u64::from(addr), data);

                    self.io_bus.write(u64::from(addr), data);
                    METRICS.vcpu.exit_io_out.inc();
                    Ok(false)
                }
                VcpuExit::MmioRead(addr, data) => {
                    if let Some(ref mmio_bus) = self.mmio_bus {
                        mmio_bus.read(addr, data);
                        METRICS.vcpu.exit_mmio_read.inc();
                    }
                    Ok(false)
                }
                VcpuExit::MmioWrite(addr, data) => {
                    if let Some(ref mmio_bus) = self.mmio_bus {
                        #[cfg(target_arch = "aarch64")]
                        self.check_boot_complete_signal(addr, data);

                        mmio_bus.write(addr, data);
                        METRICS.vcpu.exit_mmio_write.inc();
                    }
                    Ok(false)
                }
                VcpuExit::Hlt => {
                    info!("Received KVM_EXIT_HLT signal");
                    Err(Error::VcpuUnhandledKvmExit)
                }
                VcpuExit::Shutdown => {
                    info!("Received KVM_EXIT_SHUTDOWN signal");
                    Err(Error::VcpuUnhandledKvmExit)
                }
                // Documentation specifies that below kvm exits are considered
                // errors.
                VcpuExit::FailEntry => {
                    METRICS.vcpu.failures.inc();
                    error!("Received KVM_EXIT_FAIL_ENTRY signal");
                    Err(Error::VcpuUnhandledKvmExit)
                }
                VcpuExit::InternalError => {
                    METRICS.vcpu.failures.inc();
                    error!("Received KVM_EXIT_INTERNAL_ERROR signal");
                    Err(Error::VcpuUnhandledKvmExit)
                }
                r => {
                    METRICS.vcpu.failures.inc();
                    // TODO: Are we sure we want to finish running a vcpu upon
                    // receiving a vm exit that is not necessarily an error?
                    error!("Unexpected exit reason on vcpu run: {:?}", r);
                    Err(Error::VcpuUnhandledKvmExit)
                }
            },
            // The unwrap on raw_os_error can only fail if we have a logic
            // error in our code in which case it is better to panic.
            Err(ref e) => {
                match e.raw_os_error().unwrap() {
                    libc::EAGAIN => Ok(false),
                    libc::EINTR => {
                        self.fd.set_kvm_immediate_exit(0);
                        // Notify that this KVM_RUN was interrupted.
                        Ok(true)
                    }
                    _ => {
                        METRICS.vcpu.failures.inc();
                        error!("Failure during vcpu run: {}", e);
                        Err(Error::VcpuUnhandledKvmExit)
                    }
                }
            }
        }
    }

    /// Registers a signal handler which makes use of TLS and kvm immediate exit to
    /// kick the vcpu running on the current thread, if there is one.
    pub fn register_vcpu_kick_signal_handler() {
        unsafe {
            extern "C" fn handle_signal(_: c_int, _: *mut siginfo_t, _: *mut c_void) {
                let _ = Vcpu::run_on_thread_local_vcpu(|vcpu| {
                    vcpu.fd.set_kvm_immediate_exit(1);
                    fence(Ordering::Release);
                });
            }
            // This uses an async signal safe handler to kill the vcpu handles.
            register_vcpu_signal_handler(VCPU_RTSIG_OFFSET, handle_signal)
                .expect("Failed to register vcpu signal handler");
        }
    }

    /// Main loop of the vCPU thread.
    ///
    /// Runs the vCPU in KVM context in a loop. Handles KVM_EXITs then goes back in.
    /// Note that the state of the VCPU and associated VM must be setup first for this to do
    /// anything useful.
    pub fn run(&mut self, seccomp_level: u32) {
        self.init_thread_local_data();

        // Load seccomp filters for this vCPU thread.
        // Execution panics if filters cannot be loaded, use --seccomp-level=0 if skipping filters
        // altogether is the desired behaviour.
        if let Err(e) = default_syscalls::set_seccomp_level(seccomp_level) {
            panic!(
                "Failed to set the requested seccomp filters on vCPU {}: Error: {}",
                self.id, e
            );
        }

        // Start off in the `Paused` state.
        let mut sf = StateStruct::new(Self::paused, false);

        // While current state is not a final/end state, keep churning.
        while !sf.end_state {
            // Run the current state handler, and get the next one.
            sf = sf(self);
        }
    }

    // This is the main loop of the `Running` state.
    fn running(&mut self) -> StateStruct<Self> {
        // This loop is here just for optimizing the emulation path.
        // No point in ticking the state machine if there are no external events.
        loop {
            match self.run_emulation() {
                // Emulation ran successfully, continue.
                Ok(false) => (),
                // Emulation was interrupted, check external events.
                Ok(true) => break,
                // Emulation errors lead to vCPU exit.
                Err(_) => return StateStruct::next(Self::exited),
            }
        }

        // Break this emulation loop on any transition request/external event.
        match self.event_receiver.try_recv() {
            // Running ---- Pause ----> Paused
            Ok(VcpuEvent::Pause) => {
                // Nothing special to do.
                self.response_sender
                    .send(VcpuResponse::Paused)
                    .expect("failed to send pause status");
                // Move to 'paused' state.
                StateStruct::next(Self::paused)
            }
            // Unhandled exit of the other end.
            Err(TryRecvError::Disconnected) => {
                // Move to 'exited' state.
                StateStruct::next(Self::exited)
            }
            // All other events or lack thereof have no effect on current 'running' state.
            Ok(_) | Err(TryRecvError::Empty) => StateStruct::next(Self::running),
        }
    }

    // This is the main loop of the `Paused` state.
    fn paused(&mut self) -> StateStruct<Self> {
        match self.event_receiver.recv() {
            // Paused ---- Resume ----> Running
            Ok(VcpuEvent::Resume) => {
                // Nothing special to do.
                self.response_sender
                    .send(VcpuResponse::Resumed)
                    .expect("failed to send resume status");
                // Move to 'running' state.
                StateStruct::next(Self::running)
            }
            // All other events have no effect on current 'paused' state.
            Ok(_) => StateStruct::next(Self::paused),
            // Unhandled exit of the other end.
            Err(_) => {
                // Move to 'exited' state.
                StateStruct::next(Self::exited)
            }
        }
    }

    // This is the main loop of the `Exited` state.
    fn exited(&mut self) -> StateStruct<Self> {
        if let Err(e) = self.exit_evt.write(1) {
            METRICS.vcpu.failures.inc();
            error!("Failed signaling vcpu exit event: {}", e);
        }
        // State machine reached its end.
        StateStruct::finish(Self::exited)
    }

    fn save_state(&self) -> Result<VcpuState> {
        // Build the list of MSRs we want to save.
        let num_msrs = self.msr_list.as_original_struct().len();
        let mut msrs = KvmMsrs::new(num_msrs);
        {
            let indices = self.msr_list.as_entries_slice();
            let msr_entries = msrs.as_mut_entries_slice();
            assert_eq!(indices.len(), msr_entries.len());
            for (pos, index) in indices.iter().enumerate() {
                msr_entries[pos].index = *index;
            }
        }

        let mut state = VcpuState {
            cpuid: self.cpuid.clone(),
            msrs,
            debug_regs: Default::default(),
            lapic: Default::default(),
            mp_state: Default::default(),
            regs: Default::default(),
            sregs: Default::default(),
            vcpu_events: Default::default(),
            xcrs: Default::default(),
            xsave: Default::default(),
        };

        self.fd
            .get_mp_state(&mut state.mp_state)
            .map_err(Error::VcpuGetMpState)?;

        state.regs = self.fd.get_regs().map_err(Error::VcpuGetRegs)?;

        state.sregs = self.fd.get_sregs().map_err(Error::VcpuGetSregs)?;

        self.fd
            .get_xsave(&mut state.xsave)
            .map_err(Error::VcpuGetXsave)?;

        self.fd
            .get_xcrs(&mut state.xcrs)
            .map_err(Error::VcpuGetXcrs)?;

        self.fd
            .get_debug_regs(&mut state.debug_regs)
            .map_err(Error::VcpuGetDebugRegs)?;

        state.lapic = self.fd.get_lapic().map_err(Error::VcpuGetLapic)?;

        let nmsrs = self
            .fd
            .get_msrs(&mut state.msrs)
            .map_err(Error::VcpuGetMsrs)?;
        assert_eq!(nmsrs, num_msrs);

        self.fd
            .get_vcpu_events(&mut state.vcpu_events)
            .map_err(Error::VcpuGetVcpuEvents)?;

        Ok(state)
    }

    fn restore_state(&self, state: VcpuState) -> Result<()> {
        self.fd
            .set_cpuid2(&state.cpuid)
            .map_err(Error::VcpuSetCpuid)?;
        self.fd
            .set_mp_state(state.mp_state)
            .map_err(Error::VcpuSetMpState)?;
        self.fd.set_regs(&state.regs).map_err(Error::VcpuSetRegs)?;
        self.fd
            .set_sregs(&state.sregs)
            .map_err(Error::VcpuSetSregs)?;
        self.fd
            .set_xsave(&state.xsave)
            .map_err(Error::VcpuSetXsave)?;
        self.fd.set_xcrs(&state.xcrs).map_err(Error::VcpuSetXcrs)?;
        self.fd
            .set_debug_regs(&state.debug_regs)
            .map_err(Error::VcpuSetDebugRegs)?;
        self.fd
            .set_lapic(&state.lapic)
            .map_err(Error::VcpuSetLapic)?;
        self.fd.set_msrs(&state.msrs).map_err(Error::VcpuSetMsrs)?;
        self.fd
            .set_vcpu_events(&state.vcpu_events)
            .map_err(Error::VcpuSetVcpuEvents)?;
        Ok(())
    }
}

impl Drop for Vcpu {
    fn drop(&mut self) {
        let _ = self.reset_thread_local_data();
    }
}

/// Structure holding VCPU kvm state.
pub struct VcpuState {
    cpuid: CpuId,
    msrs: KvmMsrs,
    debug_regs: kvm_debugregs,
    lapic: kvm_lapic_state,
    mp_state: kvm_mp_state,
    regs: kvm_regs,
    sregs: kvm_sregs,
    vcpu_events: kvm_vcpu_events,
    xcrs: kvm_xcrs,
    xsave: kvm_xsave,
}

#[cfg(test)]
mod tests {
    use std::os::unix::io::AsRawFd;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use super::super::devices;
    use super::*;

    use libc::EBADF;

    impl VcpuHandle {
        // Closes our side of the channels with vcpu.
        fn close_vcpu_channel(&mut self) {
            let (s1, _) = channel();
            let (_, r2) = channel();
            self.event_sender = s1;
            self.response_receiver = r2;
        }
    }

    // Auxiliary function being used throughout the tests.
    fn setup_vcpu() -> (Vm, Vcpu) {
        let kvm = KvmContext::new().unwrap();
        let gm = GuestMemory::new_anon_from_tuples(&[(GuestAddress(0), 0x10000)]).unwrap();
        let mut vm = Vm::new(kvm.fd()).expect("Cannot create new vm");
        assert!(vm.memory_init(gm, &kvm).is_ok());

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
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
        #[cfg(target_arch = "aarch64")]
        vm.setup_irqchip(1).expect("Cannot setup irqchip");

        (vm, vcpu)
    }

    // Auxiliary function being used throughout the tests.
    fn setup_vcpu_handler() -> (Vm, VcpuHandle, EventFd) {
        let kvm = KvmContext::new().unwrap();
        let gm = GuestMemory::new_anon_from_tuples(&[(GuestAddress(0), 0x10000)]).unwrap();
        let mut vm = Vm::new(kvm.fd()).expect("Cannot create new vm");
        assert!(vm.memory_init(gm, &kvm).is_ok());

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        vm.setup_irqchip().unwrap();

        let exit_evt = EventFd::new().unwrap();
        let vcpu = VcpuHandle::new(
            1,
            &vm,
            devices::Bus::new(),
            exit_evt.try_clone().unwrap(),
            super::super::TimestampUs::default(),
        )
        .expect("Cannot create Vcpu");
        #[cfg(target_arch = "aarch64")]
        vm.setup_irqchip(1).expect("Cannot setup irqchip");

        (vm, vcpu, exit_evt)
    }

    #[test]
    fn test_set_mmio_bus() {
        let (_, mut vcpu) = setup_vcpu();
        assert!(vcpu.mmio_bus.is_none());
        vcpu.set_mmio_bus(devices::Bus::new());
        assert!(vcpu.mmio_bus.is_some());
    }

    #[test]
    fn test_create_vm() {
        let kvm = KvmContext::new().unwrap();
        let vm = Vm::new(kvm.fd()).expect("Cannot create new vm");

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let cpuid = kvm
                .kvm
                .get_supported_cpuid(MAX_KVM_CPUID_ENTRIES)
                .expect("Cannot get supported cpuid");
            assert_eq!(
                vm.supported_cpuid().as_entries_slice(),
                cpuid.as_entries_slice()
            );
        }

        use super::Error::VmFd;
        let faulty_kvm = unsafe { Kvm::new_with_fd_number(-1) };
        match Vm::new(&faulty_kvm) {
            Err(VmFd(_)) => (),
            Err(e) => panic!(
                "{:?} != {:?}.",
                e,
                VmFd(io::Error::from_raw_os_error(EBADF))
            ),
            Ok(_) => panic!(
                "Expected error {:?}, got Ok()",
                VmFd(io::Error::from_raw_os_error(EBADF))
            ),
        }
    }

    #[test]
    fn test_vm_memory_init_success() {
        let kvm = KvmContext::new().unwrap();
        let gm = GuestMemory::new_anon_from_tuples(&[(GuestAddress(0), 0x1000)]).unwrap();
        let mut vm = Vm::new(kvm.fd()).expect("Cannot create new vm");
        assert!(vm.memory_init(gm, &kvm).is_ok());
        let obj_addr = GuestAddress(0xf0);
        vm.get_memory()
            .unwrap()
            .write_obj_at_addr(67u8, obj_addr)
            .unwrap();
        let read_val: u8 = vm
            .get_memory()
            .unwrap()
            .read_obj_from_addr(obj_addr)
            .unwrap();
        assert_eq!(read_val, 67u8);
    }

    #[test]
    fn test_vm_memory_init_failure() {
        let kvm_fd = Kvm::new().unwrap();
        let mut vm = Vm::new(&kvm_fd).expect("new vm failed");

        let kvm = KvmContext {
            kvm: kvm_fd,
            max_memslots: 1,
        };
        let start_addr1 = GuestAddress(0x0);
        let start_addr2 = GuestAddress(0x1000);
        let gm = GuestMemory::new_anon_from_tuples(&[(start_addr1, 0x1000), (start_addr2, 0x1000)])
            .unwrap();

        assert!(vm.memory_init(gm, &kvm).is_err());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_setup_irqchip() {
        let kvm = KvmContext::new().unwrap();
        let vm = Vm::new(kvm.fd()).expect("Cannot create new vm");

        vm.setup_irqchip().expect("Cannot setup irqchip");
        let (_s1, r1) = channel();
        let (s2, _r2) = channel();
        let exit_evt = EventFd::new().unwrap();
        let _vcpu = Vcpu::new(
            1,
            &vm,
            devices::Bus::new(),
            exit_evt.try_clone().expect("eventfd clone failed"),
            r1,
            s2,
            super::super::TimestampUs::default(),
        )
        .unwrap();
        // Trying to setup two irqchips will result in EEXIST error.
        assert!(vm.setup_irqchip().is_err());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_setup_irqchip() {
        let kvm = KvmContext::new().unwrap();

        let mut vm = Vm::new(kvm.fd()).expect("Cannot create new vm");
        let vcpu_count = 1;
        let (_s1, r1) = channel();
        let (s2, _r2) = channel();
        let _vcpu = Vcpu::new(
            1,
            &vm,
            devices::Bus::new(),
            EventFd::new().unwrap(),
            r1,
            s2,
            super::super::TimestampUs::default(),
        )
        .unwrap();

        vm.setup_irqchip(vcpu_count).expect("Cannot setup irqchip");
        // Trying to setup two irqchips will result in EEXIST error.
        assert!(vm.setup_irqchip(vcpu_count).is_err());
    }

    #[test]
    fn test_setup_irqchip_failure() {
        let kvm = KvmContext::new().unwrap();
        // On aarch64, this needs to be mutable.
        #[allow(unused_mut)]
        let mut vm = Vm::new(kvm.fd()).expect("Cannot create new vm");
        let (_s1, r1) = channel();
        let (s2, _r2) = channel();
        let exit_evt = EventFd::new().unwrap();
        let _vcpu = Vcpu::new(
            1,
            &vm,
            devices::Bus::new(),
            exit_evt.try_clone().expect("eventfd clone failed"),
            r1,
            s2,
            super::super::TimestampUs::default(),
        )
        .unwrap();

        #[cfg(target_arch = "x86_64")]
        // Trying to setup irqchip after KVM_VCPU_CREATE was called will result in error on x86_64.
        assert!(vm.setup_irqchip().is_err());
        #[cfg(target_arch = "aarch64")]
        // Trying to setup irqchip after KVM_VCPU_CREATE is actually the way to go on aarch64.
        assert!(vm.setup_irqchip(1).is_ok());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_configure_vcpu() {
        let (mut vm, mut vcpu) = setup_vcpu();

        let vm_config = VmConfig::default();
        assert!(vcpu.configure(&vm_config, GuestAddress(0), &vm).is_ok());

        // Test configure while using the T2 template.
        let mut vm_config = VmConfig::default();
        vm_config.cpu_template = Some(CpuFeaturesTemplate::T2);
        assert!(vcpu.configure(&vm_config, GuestAddress(0), &vm).is_ok());

        // Test configure while using the C3 template.
        let mut vm_config = VmConfig::default();
        vm_config.cpu_template = Some(CpuFeaturesTemplate::C3);
        assert!(vcpu.configure(&vm_config, GuestAddress(0), &vm).is_ok());

        // Test errors.
        use super::Error::{GuestMemory, HTNotInitialized, VcpuCountNotInitialized};

        let mut vm_config = VmConfig::default();
        vm_config.vcpu_count = None;
        // Expect Error::VcpuCountNotInitialized.
        match vcpu.configure(&vm_config, GuestAddress(0), &vm) {
            Err(VcpuCountNotInitialized) => (),
            Err(e) => panic!("{:?} != {:?}.", e, VcpuCountNotInitialized),
            Ok(()) => panic!("Expected error {:?}, got Ok()", VcpuCountNotInitialized),
        };

        let mut vm_config = VmConfig::default();
        vm_config.ht_enabled = None;
        // Expect Error::HTNotInitialized.
        match vcpu.configure(&vm_config, GuestAddress(0), &vm) {
            Err(HTNotInitialized) => (),
            Err(e) => panic!("{:?} != {:?}.", e, HTNotInitialized),
            Ok(()) => panic!("Expected error {:?}, got Ok()", HTNotInitialized),
        };

        use super::super::memory_model::GuestMemoryError;
        let vm_config = VmConfig::default();
        vm.guest_mem = None;
        // Expect Error::GuestMemory.
        match vcpu.configure(&vm_config, GuestAddress(0), &vm) {
            Err(GuestMemory(_)) => (),
            Err(e) => panic!(
                "{:?} != {:?}.",
                e,
                GuestMemory(GuestMemoryError::MemoryNotInitialized)
            ),
            Ok(()) => panic!(
                "Expected error {:?}, got Ok()",
                GuestMemory(GuestMemoryError::MemoryNotInitialized)
            ),
        };
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_configure_vcpu() {
        let kvm = KvmContext::new().unwrap();
        let gm = GuestMemory::new_anon_from_tuples(&[(GuestAddress(0), 0x10000)]).unwrap();
        let mut vm = Vm::new(kvm.fd()).expect("new vm failed");
        assert!(vm.memory_init(gm, &kvm).is_ok());

        // Try it for when vcpu id is 0.
        let (_s01, r1) = channel();
        let (s2, _r02) = channel();
        let exit_evt = EventFd::new().unwrap();
        let mut vcpu = Vcpu::new(
            0,
            &vm,
            devices::Bus::new(),
            exit_evt.try_clone().expect("eventfd clone failed"),
            r1,
            s2,
            super::super::TimestampUs::default(),
        )
        .unwrap();

        let vm_config = VmConfig::default();
        assert!(vcpu.configure(&vm_config, GuestAddress(0), &vm).is_ok());

        // Try it for when vcpu id is NOT 0.
        let (_s11, r1) = channel();
        let (s2, _r12) = channel();
        let mut vcpu = Vcpu::new(
            1,
            &vm,
            devices::Bus::new(),
            exit_evt.try_clone().expect("eventfd clone failed"),
            r1,
            s2,
            super::super::TimestampUs::default(),
        )
        .unwrap();

        assert!(vcpu.configure(&vm_config, GuestAddress(0), &vm).is_ok());
    }

    #[test]
    fn vcpu_tls() {
        let (_, mut vcpu) = setup_vcpu();

        // Running on the TLS vcpu should fail before we actually initialize it.
        assert!(Vcpu::run_on_thread_local_vcpu(|_| ()).is_err());

        // Initialize vcpu TLS.
        vcpu.init_thread_local_data();

        // Validate TLS vcpu is the local vcpu by changing the `id` through the TLS
        // interface, then validating it locally.
        assert_eq!(vcpu.id, 1);
        assert!(Vcpu::run_on_thread_local_vcpu(|v| v.id = 2).is_ok());
        assert_eq!(vcpu.id, 2);

        // Reset vcpu TLS.
        assert!(vcpu.reset_thread_local_data().is_ok());

        // Running on the TLS vcpu after TLS reset should fail.
        assert!(Vcpu::run_on_thread_local_vcpu(|_| ()).is_err());

        // Second reset should return EINVAL.
        assert!(vcpu.reset_thread_local_data().is_err());
    }

    #[test]
    #[should_panic]
    fn invalid_tls() {
        let (_, mut vcpu) = setup_vcpu();
        // Initialize vcpu TLS.
        vcpu.init_thread_local_data();
        // Trying to initialize non-empty TLS should panic.
        vcpu.init_thread_local_data();
    }

    #[test]
    #[should_panic]
    fn invalid_seccomp_lvl() {
        let (_, mut vcpu) = setup_vcpu();
        // Setting an invalid seccomp level should panic.
        vcpu.run(seccomp::SECCOMP_LEVEL_ADVANCED + 10);
    }

    // Sends an event to a vcpu and expects a particular response.
    fn queue_event_expect_response(handle: &VcpuHandle, event: VcpuEvent, response: VcpuResponse) {
        handle
            .send_event(event)
            .expect("failed to send event to vcpu");
        assert_eq!(
            handle
                .response_receiver()
                .recv_timeout(Duration::from_millis(100))
                .expect("did not receive event response from vcpu"),
            response
        );
    }

    // Sends an event to a vcpu and expects no response.
    fn queue_event_expect_timeout(handle: &VcpuHandle, event: VcpuEvent) {
        handle
            .send_event(event)
            .expect("failed to send event to vcpu");
        assert_eq!(
            handle
                .response_receiver()
                .recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vcpu_states() {
        Vcpu::register_vcpu_kick_signal_handler();

        let (vm, mut vcpu_handle, vcpu_exit_evt) = setup_vcpu_handler();

        let vm_config = VmConfig::default();
        assert!(vcpu_handle
            .configure_vcpu(&vm_config, GuestAddress(0), &vm)
            .is_ok());

        let seccomp_level = 0;
        vcpu_handle
            .start_vcpu(seccomp_level, Some(devices::Bus::new()))
            .expect("failed to start vcpu");

        // Queue a Resume event, expect a response.
        queue_event_expect_response(&vcpu_handle, VcpuEvent::Resume, VcpuResponse::Resumed);

        // Queue a Pause event, expect a response.
        queue_event_expect_response(&vcpu_handle, VcpuEvent::Pause, VcpuResponse::Paused);
        // Validate vcpu handled the EINTR gracefully and didn't exit.
        let err = vcpu_exit_evt.read().unwrap_err();
        assert_eq!(err.raw_os_error().unwrap(), libc::EAGAIN);

        // Queue another Pause event, expect no answer.
        queue_event_expect_timeout(&vcpu_handle, VcpuEvent::Pause);

        // Queue a Resume event, expect a response.
        queue_event_expect_response(&vcpu_handle, VcpuEvent::Resume, VcpuResponse::Resumed);

        // Queue another Resume event, expect no answer.
        queue_event_expect_timeout(&vcpu_handle, VcpuEvent::Resume);

        // Queue another Pause event, expect a response.
        queue_event_expect_response(&vcpu_handle, VcpuEvent::Pause, VcpuResponse::Paused);

        // Close the Sender end of the channel so that the vcpu waiting on it will error.
        vcpu_handle.close_vcpu_channel();
        // Wait a bit to make sure the channel closes and vcpu errors.
        thread::sleep(Duration::from_millis(100));

        // Validate that the vCPU exited because of the error.
        assert_eq!(vcpu_exit_evt.read().unwrap(), 1);
        // Validate vCPU thread ends execution.
        vcpu_handle
            .join_vcpu_thread()
            .expect("failed to join thread");
    }

    #[test]
    fn not_enough_mem_slots() {
        let kvm_fd = Kvm::new().unwrap();
        let mut vm = Vm::new(&kvm_fd).expect("new vm failed");

        let kvm = KvmContext {
            kvm: kvm_fd,
            max_memslots: 1,
        };
        let start_addr1 = GuestAddress(0x0);
        let start_addr2 = GuestAddress(0x1000);
        let gm = GuestMemory::new_anon_from_tuples(&[(start_addr1, 0x1000), (start_addr2, 0x1000)])
            .unwrap();

        assert!(vm.memory_init(gm, &kvm).is_err());
    }

    #[test]
    fn test_vm_save_restore_state() {
        let kvm_fd = Kvm::new().unwrap();
        let vm = Vm::new(&kvm_fd).expect("new vm failed");
        assert!(vm.save_state().is_err());

        let (vm, _) = setup_vcpu();
        let res = vm.save_state();
        assert!(res.is_ok());

        let state = res.unwrap();
        let (vm, _) = setup_vcpu();
        assert!(vm.restore_state(&state).is_ok());
    }

    #[test]
    fn test_vcpu_save_restore_state() {
        let (_vm, vcpu) = setup_vcpu();
        let state = vcpu.save_state();
        assert!(state.is_ok());
        assert!(vcpu.restore_state(state.unwrap()).is_ok());

        unsafe { libc::close(vcpu.fd.as_raw_fd()) };
        let state = VcpuState {
            cpuid: CpuId::new(1),
            msrs: KvmMsrs::new(1),
            debug_regs: Default::default(),
            lapic: Default::default(),
            mp_state: Default::default(),
            regs: Default::default(),
            sregs: Default::default(),
            vcpu_events: Default::default(),
            xcrs: Default::default(),
            xsave: Default::default(),
        };
        assert!(vcpu.restore_state(state).is_err());
    }
}
