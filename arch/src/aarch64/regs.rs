// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

use std::{io, mem, result};

use kvm::VcpuFd;

use super::get_fdt_addr;
use kvm_bindings::{user_pt_regs, KVM_REG_ARM64, KVM_REG_ARM_CORE, KVM_REG_SIZE_U64};
use memory_model::GuestMemory;

#[derive(Debug)]
pub enum Error {
    SetCoreRegister(io::Error),
}

pub type Result<T> = result::Result<T, Error>;

#[allow(non_upper_case_globals)]
// PSR (Processor State Register) bits.
// Taken from arch/arm64/include/uapi/asm/ptrace.h.
const PSR_MODE_EL1h: u64 = 0x0000_0005;
const PSR_F_BIT: u64 = 0x0000_0040;
const PSR_I_BIT: u64 = 0x0000_0080;
const PSR_A_BIT: u64 = 0x0000_0100;
const PSR_D_BIT: u64 = 0x0000_0200;
// Taken from arch/arm64/kvm/inject_fault.c.
const PSTATE_FAULT_BITS_64: u64 = (PSR_MODE_EL1h | PSR_A_BIT | PSR_F_BIT | PSR_I_BIT | PSR_D_BIT);

// Following are macros that help with getting the ID of a aarch64 core register.
// The core register are represented by the user_pt_regs structure. Look for it in
// arch/arm64/include/uapi/asm/ptrace.h.

// This macro gets the offset of a structure (i.e `str`) member (i.e `field`) without having
// an instance of that structure.
// It uses a null pointer to retrieve the offset to the field.
// Inspired by C solution: `#define offsetof(str, f) ((size_t)(&((str *)0)->f))`.
// Doing `offset__of!(user_pt_regs, pstate)` in our rust code will trigger the following:
// unsafe { &(*(0 as *const user_pt_regs)).pstate as *const _ as usize }
// The dereference expression produces an lvalue, but that lvalue is not actually read from,
// we're just doing pointer math on it, so in theory, it should safe.
macro_rules! offset__of {
    ($str:ty, $field:ident) => {
        unsafe { &(*(0 as *const $str)).$field as *const _ as usize }
    };
}

macro_rules! arm64_core_reg {
    ($reg: tt) => {
        // As per `kvm_arm_copy_reg_indices`, the id of a core register can be obtained like this:
        // `const u64 core_reg = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE | i`, where i is obtained with:
        // `for (i = 0; i < sizeof(struct kvm_regs) / sizeof(__u32); i++) {`
        // We are using here `user_pt_regs` since this structure contains the core register and it is at
        // the start of `kvm_regs`.
        // struct kvm_regs {
        //	struct user_pt_regs regs;	/* sp = sp_el0 */
        //
        //	__u64	sp_el1;
        //	__u64	elr_el1;
        //
        //	__u64	spsr[KVM_NR_SPSR];
        //
        //	struct user_fpsimd_state fp_regs;
        //};
        // struct user_pt_regs {
        //	__u64		regs[31];
        //	__u64		sp;
        //	__u64		pc;
        //	__u64		pstate;
        //};
        // In our implementation we need: pc, pstate and user_pt_regs->regs[0].
        KVM_REG_ARM64 as u64
            | KVM_REG_SIZE_U64 as u64
            | u64::from(KVM_REG_ARM_CORE)
            | ((offset__of!(user_pt_regs, $reg) / mem::size_of::<u32>()) as u64)
    };
}

/// Configure core registers for a given CPU.
///
/// # Arguments
///
/// * `vcpu` - Structure for the VCPU that holds the VCPU's fd.
/// * `cpu_id` - Index of current vcpu.
/// * `boot_ip` - Starting instruction pointer.
/// * `mem` - Reserved DRAM for current VM.
pub fn setup_regs(vcpu: &VcpuFd, cpu_id: u8, boot_ip: usize, mem: &GuestMemory) -> Result<()> {
    // Get the register index of the PSTATE (Processor State) register.
    vcpu.set_one_reg(arm64_core_reg!(pstate), PSTATE_FAULT_BITS_64)
        .map_err(Error::SetCoreRegister)?;

    // Other vCPUs are powered off initially awaiting PSCI wakeup.
    if cpu_id == 0 {
        // Setting the PC (Processor Counter) to the current program address (kernel address).
        vcpu.set_one_reg(arm64_core_reg!(pc), boot_ip as u64)
            .map_err(Error::SetCoreRegister)?;

        // Last mandatory thing to set -> the address pointing to the FDT (also called DTB).
        // "The device tree blob (dtb) must be placed on an 8-byte boundary and must
        // not exceed 2 megabytes in size." -> https://www.kernel.org/doc/Documentation/arm64/booting.txt.
        // We are choosing to place it the end of DRAM. See `get_fdt_addr`.
        vcpu.set_one_reg(arm64_core_reg!(regs), get_fdt_addr(mem) as u64)
            .map_err(Error::SetCoreRegister)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarch64::{arch_memory_regions, layout};
    use kvm::Kvm;

    #[test]
    fn test_setup_regs() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let vcpu = vm.create_vcpu(0).unwrap();
        let regions = arch_memory_regions(layout::FDT_MAX_SIZE + 0x1000);
        let mem = GuestMemory::new_anon_from_tuples(&regions).expect("Cannot initialize memory");

        match setup_regs(&vcpu, 0, 0x0, &mem).unwrap_err() {
            Error::SetCoreRegister(ref e) => assert_eq!(e.raw_os_error(), Some(libc::ENOEXEC)),
        }
        let mut kvi: kvm_bindings::kvm_vcpu_init = kvm_bindings::kvm_vcpu_init::default();
        vm.get_preferred_target(&mut kvi).unwrap();
        vcpu.vcpu_init(&kvi).unwrap();

        assert!(setup_regs(&vcpu, 0, 0x0, &mem).is_ok());
    }
}
