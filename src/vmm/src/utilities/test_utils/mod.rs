// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#![allow(missing_docs)]

use crate::Vmm;
use std::panic;
use std::sync::{Arc, Mutex};

use crate::builder::build_microvm_for_boot;
use crate::resources::VmResources;
use crate::seccomp_filters::{get_filters, SeccompConfig};
use crate::utilities::mock_resources::{MockBootSourceConfig, MockVmConfig, MockVmResources};
use crate::vmm_config::boot_source::BootSourceConfig;
use crate::vmm_config::instance_info::InstanceInfo;
use polly::event_manager::{EventManager, ExitCode};

pub fn create_vmm(_kernel_image: Option<&str>, is_diff: bool) -> (Arc<Mutex<Vmm>>, EventManager) {
    let mut event_manager = EventManager::new().unwrap();
    let empty_seccomp_filters = get_filters(SeccompConfig::None).unwrap();

    let boot_source_cfg = MockBootSourceConfig::new().with_default_boot_args();
    #[cfg(target_arch = "aarch64")]
    let boot_source_cfg: BootSourceConfig = boot_source_cfg.into();
    #[cfg(target_arch = "x86_64")]
    let boot_source_cfg: BootSourceConfig = match _kernel_image {
        Some(kernel) => boot_source_cfg.with_kernel(kernel).into(),
        None => boot_source_cfg.into(),
    };
    let mock_vm_res = MockVmResources::new().with_boot_source(boot_source_cfg);
    let resources: VmResources = if is_diff {
        mock_vm_res
            .with_vm_config(MockVmConfig::new().with_dirty_page_tracking().into())
            .into()
    } else {
        mock_vm_res.into()
    };

    (
        build_microvm_for_boot(
            &InstanceInfo::default(),
            &resources,
            &mut event_manager,
            &empty_seccomp_filters,
        )
        .unwrap(),
        event_manager,
    )
}

pub fn default_vmm(kernel_image: Option<&str>) -> (Arc<Mutex<Vmm>>, EventManager) {
    create_vmm(kernel_image, false)
}

#[cfg(target_arch = "x86_64")]
pub fn dirty_tracking_vmm(kernel_image: Option<&str>) -> (Arc<Mutex<Vmm>>, EventManager) {
    create_vmm(kernel_image, true)
}

pub fn run_events_to_completion(mut event_manager: EventManager) -> ExitCode {
    // On x86_64, the vmm should exit once its workload completes and signals the exit event.
    // On aarch64, the test kernel doesn't exit, so the vmm is force-stopped.
    // #[cfg(target_arch = "x86_64")]
    loop {
        match event_manager.run_core(500) {
            // Should not get errors.
            Err(e) => panic!("error {:?}", e),
            // Still stuff to process, carry on.
            Ok((None, _)) => (),
            // Break event loop condition, return exit-code.
            Ok((Some(exit_code), _)) => return exit_code,
        }
    }
}
