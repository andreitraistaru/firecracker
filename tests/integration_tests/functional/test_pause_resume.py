# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that verify Pause/Resume functionality."""

import host_tools.network as net_tools


def test_pause_resume(test_microvm_with_ssh, network_config):
    """Test a regular microvm API start sequence."""
    test_microvm = test_microvm_with_ssh
    test_microvm.spawn()

    # Set up the microVM with 2 vCPUs, 256 MiB of RAM and
    # a root file system with the rw permission.
    test_microvm.basic_config()

    # Create tap before configuring interface.
    _tap1, _, _ = test_microvm.ssh_network_config(network_config, '1')

    # Pausing the microVM before being started is not allowed.
    response = test_microvm.actions.put(action_type='PauseVCPUs')
    assert test_microvm.api_session.is_status_bad_request(response.status_code)

    # Resuming the microVM before being started is also not allowed.
    response = test_microvm.actions.put(action_type='ResumeVCPUs')
    assert test_microvm.api_session.is_status_bad_request(response.status_code)

    # Start microVM.
    test_microvm.start()

    ssh_connection = net_tools.SSHConnection(test_microvm.ssh_config)

    # Verify guest is active.
    retcode, _, _ = ssh_connection.execute_command("true")
    assert retcode == 0

    # Pausing the microVM after it's been started is successful.
    response = test_microvm.actions.put(action_type='PauseVCPUs')
    assert test_microvm.api_session.is_status_no_content(response.status_code)

    # Verify guest no longer responds to ping.
    retcode, _, _ = ssh_connection.execute_command("true")
    assert retcode != 0

    # Multiple sequential `PauseVCPUs` actions are also allowed.
    response = test_microvm.actions.put(action_type='PauseVCPUs')
    assert test_microvm.api_session.is_status_no_content(response.status_code)
    response = test_microvm.actions.put(action_type='PauseVCPUs')
    assert test_microvm.api_session.is_status_no_content(response.status_code)

    # Resuming the microVM is successful.
    response = test_microvm.actions.put(action_type='ResumeVCPUs')
    assert test_microvm.api_session.is_status_no_content(response.status_code)

    # Verify guest is active again.
    retcode, _, _ = ssh_connection.execute_command("true")
    assert retcode == 0

    # Multiple sequential `ResumeVCPUs` actions are also allowed.
    response = test_microvm.actions.put(action_type='ResumeVCPUs')
    assert test_microvm.api_session.is_status_no_content(response.status_code)
    response = test_microvm.actions.put(action_type='ResumeVCPUs')
    assert test_microvm.api_session.is_status_no_content(response.status_code)
