# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that verify Pause/Resume functionality."""

import time

from subprocess import run, PIPE

import host_tools.network as net_tools


def test_pause_resume(test_microvm_with_ssh, network_config):
    """Test pausing and resuming of vcpus."""
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

    # Subsequent `PauseVCPUs` actions are no longer allowed.
    response = test_microvm.actions.put(action_type='PauseVCPUs')
    assert test_microvm.api_session.is_status_bad_request(response.status_code)
    response = test_microvm.actions.put(action_type='PauseVCPUs')
    assert test_microvm.api_session.is_status_bad_request(response.status_code)

    # Resuming the microVM is successful.
    response = test_microvm.actions.put(action_type='ResumeVCPUs')
    assert test_microvm.api_session.is_status_no_content(response.status_code)

    # Verify guest is active again.
    retcode, _, _ = ssh_connection.execute_command("true")
    assert retcode == 0

    # Subsequent `ResumeVCPUs` actions are allowed.
    response = test_microvm.actions.put(action_type='ResumeVCPUs')
    assert test_microvm.api_session.is_status_no_content(response.status_code)


def test_snapshot_without_devices(test_microvm_with_api):
    """Test snapshotting functionality."""
    test_microvm = test_microvm_with_api
    test_microvm.spawn()

    # Set up the microVM with 2 vCPUs, 256 MiB of RAM and
    # a root file system with the rw permission.
    test_microvm.basic_config()

    # Snapshotting the microVM before being started should not work.
    response = test_microvm.actions.put(action_type='PauseToSnapshot')
    assert test_microvm.api_session.is_status_bad_request(response.status_code)
    assert "Microvm is not running." in response.text

    # Start microVM.
    snapshot_filename = test_microvm.snapshot_filename()
    test_microvm.start(snapshot_path=snapshot_filename)
    time.sleep(0.3)

    # The APIs for snapshot related procedures does not get timed (yet?).
    test_microvm.api_session.untime()

    response = test_microvm.actions.put(action_type='PauseToSnapshot')
    assert test_microvm.api_session.is_status_no_content(response.status_code)

    test_microvm.jailer.cleanup(force=False)
    test_microvm.spawn()

    response = test_microvm.actions.put(action_type='ResumeFromSnapshot',
                                        payload=snapshot_filename)
    assert test_microvm.api_session.is_status_no_content(response.status_code)

    # We are making sure that the firecracker process has started.
    process = run("ps -u {} -o args | grep \"firecracker --id={}\"".format(
        test_microvm.jailer.uid,
        test_microvm.id
    ), shell=True, check=True, stdout=PIPE)
    assert "firecracker --id={}".format(test_microvm.id) \
           in process.stdout.decode('utf-8')
