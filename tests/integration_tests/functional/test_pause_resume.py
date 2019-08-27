# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that verify Pause/Resume functionality."""

import os
import platform
import time

import pytest

import host_tools.drive as drive_tools
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


@pytest.mark.skipif(
    platform.machine() != "x86_64",
    reason="not yet implemented on aarch64"
)
def test_snapshot(test_microvm_with_ssh, network_config):
    """Test snapshotting functionality."""
    test_microvm = test_microvm_with_ssh

    # Cloning and network namespaces don't work well together yet.
    test_microvm.jailer.netns = None
    test_microvm.ssh_config['netns_file_path'] = None

    test_microvm.spawn()

    memfile = test_microvm.tmp_path()
    snapshot_filename = test_microvm.tmp_path()

    # Set up the microVM with 1 vCPUs, 256 MiB of file-backed RAM, no network
    # ifaces and a root file system with the rw permission. The network
    # interface is added after we get a unique MAC and IP.
    test_microvm.basic_config(memfile=memfile)

    _tap, _, _ = test_microvm.ssh_network_config(network_config, '1')

    # Add a scratch block device.
    fs = drive_tools.FilesystemFile(
        os.path.join(test_microvm.fsfiles, 'scratch')
    )
    response = test_microvm.drive.put(
        drive_id='scratch',
        path_on_host=test_microvm.create_jailed_resource(fs.path),
        is_root_device=False,
        is_read_only=False
    )
    assert test_microvm.api_session.is_status_no_content(response.status_code)

    # Snapshotting the microVM before being started should not work.
    response = test_microvm.actions.put(action_type='PauseToSnapshot',
                                        payload=snapshot_filename)
    assert test_microvm.api_session.is_status_bad_request(response.status_code)
    assert "Microvm is not running." in response.text

    # Start microVM.
    test_microvm.start()
    time.sleep(0.3)

    # The APIs for snapshot related procedures does not get timed (yet?).
    test_microvm.api_session.untime()

    # Verify that the network device works by running something over ssh.
    ssh_connection = net_tools.SSHConnection(test_microvm.ssh_config)

    # Verify that the scratch block device works by mounting it.
    magic = 42
    _, _, stderr = ssh_connection.execute_command(
        'mkdir -p {0} && mount /dev/vdb {0} && echo "{1}" > {0}/world'
        .format('/mnt/hello', magic)
    )

    assert stderr.read().decode('utf-8') == ''

    # Pause to snapshot.
    response = test_microvm.actions.put(action_type='PauseToSnapshot',
                                        payload=snapshot_filename)
    assert test_microvm.api_session.is_status_no_content(response.status_code)

    # Spawn clone.
    test_microvm.jailer.cleanup(reuse_jail=True)
    test_microvm.spawn()

    response = test_microvm.actions.put(action_type='ResumeFromSnapshot',
                                        payload=snapshot_filename)
    assert test_microvm.api_session.is_status_no_content(response.status_code)

    # Reconnect ssh.
    ssh_connection = net_tools.SSHConnection(test_microvm.ssh_config)

    # Verify that the clone sees the footprint file.
    _, stdout, stderr = ssh_connection.execute_command('cat /mnt/hello/world')

    assert stderr.read().decode('utf-8') == ''
    assert int(stdout.readline().decode('utf-8').strip()) == magic
