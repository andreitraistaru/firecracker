# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests scenarios for microVM snapshotting functionality."""

import platform
import os

import pytest

import host_tools.drive as drive_tools
import host_tools.network as net_tools


@pytest.mark.skipif(
    platform.machine() != "x86_64",
    reason="only supported on x86_64"
)
def test_snapshot(test_microvm_with_ssh, network_config):
    """Test snapshotting functionality."""
    test_microvm = test_microvm_with_ssh

    # Cloning and network namespaces don't work well together yet.
    test_microvm.disable_netns()

    test_microvm.spawn()

    # Set up the microVM with 1 vCPU, 256 MiB of file-backed RAM, no network
    # ifaces and a root file system with the rw permission. The network
    # interface is added after we get a unique MAC and IP.
    test_microvm.basic_config()

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
    resp = test_microvm.snapshot_create.put(
        mem_file_path=test_microvm.tmp_path(),
        snapshot_path=test_microvm.tmp_path()
    )
    assert test_microvm.api_session.is_status_bad_request(resp.status_code)
    assert "requested operation is not supported before starting the microVM"\
           in resp.text

    # Start microVM.
    test_microvm.start()

    # The APIs for snapshot related procedures do not get timed.
    test_microvm.api_session.untime()

    # Verify that the network device works by running something over ssh.
    ssh_connection = net_tools.SSHConnection(test_microvm.ssh_config)

    # Verify that the scratch block device works by mounting it.
    magic = 42
    _, _, stderr = ssh_connection.execute_command(
        'mkdir -p {0} && mount /dev/vdb {0} && echo "{1}" > {0}/world'
        .format('/mnt/hello', magic)
    )
    assert stderr.read() == ''

    # Snapshotting the microVM without pausing should not work.
    resp = test_microvm.snapshot_create.put(
        mem_file_path=test_microvm.tmp_path(),
        snapshot_path=test_microvm.tmp_path()
    )
    assert test_microvm.api_session.is_status_bad_request(resp.status_code)
    assert "Cannot save microvm state: Vcpu is in unexpected state" \
           in resp.text

    # Pause to snapshot.
    mem_file_path, snapshot_path = test_microvm.pause_to_snapshot()

    # Kill original firecracker microvm.
    test_microvm.kill()

    # Spawn clone.
    test_microvm.resume_from_snapshot(mem_file_path, snapshot_path)

    # Reconnect ssh.
    ssh_connection = net_tools.SSHConnection(test_microvm.ssh_config)

    # Verify that the clone sees the footprint file.
    _, stdout, stderr = ssh_connection.execute_command('cat /mnt/hello/world')

    assert stderr.read() == ''
    assert int(stdout.readline().strip()) == magic
