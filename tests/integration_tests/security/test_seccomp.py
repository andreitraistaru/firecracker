# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that the seccomp filters don't let blacklisted syscalls through."""

import os

from subprocess import run


def test_seccomp_ls(aux_bin_paths):
    """Assert that the seccomp filters deny a blacklisted syscall."""
    # pylint: disable=redefined-outer-name
    # The fixture pattern causes a pylint false positive for that rule.

    # Path to the `ls` binary, which attempts to execute the blacklisted
    # `SYS_access`.
    ls_command_path = '/bin/ls'
    demo_jailer = aux_bin_paths['demo_basic_jailer']

    assert os.path.exists(demo_jailer)

    # Compile the mini jailer.
    outcome = run([demo_jailer, ls_command_path])

    # The seccomp filters should send SIGSYS (31) to the binary. `ls` doesn't
    # handle it, so it will exit with error.
    assert outcome.returncode != 0


def test_advanced_seccomp_harmless(aux_bin_paths):
    """
    Test `demo_harmless`.

    Test that the advanced demo jailer allows the harmless demo binary.
    """
    # pylint: disable=redefined-outer-name
    # The fixture pattern causes a pylint false positive for that rule.

    demo_advanced_jailer = aux_bin_paths['demo_advanced_jailer']
    demo_harmless = aux_bin_paths['demo_harmless']

    assert os.path.exists(demo_advanced_jailer)
    assert os.path.exists(demo_harmless)

    outcome = run([demo_advanced_jailer, demo_harmless])

    # The demo harmless binary should have terminated gracefully.
    assert outcome.returncode == 0


def test_advanced_seccomp_malicious(aux_bin_paths):
    """
    Test `demo_malicious`.

    Test that the basic demo jailer denies the malicious demo binary.
    """
    # pylint: disable=redefined-outer-name
    # The fixture pattern causes a pylint false positive for that rule.

    demo_advanced_jailer = aux_bin_paths['demo_advanced_jailer']
    demo_malicious = aux_bin_paths['demo_malicious']

    assert os.path.exists(demo_advanced_jailer)
    assert os.path.exists(demo_malicious)

    outcome = run([demo_advanced_jailer, demo_malicious])

    # The demo malicious binary should have received `SIGSYS`.
    assert outcome.returncode == -31
