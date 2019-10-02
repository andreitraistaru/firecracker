# MicroVM Cloning

Using the snapshotting feature a microVM can be effectively cloned,
allowing forking a particular workload into multiple identical clones.

To create multiple microVM clones, one starts from the collection of
resources that comprises a full microVM snapshot and by sharing or
duplicating said resources can resume the original workload on multiple
Firecracker instances thus effectively cloning the original workload.

## Full microVM snapshot composition

A microVM snapshot is actually broken down and stored into individual
smaller blocks:

- vmm image saved at `snapshot_path`
- guest memory file saved at `mem_file_path`
- all files backing emulated guest block devices
- all TAPs associated with emulated guest network interfaces

Some of the above can be shared among clones while others need to
be duplicated.

### Snapshot - vmm image

When creating a snapshot (see [API](api_requests/snapshot.md)), a path
is provided to generate the `vmm image`. This file is the serialized
state of the Firecracker VMM.

This image is required for the [ResumeFromSnapshot](api_requests/snapshot.md)
API request and will be used by the destination Firecracker to load
the serialized VMM state.

This image file is only read by the destination Firecracker and **can
be shared among clones**.

### Snapshot - guest memory file

The file backing the guest memory is privately mapped by the destination
Firecracker and guest memory is page-faulted in the guest as it's being
accessed.

The guest memory file **can also be shared among clones**. Clones guest
memory is Copy-on-Write so the file is **not** modified by any of the
clones.

*All cloned microVMs have shared read-only ownership of the guest memory
file for the **full duration of their lifetime**.*

### Host files backing the guest block devices

Cloned **read-write** block devices need exclusive access to the
host-side files backing them, thus **require duplication**.

Cloned **read-only** block devices **can share access** to the host-side
files backing them.

### Host TAPs backing the guest network devices

Cloned guest network devices need exclusive access to the host TAPs
backing them. Since the clones expect host-side TAPs with the same name,
the recommended solution is to use `network namespaces` and have each
clone live in its own network namespace.

Using `netns` is sufficient to allow correct cloning. In order to allow
cloned workloads to continue running without network reconfiguration,
further steps need to be taken host-side to create the necessary
plumbing to connect the clones to the outside and remap their apparent
IP addresses.

The recommended strategy to do this plumbing is described in-depth in
the [networking for clones document](network-for-clones.md).

## Post-resume operations required for clones

### Reseeding randomness

Clones will, by definition, resume their workloads with some identical
randomness pools. Guest-side actions are required to reseed randomness.
Please see the [random for clones document](random-for-clones.md) for
in-depth description and recommended actions.

### Adjusting guest wall-clock

All resumed microVMs (clones included) will be resumed with the guest OS
wall-clock continuing from the moment of snapshot. The wall-clock needs
to be updated to the current time.

This has to be done on the guest-side using `timedatectl`,
`settimeofday` or using NTP.

If using our [recommended guest kernel config](
https://github.com/firecracker-microvm/firecracker/blob/master/resources/microvm-kernel-config),
`/dev/ptp0` is available in the guest to use as a reliable host-synchronized
time-source.
