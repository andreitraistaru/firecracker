# Snapshot API Requests

MicroVMs can be Paused-to-a-Snapshot, and Resumed-from-a-Snapshot
by using the `PUT` requests to `/snapshot/create` and `/snapshot/load`
respectively.

## Prerequisites

Snapshotting is only currently supported on `x86_64`.

Emulated devices that have host-side resources attached (like block
and network), have dependencies on said resources when resuming from
snapshot.

### Block

Files backing the guest block devices need to be made available to the
resumed microVM when resuming from a snapshot. These files will be
opened by the resumed microVM from the same path they were used by the
snapshotted microVM. Usually this is a jail-relative path so multiple
clones can use the same path with different files.

Read-write block devices need exclusive access to the host-side files
backing them.

Read-only block devices can share access to a single shared host-side
backing file.

### Network

Host-side TAP network interfaces attached to the guest emulated network
interfaces need to be made available to the resumed microVM when
resuming from a snapshot. The TAPs used by the resumed microVMs need
to have the same names as the ones used by the respective snapshotted
microVMs. For more details, please see the networking section of the
[cloning document](cloning.md).

### Memory

A microVM can have its guest memory backed by anonymous host memory or
backed by a host-side file (file-mapped memory). **Only file-backed
guest memory microVMs can be subsequently snapshotted.**

To configure a microVM to use file-backed guest memory, one must specify
the path where such a file will be created.

#### File-backed memory configuration example

```bash
curl --unix-socket /tmp/firecracker.socket -i  \
    -X PUT 'http://localhost/machine-config' \
    -H 'Accept: application/json'            \
    -H 'Content-Type: application/json'      \
    -d '{
        \"vcpu_count\": 2,
        \"mem_size_mib\": 1024,
        \"mem_file_path\": \"/path/to/mem_file\"
    }'
```

## PauseToSnapshot - creating a snapshot of a microVM

Use the `PUT /snapshot/create` API request to create a snapshot.
On receipt of this request, Firecracker pauses the VCPUs, effectively
pausing the microVM guest. It then saves VCPUs states, the VM state and
all the devices states to a file specified through the `snapshot_path`
API request parameter, and syncs guest memory to the file backing the
guest memory.

This is a synchronous request. If the save to snapshot operation
succeeds, Firecracker responds with an `OK` to this request and `150ms`
later finishes its execution. If there is an error encountered,
Firecracker responds with an `Error` to the request and continue running
unimpeded.
The Firecracker process end is delayed by `150ms` after sending the `OK`
response so that the response has time to be pushed through the relevant
levels of the stack and ensure its correct transmission.

**NOTE** if one wishes to **immediately** start a `clone` microVM from
the generated snapshot, they should `kill` this Firecracker process
after receiving the `OK` response to release its host-side resources,
and not wait for the `original` microVM to kill itself (in <150ms).

This API request will only be successful if called on a Firecracker
which had previously started its guest - it had previously received an
`InstanceStart` API action request.

Only file-backed guest memory microVMs can be snapshotted.

### PauseToSnapshot Example

```bash
curl --unix-socket /tmp/firecracker.socket -i \
    -X PUT "http://localhost/snapshot/create" \
    -H  "accept: application/json" \
    -H  "Content-Type: application/json" \
    -d "{
             \"snapshot_path\": \"/path/to/snapshot\"
    }"
```

### Resources comprising the full snapshot

To be able to resume the microVM the following resources will be needed:

- vmm image saved at `snapshot_path`
- guest memory file saved at `mem_file_path`
- all files backing emulated guest block devices
- all TAPs associated with emulated guest network interfaces

## ResumeFromSnapshot - loading a snapshot in a fresh Firecracker

Use the `PUT /snapshot/load` API request to load a snapshot and resume
a snapshotted microVM guest.

Required resources:

- vmm image specified through `snapshot_path`
- guest memory file specified through `mem_file_path`
- all files backing emulated guest block devices made available at the
  same relative paths they were used by the original microVM
- all TAPs associated with emulated guest network interfaces used by the
  original microVM, made available in the network namespace where the
  to-be-resumed Firecracker is running.

If one wishes to reuse an existing host TAP for restoring a microVM,
make sure the *original/source* microVM Firecracker process has ended so
that it released its grasp on the respective host TAP device.

This is a synchronous request. If the operation is successful,
Firecracker restores VCPUs, VM, devices, and guest memory from said
files, resumes execution and responds with an `OK` API response.
If there is any error encountered, Firecracker simply crashes since it
never ran any guest code and trying to recover to a fresh state is
harder than simply crashing and starting anew.

Can only be called in a fresh Firecracker - one where no guest resources
have yet been configured.

### ResumeFromSnapshot Example

```bash
curl --unix-socket /tmp/firecracker.socket -i \
    -X PUT "http://localhost/snapshot/load" \
    -H  "accept: application/json" \
    -H  "Content-Type: application/json" \
    -d "{
             \"snapshot_path\": \"/path/to/snapshot\",
             \"mem_file_path\": \"/path/to/mem_file\"
    }"
```

## microVM flavors

MicroVMs are be grouped into categories: **Original** or **Clone**.

An *Original* is a microVM which has been configured and booted
from scratch, whereas a *Clone* is a microVM loaded from a snapshot.

A fresh Firecracker process will turn into an *Original microVM* or a
*Cloned microVM* depending on the API commands it gets. This divergence
is clear and irreversible.

API paths on a fresh Firecracker:
* Issuing API calls to configure devices (block, net, cpu,
  memory) or bootsource will set up this Firecracker instance as an
  *Original microVM*. The file backing the guest memory is created on
  `InstanceStart` and is *exclusively owned by this microVM* for its
  entire lifetime.
* Issuing a `PUT /snapshot/load` API request will turn it into a
  *Cloned microVM*.
  *All cloned microVMs have shared read-only ownership of the
  guest-mem-file for the full duration of their lifetime*.
  Even after the resume is complete and the clone is active, the guest
  will continue to load memory pages on demand from the snapshot file,
  therefore any external modifications to the snapshot file are not
  recommended as they can break all the clones that share it.

NOTE: For both microVM types the `logger` should be configured as
needed - configuring logger is allowed before the guest becomes active,
on original microVMs as well as clones.
