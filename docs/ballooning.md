# Using the balloon device with Firecracker

## What is the balloon device

A memory balloon device is a virtio device that can be used to reclaim guest
memory. It does this by allocating memory in the guest, and then sending the
addresses of that memory to the host; the host may then remove that memory at
will. The device is configured through a number of options, and an integer,
which represents the target size of the balloon, in `4096` byte pages. The
options cannot be changed during operation, but the target size can be changed.
The behaviour of the balloon is the following: while the actual size of the
balloon (i.e. the number of pages it has allocated) is smaller than the target
size, it continually tries to allocate new pages -- if it fails, it prints an
error message, sleeps for some time, and then tries again. While the actual
size of the balloon is larger than the target size, it will free pages until
it hits the target size. The following options are available:

* `deflate_on_oom`: if this is set to `true`, then the kernel will deflate the
balloon rather than entering an out-of-memory condition. Note that this does
not mean that the kernel will deflate the balloon if it needs memory, but it
would not enter an out-of-memory condition without it (i.e. it will not deflate
the balloon to construct caches).
* `must_tell_host`: if this is set to `true`, the kernel will wait for host
confirmation before reclaiming memory from the host. This option is not useful
in Firecracker's implementation of the balloon device, and can safely be set to
`false`.

## Prerequisites

To support memory ballooning, you must use a kernel that has the memory
ballooning driver installed (on Linux 4.14.7, the relevant settings are
`CONFIG_MEMORY_BALLOON=y`, `CONFIG_VIRTIO_BALLOON=y`). Other than that, only
the requirements mentioned in the `getting-started` document are needed.

## Installing the balloon device

In order to use a balloon device, you must install it during virtual machine
setup (i.e. before starting the virtual machine). This can be done through the
following command:

```
socket_location=...
num_pages=...
must_tell_host=...
deflate_on_oom=...

curl --unix-socket $socket_location -i \
        -X PUT 'http://localhost/balloon' \
        -H 'Accept: application/json' \
        -H 'Content-Type: application/json' \
        -d "{
            \"num_pages\": $num_pages, \
            \"must_tell_host\": $must_tell_host, \
            \"deflate_on_oom\": $deflate_on_oom \
        }"
```

To use this, set `socket_location` to the location of the firecracker socket
(by default, at `/tmp/firecracker.socket`. Then, set `num_pages`,
`must_tell_host`, `deflate_on_oom` as desired: `num_pages` represents the
target size of the balloon, and `must_tellhost` and `deflate_on_oom` represent
the options mentioned before.

## Operating the balloon device

After it has been installed, the balloon device can then be operated via the
following command:

```
socket_location=...
num_pages=...

curl --unix-socket $socket_location -i \
        -X PATCH 'http://localhost/balloon' \
        -H 'Accept: application/json' \
        -H 'Content-Type: application/json' \
        -d "{
            \"num_pages\": $num_pages, \
        }"
```

This will update the target size of the balloon to `num_pages`. 
