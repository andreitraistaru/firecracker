# Changelog

## [Unreleased]

### Fixed

- Raise interrupt for TX queue used descriptors - Github issue #151
- Fixed a bug that causes 100% cpu load when the net device rx is throttled 
by the ratelimiter - Github issue #152

## [0.19.1]

### Added

- Merged Firecracker 0.19.0.
- New snapshot API parameter `mem_file_path`, representing the path to the
  incremental guest memory snapshot on disk.
- New `machine-config` parameter `track_dirty_pages`, marking whether
  dirty pages should be tracked, which allows incremental snapshots to be
  created.   
- New snapshot API parameter `track_dirty_pages`, marking whether dirty pages
  should be tracked in order to generate a subsequent incremental snapshot.

### Fixed

- Fixed a logical error in bounds checking performed on vsock virtio
descriptors.

- Fixed a deserialization issue that could cause Firecracker to crash while
  attempting to load a corrupt snapshot file.

## [1.1.1]

### Added

- Documentation for guest timekeeping.
- VirtIO Balloon support.
- New command-line parameter for `firecracker`, named `--config-file`, which
  represents the path to a file that contains a JSON which can be used for
  configuring and starting a microVM without sending any API requests.
- The jailer adheres to the "end of command options" convention, meaning
  all parameters specified after `--` are forwarded verbatim to Firecracker.

### Changed

- Vsock API call: `PUT /vsocks/{id}` changed to `PUT /vsock` and no longer
  appear to support multiple vsock devices. Any subsequent calls to this API
  endpoint will override the previous vsock device configuration.
- Removed unused 'Halting' and 'Halted' instance states and added
  new 'Configuring' state where the microVM resources have been
  partially or fully configured, but the microVM has not been yet
  started.

### Fixed

- Restoring tap ifaces with names of max length.
- Errors triggered during device state restoration contain details
  identifying that device (i.e type and id).
- Upon panic, the terminal is now reset to canonical mode.
- Explicit error upon failure of vsock device creation.
- The failure message returned by an API call is flushed in the log FIFOs.

## [1.0.1]

### Added

- Added documentation for snapshotting support. See docs/snapshotting.
- Merged Firecracker 0.18.0.

### Fixed

- Snapshot is properly flushed before `/snapshot/create` API response.
- The failure message returned by an API call is flushed in the log FIFOs.

## [0.9.1]

### Added

- Added snapshotting support.
