# Changelog

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
  appear to support multiple vsock devices. Any subsequent calls to this API endpoint
  will override the previous vsock device configuration.
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
