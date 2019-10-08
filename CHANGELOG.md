# Changelog

## [Unreleased]

### Changed

- Removed unused 'Halting' and 'Halted' instance states and added
  new 'Configuring' state where the microVM resources have been
  partially or fully configured, but the microVM has not been yet
  started.

### Fixed

- Errors triggered during device state restoration contain details
  identifying that device (i.e type and id).

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
