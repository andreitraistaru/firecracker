# Changelog

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
