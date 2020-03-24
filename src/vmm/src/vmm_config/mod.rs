// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use libc::O_NONBLOCK;

/// Wrapper for configuring the microVM boot source.
pub mod boot_source;
/// Wrapper for configuring the block devices.
pub mod drive;
/// Wrapper over the microVM general information attached to the microVM.
pub mod instance_info;
/// Wrapper for configuring the logger.
pub mod logger;
/// Wrapper for configuring the memory and CPU of the microVM.
pub mod machine_config;
/// Wrapper for configuring the metrics.
pub mod metrics;
/// Wrapper for configuring the network devices attached to the microVM.
pub mod net;
/// Wrapper for configuring rate limiters.
pub mod rate_limiter;
/// Wrapper for configuring the vsock devices attached to the microVM.
pub mod vsock;

type Result<T> = std::result::Result<T, std::io::Error>;

/// Structure `Writer` used for writing to a FIFO.
pub struct Writer {
    line_writer: Mutex<io::LineWriter<File>>,
}

impl Writer {
    /// Create and open a FIFO for writing to it.
    /// In order to not block the instance if nobody is consuming the message that is flushed to the
    /// two pipes, we are opening it with `O_NONBLOCK` flag. In this case, writing to a pipe will
    /// start failing when reaching 64K of unconsumed content. Simultaneously,
    /// the `missed_metrics_count` metric will get increased.
    pub fn new(fifo_path: PathBuf) -> Result<Writer> {
        OpenOptions::new()
            .custom_flags(O_NONBLOCK)
            .read(true)
            .write(true)
            .open(&fifo_path)
            .map(|t| Writer {
                line_writer: Mutex::new(io::LineWriter::new(t)),
            })
    }

    fn get_line_writer(&self) -> MutexGuard<io::LineWriter<File>> {
        match self.line_writer.lock() {
            Ok(guard) => guard,
            // If a thread panics while holding this lock, the writer within should still be usable.
            // (we might get an incomplete log line or something like that).
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

impl io::Write for Writer {
    fn write(&mut self, msg: &[u8]) -> Result<(usize)> {
        let mut line_writer = self.get_line_writer();
        line_writer.write_all(msg).map(|()| msg.len())
    }

    fn flush(&mut self) -> Result<()> {
        let mut line_writer = self.get_line_writer();
        line_writer.flush()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use utils::tempfile::TempFile;

    use super::*;

    #[test]
    fn test_log_writer() {
        let log_file_temp =
            TempFile::new().expect("Failed to create temporary output logging file.");
        let good_file = log_file_temp.as_path().to_path_buf();
        let res = Writer::new(good_file);
        assert!(res.is_ok());

        let mut fw = res.unwrap();
        let msg = String::from("some message");
        assert!(fw.write(&msg.as_bytes()).is_ok());
        assert!(fw.flush().is_ok());
    }
}
