// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(warnings)]

// Currently only supports X86_64.
#![cfg(target_arch = "x86_64")]

// Allow pointer arithmetic with casts.
#![allow(clippy::ptr_offset_with_cast)]

use std::fs::File;
use std::io::{self, Read, Write};
use std::os::unix::io::AsRawFd;
use std::ptr::null_mut;
use std::slice;

/// Read/write serialized objects from/into a memory mapped snapshot file.
pub struct SnapshotReaderWriter {
    base_addr: *const libc::c_void,
    cursor: *mut libc::c_void,
    mapping_size: usize,
}

impl SnapshotReaderWriter {
    /// Read/write from/to a snapshot file, mapped at the specified address.
    pub fn new(
        file: &File,
        offset: u64,
        mapping_size: u64,
        shared_mapping: bool,
    ) -> io::Result<Self> {
        let base_addr = Self::mmap_region(file, offset, mapping_size, shared_mapping)?;
        Ok(SnapshotReaderWriter {
            base_addr,
            cursor: base_addr,
            mapping_size: mapping_size as usize,
        })
    }

    /// Maps a file into memory.
    fn mmap_region(
        file: &File,
        offset: u64,
        size: u64,
        shared_mapping: bool,
    ) -> io::Result<*mut libc::c_void> {
        let flags = libc::MAP_NORESERVE
            | if shared_mapping {
                libc::MAP_SHARED
            } else {
                libc::MAP_PRIVATE
            };
        // Safe because we are checking the return value and also unmapping in destructor.
        let addr = unsafe {
            libc::mmap(
                null_mut(),
                size as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                file.as_raw_fd(),
                offset as i64,
            )
        };
        match addr {
            libc::MAP_FAILED => Err(io::Error::last_os_error()),
            addr => Ok(addr),
        }
    }

    fn cursor_offset(&self) -> usize {
        (self.cursor as u64 - self.base_addr as u64) as usize
    }

    fn check_offset(&self, num_bytes: usize) -> std::result::Result<(), io::Error> {
        if self.cursor_offset() + num_bytes > self.mapping_size {
            Err(io::Error::from_raw_os_error(libc::EINVAL))
        } else {
            Ok(())
        }
    }
}

impl Write for SnapshotReaderWriter {
    fn write(&mut self, buf: &[u8]) -> std::result::Result<usize, io::Error> {
        self.check_offset(buf.len())?;
        unsafe {
            libc::memcpy(self.cursor, buf.as_ptr() as *const libc::c_void, buf.len());
            self.cursor = self.cursor.offset(buf.len() as isize);
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::result::Result<(), io::Error> {
        let ret = unsafe {
            libc::msync(
                self.base_addr as *mut libc::c_void,
                self.mapping_size,
                libc::MS_SYNC,
            )
        };
        if ret == -1 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

impl Read for SnapshotReaderWriter {
    fn read(&mut self, buf: &mut [u8]) -> std::result::Result<usize, io::Error> {
        self.check_offset(buf.len())?;
        let tmp_slice: &[u8] =
            unsafe { slice::from_raw_parts(self.cursor as *const u8, buf.len()) };
        buf.copy_from_slice(tmp_slice);
        self.cursor = unsafe { self.cursor.offset(buf.len() as isize) };
        Ok(buf.len())
    }
}

impl Drop for SnapshotReaderWriter {
    fn drop(&mut self) {
        unsafe {
            libc::msync(
                self.base_addr as *mut libc::c_void,
                self.mapping_size,
                libc::MS_SYNC,
            );
            libc::munmap(self.base_addr as *mut libc::c_void, self.mapping_size);
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate tempfile;

    use super::*;

    use std::fs::OpenOptions;

    use self::tempfile::NamedTempFile;

    #[test]
    fn test_read_write() {
        let tmp_file = NamedTempFile::new().unwrap();
        let fsize = 4;
        let mut buf: Vec<u8> = vec![42u8; fsize];
        let mut long_buf: Vec<u8> = vec![0u8; fsize + 1];

        // Make it a sparse empty file.
        tmp_file.as_file().set_len(fsize as u64).unwrap();
        {
            // Error case: bad memory mapping.
            let ret = SnapshotReaderWriter::new(tmp_file.as_file(), 0, 0, false);
            assert!(ret.is_err());

            let ret = SnapshotReaderWriter::new(tmp_file.as_file(), 0, fsize as u64, false);
            assert!(ret.is_ok());
            let mut reader_writer = ret.unwrap();

            // Error case: try to read more than the mapping size.
            let ret = reader_writer.read(&mut long_buf.as_mut_slice());
            assert!(ret.is_err());

            // Read an empty buffer from the empty sparse file.
            let ret = reader_writer.read(&mut buf.as_mut_slice());
            assert!(ret.is_ok());
            assert_eq!(ret.unwrap(), fsize);
            assert_eq!(buf, vec![0u8; fsize]);
        }

        {
            let ret = SnapshotReaderWriter::new(tmp_file.as_file(), 0, fsize as u64, true);
            assert!(ret.is_ok());
            let mut reader_writer = ret.unwrap();

            // Error case: try to write more than the mapping size.
            let ret = reader_writer.write(long_buf.as_slice());
            assert!(ret.is_err());

            // Write a valid value and unmap.
            let ret = reader_writer.write(&42u32.to_le_bytes());
            assert!(ret.is_ok());
            assert_eq!(ret.unwrap(), fsize);
            assert!(reader_writer.flush().is_ok());
        }

        // Save the path. This also closes the file.
        let tmp_path = tmp_file.into_temp_path();

        // Read the file, check that the data was sync'ed.
        let mut file = OpenOptions::new()
            .read(true)
            .write(false)
            .open(tmp_path)
            .unwrap();
        buf = vec![0u8; 4];
        let len_read = file.read(&mut buf).unwrap();
        assert_eq!(len_read, 4);
        let mut num_buf = [0u8; 4];
        num_buf.copy_from_slice(buf.as_slice());
        assert_eq!(u32::from_le_bytes(num_buf), 42);
    }
}
