// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

//! The mmap module provides a safe interface to mmap memory and ensures unmap is called when the
//! mmap object leaves scope.

use std;
use std::io::{self, Read, Write};
use std::os::unix::io::RawFd;
use std::ptr::null_mut;
use std::sync::Mutex;

use libc;

use guest_address::GuestAddress;
use DataInit;

/// Errors associated with memory mapping.
#[derive(Debug)]
pub enum Error {
    /// Requested memory out of range.
    InvalidAddress,
    /// Requested memory range spans past the end of the region.
    InvalidRange(usize, usize),
    /// Couldn't read from the given source.
    ReadFromSource(io::Error),
    /// `mmap` or `madvise` returned the given error.
    SystemCallFailed(io::Error),
    /// Writing to memory failed.
    WriteToMemory(io::Error),
    /// Reading from memory failed.
    ReadFromMemory(io::Error),
    /// A madvise was called with unsupported advice.
    UnsupportedMadvise(libc::c_int),
}
type Result<T> = std::result::Result<T, Error>;

fn range_overlap(range1: (usize, usize), range2: (usize, usize)) -> bool {
    let first_start = std::cmp::min(range1.0, range2.0);
    let second_start = std::cmp::max(range1.0, range2.0);
    let first_size = if first_start == range1.0 {
        range1.1
    } else {
        range2.1
    };
    if first_start
        .checked_add(first_size)
        .map_or(true, |first_end| first_end > second_start)
    {
        return true;
    }
    false
}

/// Describes an anonymous memory region mapping.
pub struct AnonMemoryDesc {
    /// Guest physical address.
    pub gpa: GuestAddress,
    /// Size of the memory region.
    pub size: usize,
}

impl AnonMemoryDesc {
    /// Returns true if the two memory regions overlap.
    pub fn overlap(&self, other: &AnonMemoryDesc) -> bool {
        range_overlap((self.gpa.0, self.size), (other.gpa.0, other.size))
    }
}

impl From<&(GuestAddress, usize)> for AnonMemoryDesc {
    fn from(tuple: &(GuestAddress, usize)) -> Self {
        AnonMemoryDesc {
            gpa: tuple.0,
            size: tuple.1,
        }
    }
}

/// Describes a file-backed memory region mapping.
pub struct FileMemoryDesc {
    /// Guest physical address.
    pub gpa: GuestAddress,
    /// Size of the memory region.
    pub size: usize,
    /// File descriptor of backing file.
    pub fd: RawFd,
    /// Offset in file where mapping starts.
    pub offset: usize,
    /// Visibility of mapping.
    pub shared: bool,
}

impl FileMemoryDesc {
    /// Returns true if the two memory region mappings overlap. Overlap occurs when either:
    ///   1) The [`GuestAddress`](struct.GuestAddress.html)es overlap.
    ///   2) The physical backings overlap.
    pub fn overlap(&self, other: &FileMemoryDesc) -> bool {
        range_overlap((self.gpa.0, self.size), (other.gpa.0, other.size))
            || (self.fd == other.fd
                && range_overlap((self.offset, self.size), (other.offset, other.size)))
    }
}

/// Wraps an anonymous shared memory mapping in the current process.
pub struct MemoryMapping {
    addr: *mut u8,
    size: usize,
    // `dirty_areas` tracks the areas of memory which have been written to through the
    // `MemoryMapping` interface.
    dirty_areas: Option<Mutex<Bitmap>>,
}

// Send and Sync aren't automatically inherited for the raw address pointer.
// Accessing that pointer is only done through the stateless interface which
// allows the object to be shared by multiple threads without a decrease in
// safety.
unsafe impl Send for MemoryMapping {}
unsafe impl Sync for MemoryMapping {}

impl MemoryMapping {
    /// Creates a shared memory mapping of described by a `FileMemoryDesc` descriptor.
    ///
    /// # Arguments
    /// * `descriptor` - `FileMemoryDesc` describing mapping details.
    pub fn new_file_backed(
        descriptor: &FileMemoryDesc,
        track_dirty_pages: bool,
    ) -> Result<MemoryMapping> {
        let addr = unsafe {
            libc::mmap(
                null_mut(),
                descriptor.size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_NORESERVE
                    | if descriptor.shared {
                        libc::MAP_SHARED
                    } else {
                        libc::MAP_PRIVATE
                    },
                descriptor.fd,
                descriptor.offset as i64,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(Error::SystemCallFailed(io::Error::last_os_error()));
        }
        // If we're tracking dirty pages, then create the bitmap that does the tracking
        let dirty_areas = if track_dirty_pages {
            Some(Mutex::new(Bitmap::new(descriptor.size, 4096)))
        } else {
            None
        };
        Ok(MemoryMapping {
            addr: addr as *mut u8,
            size: descriptor.size,
            dirty_areas,
        })
    }

    /// Creates an anonymous shared memory mapping.
    ///
    /// # Arguments
    /// * `size` - Size of the memory mapping.
    pub fn new_anon(size: usize, track_dirty_pages: bool) -> Result<MemoryMapping> {
        let addr = unsafe {
            libc::mmap(
                null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | libc::MAP_SHARED,
                -1,
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(Error::SystemCallFailed(io::Error::last_os_error()));
        }
        // If we're tracking dirty pages, then create the bitmap that does the tracking
        let dirty_areas = if track_dirty_pages {
            Some(Mutex::new(Bitmap::new(size, 4096)))
        } else {
            None
        };
        Ok(MemoryMapping {
            addr: addr as *mut u8,
            size,
            dirty_areas,
        })
    }

    /// Returns a pointer to the beginning of the memory region.  Should only be
    /// used for passing this region to ioctls for setting guest memory.
    pub fn as_ptr(&self) -> *mut u8 {
        self.addr
    }

    /// Memory syncs the underlying mappings for all regions.
    pub fn sync(&self) -> io::Result<()> {
        // Safe because we check the return value.
        let ret = unsafe { libc::msync(self.addr as *mut libc::c_void, self.size, libc::MS_SYNC) };
        if ret == -1 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    /// Returns the size of the memory region in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get a copy of the dirty bitmap for this region
    pub fn get_dirty_bitmap(&self) -> Option<Bitmap> {
        // This unwrap() is OK because the only reason it would fail is because another thread
        // panicked while holding this lock, in which case the FC process would be dead anyway.
        if let Some(dirty_areas) = &self.dirty_areas {
            Some(dirty_areas.lock().unwrap().clone())
        } else {
            None
        }
    }

    /// Mark the pages dirty in the range from `offset` to `offset+len`
    fn mark_areas_dirty(&self, offset: usize, len: usize) {
        // This unwrap() is OK because the only reason it would fail is because another thread
        // panicked while holding this lock, in which case the FC process would be dead anyway.
        if let Some(dirty_areas) = &self.dirty_areas {
            dirty_areas.lock().unwrap().set_addr_range(offset, len);
        }
    }

    /// Writes a slice to the memory region at the specified offset.
    /// Returns the number of bytes written.  The number of bytes written can
    /// be less than the length of the slice if there isn't enough room in the
    /// memory region.
    ///
    /// # Examples
    /// * Write a slice at offset 256.
    ///
    /// ```
    /// #   use memory_model::MemoryMapping;
    /// #   let mut mem_map = MemoryMapping::new_anon(1024, true).unwrap();
    ///     let res = mem_map.write_slice(&[1,2,3,4,5], 256);
    ///     assert!(res.is_ok());
    ///     assert_eq!(res.unwrap(), 5);
    /// ```
    pub fn write_slice(&self, buf: &[u8], offset: usize) -> Result<usize> {
        if offset >= self.size {
            return Err(Error::InvalidAddress);
        }
        self.mark_areas_dirty(offset, buf.len());
        unsafe {
            // Guest memory can't strictly be modeled as a slice because it is
            // volatile.  Writing to it with what compiles down to a memcpy
            // won't hurt anything as long as we get the bounds checks right.
            let mut slice: &mut [u8] = &mut self.as_mut_slice()[offset..];
            Ok(slice.write(buf).map_err(Error::WriteToMemory)?)
        }
    }

    /// Reads to a slice from the memory region at the specified offset.
    /// Returns the number of bytes read.  The number of bytes read can
    /// be less than the length of the slice if there isn't enough room in the
    /// memory region.
    ///
    /// # Examples
    /// * Read a slice of size 16 at offset 256.
    ///
    /// ```
    /// #   use memory_model::MemoryMapping;
    /// #   let mut mem_map = MemoryMapping::new_anon(1024, true).unwrap();
    ///     let buf = &mut [0u8; 16];
    ///     let res = mem_map.read_slice(buf, 256);
    ///     assert!(res.is_ok());
    ///     assert_eq!(res.unwrap(), 16);
    /// ```
    pub fn read_slice(&self, mut buf: &mut [u8], offset: usize) -> Result<usize> {
        if offset >= self.size {
            return Err(Error::InvalidAddress);
        }
        unsafe {
            // Guest memory can't strictly be modeled as a slice because it is
            // volatile.  Writing to it with what compiles down to a memcpy
            // won't hurt anything as long as we get the bounds checks right.
            let slice: &[u8] = &self.as_slice()[offset..];
            Ok(buf.write(slice).map_err(Error::ReadFromMemory)?)
        }
    }

    /// Writes an object to the memory region at the specified offset.
    /// Returns Ok(()) if the object fits, or Err if it extends past the end.
    ///
    /// # Examples
    /// * Write a u64 at offset 16.
    ///
    /// ```
    /// #   use memory_model::MemoryMapping;
    /// #   let mut mem_map = MemoryMapping::new_anon(1024, true).unwrap();
    ///     let res = mem_map.write_obj(55u64, 16);
    ///     assert!(res.is_ok());
    /// ```
    pub fn write_obj<T: DataInit>(&self, val: T, offset: usize) -> Result<()> {
        unsafe {
            // Guest memory can't strictly be modeled as a slice because it is
            // volatile.  Writing to it with what compiles down to a memcpy
            // won't hurt anything as long as we get the bounds checks right.
            let (end, fail) = offset.overflowing_add(std::mem::size_of::<T>());
            if fail || end > self.size {
                return Err(Error::InvalidAddress);
            }
            std::ptr::write_volatile(&mut self.as_mut_slice()[offset..] as *mut _ as *mut T, val);
        }
        self.mark_areas_dirty(offset, std::mem::size_of::<T>());
        Ok(())
    }

    /// Reads on object from the memory region at the given offset.
    /// Reading from a volatile area isn't strictly safe as it could change
    /// mid-read.  However, as long as the type T is plain old data and can
    /// handle random initialization, everything will be OK.
    ///
    /// # Examples
    /// * Read a u64 written to offset 32.
    ///
    /// ```
    /// #   use memory_model::MemoryMapping;
    /// #   let mut mem_map = MemoryMapping::new_anon(1024, true).unwrap();
    ///     let res = mem_map.write_obj(55u64, 32);
    ///     assert!(res.is_ok());
    ///     let num: u64 = mem_map.read_obj(32).unwrap();
    ///     assert_eq!(55, num);
    /// ```
    pub fn read_obj<T: DataInit>(&self, offset: usize) -> Result<T> {
        let (end, fail) = offset.overflowing_add(std::mem::size_of::<T>());
        if fail || end > self.size {
            return Err(Error::InvalidAddress);
        }
        unsafe {
            // This is safe because by definition Copy types can have their bits
            // set arbitrarily and still be valid.
            Ok(std::ptr::read_volatile(
                &self.as_slice()[offset..] as *const _ as *const T,
            ))
        }
    }

    /// Reads data from a readable object like a File and writes it to guest memory.
    ///
    /// # Arguments
    /// * `mem_offset` - Begin writing memory at this offset.
    /// * `src` - Read from `src` to memory.
    /// * `count` - Read `count` bytes from `src` to memory.
    ///
    /// # Examples
    ///
    /// * Read bytes from /dev/urandom
    ///
    /// ```
    /// # use memory_model::MemoryMapping;
    /// # use std::fs::File;
    /// # use std::path::Path;
    /// # fn test_read_random() -> Result<u32, ()> {
    /// #     let mut mem_map = MemoryMapping::new_anon(1024, true).unwrap();
    ///       let mut file = File::open(Path::new("/dev/urandom")).map_err(|_| ())?;
    ///       mem_map.read_to_memory(32, &mut file, 128).map_err(|_| ())?;
    ///       let rand_val: u32 =  mem_map.read_obj(40).map_err(|_| ())?;
    /// #     Ok(rand_val)
    /// # }
    /// ```
    pub fn read_to_memory<F>(&self, mem_offset: usize, src: &mut F, count: usize) -> Result<()>
    where
        F: Read,
    {
        let (mem_end, fail) = mem_offset.overflowing_add(count);
        if fail || mem_end > self.size {
            return Err(Error::InvalidRange(mem_offset, count));
        }
        unsafe {
            // It is safe to overwrite the volatile memory. Accessing the guest
            // memory as a mutable slice is OK because nothing assumes another
            // thread won't change what is loaded.
            let dst = &mut self.as_mut_slice()[mem_offset..mem_end];
            src.read_exact(dst).map_err(Error::ReadFromSource)?;
        }
        self.mark_areas_dirty(mem_offset, count);
        Ok(())
    }

    /// Writes data from memory to a writable object.
    ///
    /// # Arguments
    /// * `mem_offset` - Begin reading memory from this offset.
    /// * `dst` - Write from memory to `dst`.
    /// * `count` - Read `count` bytes from memory to `src`.
    ///
    /// # Examples
    ///
    /// * Write 128 bytes to /dev/null
    ///
    /// ```
    /// # use memory_model::MemoryMapping;
    /// # use std::fs::File;
    /// # use std::path::Path;
    /// # fn test_write_null() -> Result<(), ()> {
    /// #     let mut mem_map = MemoryMapping::new_anon(1024, true).unwrap();
    ///       let mut file = File::open(Path::new("/dev/null")).map_err(|_| ())?;
    ///       mem_map.write_from_memory(32, &mut file, 128).map_err(|_| ())?;
    /// #     Ok(())
    /// # }
    /// ```
    pub fn write_from_memory<F>(&self, mem_offset: usize, dst: &mut F, count: usize) -> Result<()>
    where
        F: Write,
    {
        let (mem_end, fail) = mem_offset.overflowing_add(count);
        if fail || mem_end > self.size {
            return Err(Error::InvalidRange(mem_offset, count));
        }
        unsafe {
            // It is safe to read from volatile memory. Accessing the guest
            // memory as a slice is OK because nothing assumes another thread
            // won't change what is loaded.
            let src = &self.as_mut_slice()[mem_offset..mem_end];
            dst.write_all(src).map_err(Error::ReadFromSource)?;
        }
        Ok(())
    }

    unsafe fn as_slice(&self) -> &[u8] {
        // This is safe because we mapped the area at addr ourselves, so this slice will not
        // overflow. However, it is possible to alias.
        std::slice::from_raw_parts(self.addr, self.size)
    }

    #[allow(clippy::mut_from_ref)]
    unsafe fn as_mut_slice(&self) -> &mut [u8] {
        // This is safe because we mapped the area at addr ourselves, so this slice will not
        // overflow. However, it is possible to alias.
        std::slice::from_raw_parts_mut(self.addr, self.size)
    }

    /// Call madvise on a certain range, with a certain advice.
    ///
    /// Only MADV_DONTNEED and MADV_REMOVE are currently supported.
    ///
    /// If called with MADV_DONTNEED it takes back pages from the guest. If read from after this
    /// call, the memory will contain only zeroes, if the underlying memory region is an anonymous
    /// private mapping, or will result in repopulating the memory contents from the up-to-date
    /// contents of the underlying mapped file. The given offset must be page aligned.
    ///
    /// If called with MADV_REMOVE it takes back pages from the guest. If read from after this
    /// call, the memory will contain only zeroes. The given offset must be page aligned.
    ///
    /// To learn more about madvise, read this manual page:
    /// http://man7.org/linux/man-pages/man2/madvise.2.html
    ///
    /// # Examples
    ///
    /// ```
    /// # use memory_model::MemoryMapping;
    /// # fn test_dontneed_range() -> Result<(), ()> {
    ///   // Function use for shared anonymous mappings.
    ///   let mut mem_map = MemoryMapping::new_anon(1024, true).unwrap();
    ///   assert!(mem_map.write_obj(123u32, 0).is_ok());
    /// # assert_eq!(mem_map.read_obj::<u32>(32).unwrap(), 123u32);
    ///   assert!(mem_map.madvise_range(0, 32, libc::MADV_DONTNEED).is_ok());
    ///   // The kernel is now advised that the first 32 bytes of `mem_map` is not needed
    ///   // A read will now yield the old contents, since the mapping is anonymous and shared.
    ///   assert_eq!(mem_map.read_obj::<u32>(32).unwrap(), 123u32);
    /// # Ok(())
    /// # }
    ///
    /// # fn test_remove_range() -> Result<(), ()> {
    ///   let mut mem_map = MemoryMapping::new_anon(1024, true).unwrap();
    ///   assert!(mem_map.write_obj(123u32, 0).is_ok());
    ///   assert!(mem_map.madvise_range(0, 32, libc::MADV_REMOVE).is_ok());
    ///   // The kernel is now advised that the first 32 bytes of `mem_map` can be removed.
    ///   assert_eq!(mem_map.read_obj::<u32>(32).unwrap(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn madvise_range(
        &self,
        mem_offset: usize,
        count: usize,
        advice: libc::c_int,
    ) -> Result<()> {
        if advice != libc::MADV_REMOVE && advice != libc::MADV_DONTNEED {
            return Err(Error::UnsupportedMadvise(advice));
        }

        self.range_end(mem_offset, count)
            .map_err(|_| Error::InvalidRange(mem_offset, count))?;

        let ret = unsafe {
            // This is safe for the same reason that write_obj is safe:
            // `madvise`-ing the kernel that some memory can be removed is,
            // from the point of view of a userspace process, either the same as
            // setting the memory to 0 (for remove or dontneed) -- which is fine as
            // long as we make sure that we get the bound check correct
            // (which was done through the call to range_end previously); or,
            // does not modify it, which is clearly safe (for dontneed).
            libc::madvise((self.addr as usize + mem_offset) as *mut _, count, advice)
        };
        if ret < 0 {
            Err(Error::SystemCallFailed(io::Error::last_os_error()))
        } else {
            Ok(())
        }
    }

    /// Check that offset + count falls within our mapped range,
    /// and return the result.
    fn range_end(&self, offset: usize, count: usize) -> Result<usize> {
        let mem_end = offset.checked_add(count).ok_or(Error::InvalidAddress)?;
        if mem_end > self.size() {
            return Err(Error::InvalidAddress);
        }
        Ok(mem_end)
    }
}

impl Drop for MemoryMapping {
    fn drop(&mut self) {
        // This is safe because we mmap the area at addr ourselves, and nobody
        // else is holding a reference to it.
        unsafe {
            libc::munmap(self.addr as *mut libc::c_void, self.size);
        }
    }
}

/// `Bitmap` implements a simple bit map on the page level with test and set operations. It is
/// page-size aware, so it converts addresses to page numbers before setting or clearing the bits.
#[derive(Debug, Clone)]
pub struct Bitmap {
    map: Vec<u64>,
    size: usize,
    page_size: usize,
}

impl Bitmap {
    /// Create a new bitmap of `byte_size`, with one bit per `page_size`.
    /// In reality this is rounded up, and you get a new vector of the next multiple of 64 bigger
    /// than `size` for free.
    fn new(byte_size: usize, page_size: usize) -> Self {
        // Bit size is the number of bits in the bitmap, always at least 1 (to store the state of
        // the '0' address).
        let bit_size = std::cmp::max(1, byte_size / page_size);
        Bitmap {
            map: vec![0; ((bit_size - 1) >> 6) + 1],
            size: bit_size,
            page_size,
        }
    }

    /// Is bit `n` set? Bits outside the range of the bitmap are always unset.
    #[inline]
    fn is_bit_set(&self, n: usize) -> bool {
        if n <= self.size {
            (self.map[n >> 6] & (1 << (n & 63))) != 0
        } else {
            // Out-of-range bits are always unset.
            false
        }
    }

    /// Is the bit corresponding to address `addr` set?
    pub fn is_addr_set(&self, addr: usize) -> bool {
        self.is_bit_set(addr / self.page_size)
    }

    /// Set a range of bits starting at `start_addr` and continuing for the next `len` bytes.
    pub fn set_addr_range(&mut self, start_addr: usize, len: usize) {
        let first_bit = start_addr / self.page_size;
        let page_count = (len + self.page_size - 1) / self.page_size;
        for n in first_bit..(first_bit + page_count) {
            if n > self.size {
                // Attempts to set bits beyond the end of the bitmap are simply ignored.
                break;
            }
            self.map[n >> 6] |= 1 << (n & 63);
        }
    }

    /// Get the length of the bitmap in bits (i.e. in how many pages it can represent).
    pub fn len(&self) -> usize {
        self.size
    }

    /// Is the bitmap empty (i.e. has zero size)? This is always false, because we explicitly
    /// round up the size when creating the bitmap.
    pub fn is_empty(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    extern crate tempfile;

    use self::tempfile::tempfile;
    use super::*;
    use std::fs::File;
    use std::mem;
    use std::path::Path;

    /// This asserts that $lhs matches $rhs.
    macro_rules! assert_match {
        ($lhs:expr, $rhs:pat) => {{
            assert!(match $lhs {
                $rhs => true,
                _ => false,
            })
        }};
    }

    #[test]
    fn bitmap_basic() {
        let mut b = Bitmap::new(1024, 128);
        assert_eq!(b.is_empty(), false);
        assert_eq!(b.len(), 8);
        b.set_addr_range(128, 129);
        assert!(!b.is_addr_set(0));
        assert!(b.is_addr_set(128));
        assert!(b.is_addr_set(256));
        assert!(!b.is_addr_set(384));
    }

    #[test]
    fn bitmap_out_of_range() {
        let mut b = Bitmap::new(1024, 128);
        // Set a partial range that goes beyond the end of the bitmap
        b.set_addr_range(768, 512);
        assert!(b.is_addr_set(768));
        // The bitmap is never set beyond its end
        assert!(!b.is_addr_set(1152));
    }

    #[test]
    fn basic_map() {
        let m = MemoryMapping::new_anon(1024, true).unwrap();
        assert_eq!(1024, m.size());
    }

    #[test]
    fn map_invalid_size() {
        let res = MemoryMapping::new_anon(0, true);
        match res {
            Ok(_) => panic!("should panic!"),
            Err(err) => {
                if let Error::SystemCallFailed(e) = err {
                    assert_eq!(e.raw_os_error(), Some(libc::EINVAL));
                } else {
                    panic!("unexpected error: {:?}", err);
                }
            }
        }
    }

    #[test]
    fn test_write_past_end() {
        let m = MemoryMapping::new_anon(5, true).unwrap();
        let res = m.write_slice(&[1, 2, 3, 4, 5, 6], 0);
        assert!(res.is_ok());
        assert_eq!(res.unwrap(), 5);
    }

    #[test]
    fn slice_read_and_write() {
        let mem_map = MemoryMapping::new_anon(5, true).unwrap();
        let sample_buf = [1, 2, 3];
        assert!(mem_map.write_slice(&sample_buf, 5).is_err());
        assert!(mem_map.write_slice(&sample_buf, 2).is_ok());
        let mut buf = [0u8; 3];
        assert!(mem_map.read_slice(&mut buf, 5).is_err());
        assert!(mem_map.read_slice(&mut buf, 2).is_ok());
        assert_eq!(buf, sample_buf);
    }

    #[test]
    fn obj_read_and_write() {
        let mem_map = MemoryMapping::new_anon(5, true).unwrap();
        assert!(mem_map.write_obj(55u16, 4).is_err());
        assert!(mem_map.write_obj(55u16, core::usize::MAX).is_err());
        assert!(mem_map.write_obj(55u16, 2).is_ok());
        assert_eq!(mem_map.read_obj::<u16>(2).unwrap(), 55u16);
        assert!(mem_map.read_obj::<u16>(4).is_err());
        assert!(mem_map.read_obj::<u16>(core::usize::MAX).is_err());
    }

    #[test]
    fn test_unsupported_madvise() {
        let mem_map = MemoryMapping::new_anon(1024, false).unwrap();
        let unsupported_madvises = vec![
            libc::MADV_DODUMP,
            libc::MADV_DOFORK,
            libc::MADV_DONTDUMP,
            libc::MADV_DONTFORK,
            libc::MADV_HUGEPAGE,
            libc::MADV_HWPOISON,
            libc::MADV_MERGEABLE,
            libc::MADV_NOHUGEPAGE,
            libc::MADV_NORMAL,
            libc::MADV_RANDOM,
            libc::MADV_SEQUENTIAL,
            libc::MADV_UNMERGEABLE,
            libc::MADV_WILLNEED,
        ];
        for &advice in unsupported_madvises.iter() {
            if let Error::UnsupportedMadvise(x) = mem_map.madvise_range(1, 2, advice).unwrap_err() {
                assert_eq!(x, advice);
            } else {
                panic!();
            }
        }
    }

    #[test]
    fn test_remove_range() {
        let mem_map = MemoryMapping::new_anon(1024, false).unwrap();

        // This should fail, since it's out of the mapping's bounds.
        assert_match!(
            mem_map.madvise_range(600, 600, libc::MADV_REMOVE),
            Err(Error::InvalidRange(600, 600))
        );

        // This should fail, since this address is not page aligned.
        #[allow(clippy::match_wild_err_arm)]
        match mem_map.madvise_range(1, 100, libc::MADV_REMOVE) {
            Ok(_) => panic!("This remove_range should return an error"),
            Err(Error::SystemCallFailed(x)) => assert_eq!(x.kind(), io::ErrorKind::InvalidInput),
            _ => panic!("This remove_range returned the wrong type of error"),
        }

        // Write, to test later that this disappears after the madvise.
        assert!(mem_map.write_obj(123 as u32, 0).is_ok());

        // Check that write was succesful.
        assert_match!(mem_map.read_obj::<u32>(0), Ok(123));

        // Remove range. This should succeed.
        assert!(mem_map.madvise_range(0, 500, libc::MADV_REMOVE).is_ok());

        // Now, reading the integer at 0 should yield 0.
        assert_match!(mem_map.read_obj::<u32>(0), Ok(0));
    }

    #[test]
    fn test_dontneed_range() {
        let mem_map = MemoryMapping::new_anon(1024, false).unwrap();

        // This should fail, since it's out of the mapping's bounds.
        assert_match!(
            mem_map.madvise_range(600, 600, libc::MADV_DONTNEED),
            Err(Error::InvalidRange(600, 600))
        );

        // This should fail, since this address is not page aligned.
        #[allow(clippy::match_wild_err_arm)]
        match mem_map.madvise_range(1, 100, libc::MADV_DONTNEED) {
            Ok(_) => panic!("This dontneed_range should give an error"),
            Err(Error::SystemCallFailed(x)) => assert_eq!(x.kind(), io::ErrorKind::InvalidInput),
            Err(_) => panic!("This dontneed_range returned the wrong type of error"),
        }

        // This should fail, since it's out of the mapping's bounds.
        assert_match!(
            mem_map.madvise_range(600, 600, libc::MADV_DONTNEED),
            Err(Error::InvalidRange(600, 600))
        );

        // Write, to test later that this disappears after the madvise.
        assert!(mem_map.write_obj(123 as u32, 0).is_ok());

        // Check that write was succesful.
        assert_match!(mem_map.read_obj::<u32>(0), Ok(123));

        // Remove range. This should succeed.
        assert!(mem_map.madvise_range(0, 500, libc::MADV_DONTNEED).is_ok());

        // Now, reading the integer at 0 should yield the previous value, since the
        // mapping is shared.
        assert_match!(mem_map.read_obj::<u32>(0), Ok(123));
    }

    #[test]
    fn test_range_end() {
        let mm = MemoryMapping::new_anon(123, false).unwrap();

        // This should work, since 50 + 50 < 123.
        assert_match!(mm.range_end(50, 50), Ok(100));

        // This should return an error, since 50 + 80 >= 123.
        assert_match!(mm.range_end(50, 80), Err(Error::InvalidAddress));
    }

    #[test]
    fn test_dirty_bitmap() {
        let mem_map = MemoryMapping::new_anon(4096 * 3, true).unwrap();
        // write_obj should dirty the bitmap
        assert!(!mem_map.get_dirty_bitmap().unwrap().is_addr_set(0));
        assert!(mem_map.write_obj(55u16, 2).is_ok());
        assert!(mem_map.get_dirty_bitmap().unwrap().is_addr_set(0));

        // write_slice should dirty the bitmap
        let sample_buf = [1, 2, 3];
        assert!(!mem_map.get_dirty_bitmap().unwrap().is_addr_set(4096));
        assert!(mem_map.write_slice(&sample_buf, 4096).is_ok());
        assert!(mem_map.get_dirty_bitmap().unwrap().is_addr_set(4096));
    }

    #[test]
    fn mem_read_and_write() {
        let mem_map = MemoryMapping::new_anon(5, false).unwrap();
        assert!(mem_map.write_obj(!0u32, 1).is_ok());
        let mut file = File::open(Path::new("/dev/zero")).unwrap();
        assert!(mem_map
            .read_to_memory(2, &mut file, mem::size_of::<u32>())
            .is_err());
        assert!(mem_map
            .read_to_memory(core::usize::MAX, &mut file, mem::size_of::<u32>())
            .is_err());

        assert!(mem_map
            .read_to_memory(1, &mut file, mem::size_of::<u32>())
            .is_ok());

        let mut f = tempfile().unwrap();
        assert!(mem_map
            .read_to_memory(1, &mut f, mem::size_of::<u32>())
            .is_err());
        format!(
            "{:?}",
            mem_map.read_to_memory(1, &mut f, mem::size_of::<u32>())
        );

        assert_eq!(mem_map.read_obj::<u32>(1).unwrap(), 0);

        let mut sink = Vec::new();
        assert!(mem_map
            .write_from_memory(1, &mut sink, mem::size_of::<u32>())
            .is_ok());
        assert!(mem_map
            .write_from_memory(2, &mut sink, mem::size_of::<u32>())
            .is_err());
        assert!(mem_map
            .write_from_memory(core::usize::MAX, &mut sink, mem::size_of::<u32>())
            .is_err());
        format!(
            "{:?}",
            mem_map.write_from_memory(2, &mut sink, mem::size_of::<u32>())
        );
        assert_eq!(sink, vec![0; mem::size_of::<u32>()]);
    }
}
