// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

//! Track memory regions that are mapped to the guest microVM.

use std::io::{Read, Write};
use std::sync::Arc;
use std::{mem, result};

use guest_address::GuestAddress;
use mmap::{self, AnonMemoryDesc, FileMemoryDesc, MemoryMapping};
use DataInit;

/// Errors associated with handling guest memory regions.
#[derive(Debug)]
pub enum Error {
    /// Failure in creating guest memory backing file.
    CreateFile(std::io::Error),
    /// Invalid size for guest memory backing file.
    FileSize,
    /// Failure in finding a guest address in any memory regions mapped by this guest.
    InvalidGuestAddress(GuestAddress),
    /// Failure in finding a guest address range in any memory regions mapped by this guest.
    InvalidGuestAddressRange(GuestAddress, usize),
    /// Failure in accessing the memory located at some address.
    MemoryAccess(GuestAddress, mmap::Error),
    /// Failure in creating an anonymous shared mapping.
    MemoryMappingFailed(mmap::Error),
    /// Failure in initializing guest memory.
    MemoryNotInitialized,
    /// Two of the memory regions are overlapping.
    MemoryRegionOverlap,
    /// Syncing memory failed for one of the regions.
    MemorySync(std::io::Error),
    /// No memory regions were provided for initializing the guest memory.
    NoMemoryRegions,
    /// Failure in setting the size of the guest memory backing file.
    Truncate(std::io::Error),
}
type Result<T> = result::Result<T, Error>;

/// Tracks a mapping of anonymous memory in the current process and the corresponding base address
/// in the guest's memory space.
pub struct MemoryRegion {
    mapping: MemoryMapping,
    guest_base: GuestAddress,
}

impl MemoryRegion {
    pub fn size(&self) -> usize {
        self.mapping.size()
    }
}

fn region_end(region: &MemoryRegion) -> GuestAddress {
    // unchecked_add is safe as the region bounds were checked when it was created.
    region.guest_base.unchecked_add(region.mapping.size())
}

/// Tracks all memory regions allocated for the guest in the current process.
#[derive(Clone)]
pub struct GuestMemory {
    regions: Arc<Vec<MemoryRegion>>,
}

impl GuestMemory {
    /// Maps and creates a container for guest memory regions as described by the argument.
    ///
    /// # Arguments
    /// * `ranges` - a slice of `FileMemoryDesc` describing the guest memory mappings.
    pub fn new_file_backed(ranges: &[FileMemoryDesc]) -> Result<GuestMemory> {
        if ranges.is_empty() {
            return Err(Error::NoMemoryRegions);
        }

        // Guard against overlapping regions.
        let mut iter = ranges.iter();
        while let Some(range1) = iter.next() {
            for range2 in iter.clone() {
                if range1.overlap(range2) {
                    return Err(Error::MemoryRegionOverlap);
                }
            }
        }

        let mut regions = Vec::<MemoryRegion>::with_capacity(ranges.len());
        for range in ranges {
            let mapping =
                MemoryMapping::new_file_backed(range).map_err(Error::MemoryMappingFailed)?;
            regions.push(MemoryRegion {
                mapping,
                guest_base: range.gpa,
            });
        }

        Ok(GuestMemory {
            regions: Arc::new(regions),
        })
    }

    /// Maps and creates a container for anonymous guest memory given a list of memory regions.
    ///
    /// # Arguments
    /// * `ranges` - a slice of `AnonMemoryDesc` describing guest memory regions.
    pub fn new_anon(ranges: &[AnonMemoryDesc]) -> Result<GuestMemory> {
        if ranges.is_empty() {
            return Err(Error::NoMemoryRegions);
        }

        // Guard against overlapping regions.
        let mut iter = ranges.iter();
        while let Some(range1) = iter.next() {
            for range2 in iter.clone() {
                if range1.overlap(range2) {
                    return Err(Error::MemoryRegionOverlap);
                }
            }
        }

        let mut regions = Vec::<MemoryRegion>::with_capacity(ranges.len());
        for range in ranges {
            let mapping =
                MemoryMapping::new_anon(range.size).map_err(Error::MemoryMappingFailed)?;
            regions.push(MemoryRegion {
                mapping,
                guest_base: range.gpa,
            });
        }

        Ok(GuestMemory {
            regions: Arc::new(regions),
        })
    }

    /// Maps and creates a container for anonymous guest memory given a list of memory regions.
    ///
    /// # Arguments
    /// * `ranges` - a slice of tuples `(GuestAddress, usize)` describing guest memory regions.
    pub fn new_anon_from_tuples(ranges: &[(GuestAddress, usize)]) -> Result<GuestMemory> {
        let descriptors: Vec<AnonMemoryDesc> = ranges.iter().map(AnonMemoryDesc::from).collect();
        Self::new_anon(&descriptors)
    }

    /// Memory syncs the underlying mappings for all regions.
    pub fn sync(&self) -> Result<()> {
        for region in self.regions.iter() {
            region.mapping.sync().map_err(Error::MemorySync)?;
        }
        Ok(())
    }

    /// Returns the end address of memory.
    ///
    /// # Examples
    ///
    /// ```
    /// # use memory_model::{GuestAddress, GuestMemory, MemoryMapping};
    /// # fn test_end_addr() -> Result<(), ()> {
    ///     let start_addr = GuestAddress(0x1000);
    ///     let mut gm = GuestMemory::new_anon_from_tuples(&vec![(start_addr, 0x400)]).map_err(|_| ())?;
    ///     assert_eq!(start_addr.checked_add(0x400), Some(gm.end_addr()));
    ///     Ok(())
    /// # }
    /// ```
    pub fn end_addr(&self) -> GuestAddress {
        self.regions
            .iter()
            .max_by_key(|region| region.guest_base)
            .map_or(GuestAddress(0), |region| region_end(region))
    }

    /// Returns true if the given address is within the memory range available to the guest.
    pub fn address_in_range(&self, addr: GuestAddress) -> bool {
        for region in self.regions.iter() {
            if addr >= region.guest_base && addr < region_end(region) {
                return true;
            }
        }
        false
    }

    /// Returns the address plus the offset if it is in range.
    pub fn checked_offset(&self, base: GuestAddress, offset: usize) -> Option<GuestAddress> {
        if let Some(addr) = base.checked_add(offset) {
            for region in self.regions.iter() {
                if addr >= region.guest_base && addr < region_end(region) {
                    return Some(addr);
                }
            }
        }
        None
    }

    /// Returns the size of the memory region in bytes.
    pub fn num_regions(&self) -> usize {
        self.regions.len()
    }

    /// Perform the specified action on each region's addresses.
    pub fn with_regions<F, E>(&self, cb: F) -> result::Result<(), E>
    where
        F: Fn(usize, GuestAddress, usize, usize) -> result::Result<(), E>,
    {
        for (index, region) in self.regions.iter().enumerate() {
            cb(
                index,
                region.guest_base,
                region.mapping.size(),
                region.mapping.as_ptr() as usize,
            )?;
        }
        Ok(())
    }

    /// Perform the specified action on each region's addresses mutably.
    pub fn with_regions_mut<F, E>(&self, mut cb: F) -> result::Result<(), E>
    where
        F: FnMut(usize, GuestAddress, usize, usize) -> result::Result<(), E>,
    {
        for (index, region) in self.regions.iter().enumerate() {
            cb(
                index,
                region.guest_base,
                region.mapping.size(),
                region.mapping.as_ptr() as usize,
            )?;
        }
        Ok(())
    }

    /// Writes a slice to guest memory at the specified guest address.
    /// Returns the number of bytes written. The number of bytes written can
    /// be less than the length of the slice if there isn't enough room in the
    /// memory region.
    ///
    /// # Examples
    /// * Write a slice at guestaddress 0x200.
    ///
    /// ```
    /// # use memory_model::{GuestAddress, GuestMemory, MemoryMapping};
    /// # fn test_write_u64() -> Result<(), ()> {
    /// #   let start_addr = GuestAddress(0x1000);
    /// #   let mut gm = GuestMemory::new_anon_from_tuples(&vec![(start_addr, 0x400)]).map_err(|_| ())?;
    ///     let res = gm.write_slice_at_addr(&[1,2,3,4,5], GuestAddress(0x200)).map_err(|_| ())?;
    ///     assert_eq!(5, res);
    ///     Ok(())
    /// # }
    /// ```
    pub fn write_slice_at_addr(&self, buf: &[u8], guest_addr: GuestAddress) -> Result<usize> {
        self.do_in_region_partial(guest_addr, move |mapping, offset| {
            mapping
                .write_slice(buf, offset)
                .map_err(|e| Error::MemoryAccess(guest_addr, e))
        })
    }

    /// Reads to a slice from guest memory at the specified guest address.
    /// Returns the number of bytes read.  The number of bytes read can
    /// be less than the length of the slice if there isn't enough room in the
    /// memory region.
    ///
    /// # Examples
    /// * Read a slice of length 16 at guestaddress 0x200.
    ///
    /// ```
    /// # use memory_model::{GuestAddress, GuestMemory, MemoryMapping};
    /// # fn test_write_u64() -> Result<(), ()> {
    /// #   let start_addr = GuestAddress(0x1000);
    /// #   let mut gm = GuestMemory::new_anon_from_tuples(&vec![(start_addr, 0x400)]).map_err(|_| ())?;
    ///     let buf = &mut [0u8; 16];
    ///     let res = gm.read_slice_at_addr(buf, GuestAddress(0x200)).map_err(|_| ())?;
    ///     assert_eq!(16, res);
    ///     Ok(())
    /// # }
    /// ```
    pub fn read_slice_at_addr(
        &self,
        mut buf: &mut [u8],
        guest_addr: GuestAddress,
    ) -> Result<usize> {
        self.do_in_region_partial(guest_addr, move |mapping, offset| {
            mapping
                .read_slice(buf, offset)
                .map_err(|e| Error::MemoryAccess(guest_addr, e))
        })
    }

    /// Reads an object from guest memory at the given guest address.
    /// Reading from a volatile area isn't strictly safe as it could change
    /// mid-read.  However, as long as the type T is plain old data and can
    /// handle random initialization, everything will be OK.
    ///
    /// Caller needs to guarantee that the object does not cross MemoryRegion
    /// boundary, otherwise it fails.
    ///
    /// # Examples
    /// * Read a u64 from two areas of guest memory backed by separate mappings.
    ///
    /// ```
    /// # use memory_model::{GuestAddress, GuestMemory, MemoryMapping};
    /// # fn test_read_u64() -> Result<u64, ()> {
    /// #     let start_addr1 = GuestAddress(0x0);
    /// #     let start_addr2 = GuestAddress(0x400);
    /// #     let mut gm = GuestMemory::new_anon_from_tuples(&vec![(start_addr1, 0x400), (start_addr2, 0x400)])
    /// #         .map_err(|_| ())?;
    ///       let num1: u64 = gm.read_obj_from_addr(GuestAddress(32)).map_err(|_| ())?;
    ///       let num2: u64 = gm.read_obj_from_addr(GuestAddress(0x400+32)).map_err(|_| ())?;
    /// #     Ok(num1 + num2)
    /// # }
    /// ```
    pub fn read_obj_from_addr<T: DataInit>(&self, guest_addr: GuestAddress) -> Result<T> {
        self.do_in_region(guest_addr, mem::size_of::<T>(), |mapping, offset| {
            mapping
                .read_obj(offset)
                .map_err(|e| Error::MemoryAccess(guest_addr, e))
        })
    }

    /// Writes an object to the memory region at the specified guest address.
    /// Returns Ok(()) if the object fits, or Err if it extends past the end.
    ///
    /// Caller needs to guarantee that the object does not cross MemoryRegion
    /// boundary, otherwise it fails.
    ///
    /// # Examples
    /// * Write a u64 at guest address 0x1100.
    ///
    /// ```
    /// # use memory_model::{GuestAddress, GuestMemory, MemoryMapping};
    /// # fn test_write_u64() -> Result<(), ()> {
    /// #   let start_addr = GuestAddress(0x1000);
    /// #   let mut gm = GuestMemory::new_anon_from_tuples(&vec![(start_addr, 0x400)]).map_err(|_| ())?;
    ///     gm.write_obj_at_addr(55u64, GuestAddress(0x1100))
    ///         .map_err(|_| ())
    /// # }
    /// ```
    pub fn write_obj_at_addr<T: DataInit>(&self, val: T, guest_addr: GuestAddress) -> Result<()> {
        self.do_in_region(guest_addr, mem::size_of::<T>(), move |mapping, offset| {
            mapping
                .write_obj(val, offset)
                .map_err(|e| Error::MemoryAccess(guest_addr, e))
        })
    }

    /// Reads data from a readable object like a File and writes it to guest memory.
    ///
    /// # Arguments
    /// * `guest_addr` - Begin writing memory at this offset.
    /// * `src` - Read from `src` to memory.
    /// * `count` - Read `count` bytes from `src` to memory.
    ///
    /// # Examples
    ///
    /// * Read bytes from /dev/urandom
    ///
    /// ```
    /// # use memory_model::{GuestAddress, GuestMemory, MemoryMapping};
    /// # use std::fs::File;
    /// # use std::path::Path;
    /// # fn test_read_random() -> Result<u32, ()> {
    /// #     let start_addr = GuestAddress(0x1000);
    /// #     let gm = GuestMemory::new_anon_from_tuples(&vec![(start_addr, 0x400)]).map_err(|_| ())?;
    ///       let mut file = File::open(Path::new("/dev/urandom")).map_err(|_| ())?;
    ///       let addr = GuestAddress(0x1010);
    ///       gm.read_to_memory(addr, &mut file, 128).map_err(|_| ())?;
    ///       let read_addr = addr.checked_add(8).ok_or(())?;
    ///       let rand_val: u32 = gm.read_obj_from_addr(read_addr).map_err(|_| ())?;
    /// #     Ok(rand_val)
    /// # }
    /// ```
    pub fn read_to_memory<F>(
        &self,
        guest_addr: GuestAddress,
        src: &mut F,
        count: usize,
    ) -> Result<()>
    where
        F: Read,
    {
        self.do_in_region(guest_addr, count, move |mapping, offset| {
            mapping
                .read_to_memory(offset, src, count)
                .map_err(|e| Error::MemoryAccess(guest_addr, e))
        })
    }

    /// Writes data from memory to a writable object.
    ///
    /// # Arguments
    /// * `guest_addr` - Begin reading memory from this offset.
    /// * `dst` - Write from memory to `dst`.
    /// * `count` - Read `count` bytes from memory to `src`.
    ///
    /// # Examples
    ///
    /// * Write 128 bytes to /dev/null
    ///
    /// ```
    /// # use memory_model::{GuestAddress, GuestMemory, MemoryMapping};
    /// # use std::fs::File;
    /// # use std::path::Path;
    /// # fn test_write_null() -> Result<(), ()> {
    /// #     let start_addr = GuestAddress(0x1000);
    /// #     let gm = GuestMemory::new_anon_from_tuples(&vec![(start_addr, 0x400)]).map_err(|_| ())?;
    ///       let mut file = File::open(Path::new("/dev/null")).map_err(|_| ())?;
    ///       let addr = GuestAddress(0x1010);
    ///       gm.write_from_memory(addr, &mut file, 128).map_err(|_| ())?;
    /// #     Ok(())
    /// # }
    /// ```
    pub fn write_from_memory<F>(
        &self,
        guest_addr: GuestAddress,
        dst: &mut F,
        count: usize,
    ) -> Result<()>
    where
        F: Write,
    {
        self.do_in_region(guest_addr, count, move |mapping, offset| {
            mapping
                .write_from_memory(offset, dst, count)
                .map_err(|e| Error::MemoryAccess(guest_addr, e))
        })
    }

    /// Converts a GuestAddress into a pointer in the address space of this
    /// process. This should only be necessary for giving addresses to the
    /// kernel, as with vhost ioctls. Normal reads/writes to guest memory should
    /// be done through `write_from_memory`, `read_obj_from_addr`, etc.
    ///
    /// # Arguments
    /// * `guest_addr` - Guest address to convert.
    ///
    /// # Examples
    ///
    /// ```
    /// # use memory_model::{GuestAddress, GuestMemory};
    /// # fn test_host_addr() -> Result<(), ()> {
    ///     let start_addr = GuestAddress(0x1000);
    ///     let mut gm = GuestMemory::new_anon_from_tuples(&vec![(start_addr, 0x500)]).map_err(|_| ())?;
    ///     let addr = gm.get_host_address(GuestAddress(0x1200)).unwrap();
    ///     println!("Host address is {:p}", addr);
    ///     Ok(())
    /// # }
    /// ```
    pub fn get_host_address(&self, guest_addr: GuestAddress) -> Result<*const u8> {
        self.do_in_region(guest_addr, 1, |mapping, offset| {
            // This is safe; `do_in_region` already checks that offset is in
            // bounds.
            Ok(unsafe { mapping.as_ptr().add(offset) } as *const u8)
        })
    }

    /// Applies two functions, specified as callbacks, on the inner memory regions.
    ///
    /// # Arguments
    /// * `init` - Starting value of the accumulator for the `foldf` function.
    /// * `mapf` - "Map" function, applied to all the inner memory regions. It returns an array of
    ///            the same size as the memory regions array, containing the function's results
    ///            for each region.
    /// * `foldf` - "Fold" function, applied to the array returned by `mapf`. It acts as an
    ///             operator, applying itself to the `init` value and to each subsequent elemnent
    ///             in the array returned by `mapf`.
    ///
    /// # Examples
    ///
    /// * Compute the total size of all memory mappings in KB by iterating over the memory regions
    ///   and dividing their sizes to 1024, then summing up the values in an accumulator.
    ///
    /// ```
    /// # use memory_model::{GuestAddress, GuestMemory};
    /// # fn test_map_fold() -> Result<(), ()> {
    ///     let start_addr1 = GuestAddress(0x0);
    ///     let start_addr2 = GuestAddress(0x400);
    ///     let mem = GuestMemory::new_anon_from_tuples(&vec![(start_addr1, 1024), (start_addr2, 2048)]).unwrap();
    ///     let total_size = mem.map_and_fold(
    ///         0,
    ///         |(_, region)| region.size() / 1024,
    ///         |acc, size| acc + size
    ///     );
    ///     println!("Total memory size = {} KB", total_size);
    ///     Ok(())
    /// # }
    /// ```
    pub fn map_and_fold<F, G, T>(&self, init: T, mapf: F, foldf: G) -> T
    where
        F: Fn((usize, &MemoryRegion)) -> T,
        G: Fn(T, T) -> T,
    {
        self.regions.iter().enumerate().map(mapf).fold(init, foldf)
    }

    /// Read the whole object from a single MemoryRegion
    fn do_in_region<F, T>(&self, guest_addr: GuestAddress, size: usize, cb: F) -> Result<T>
    where
        F: FnOnce(&MemoryMapping, usize) -> Result<T>,
    {
        for region in self.regions.iter() {
            if guest_addr >= region.guest_base && guest_addr < region_end(region) {
                let offset = guest_addr.offset_from(region.guest_base);
                if size <= region.mapping.size() - offset {
                    return cb(&region.mapping, offset);
                }
                break;
            }
        }
        Err(Error::InvalidGuestAddressRange(guest_addr, size))
    }

    /// Read the whole or partial content from a single MemoryRegion
    fn do_in_region_partial<F>(&self, guest_addr: GuestAddress, cb: F) -> Result<usize>
    where
        F: FnOnce(&MemoryMapping, usize) -> Result<usize>,
    {
        for region in self.regions.iter() {
            if guest_addr >= region.guest_base && guest_addr < region_end(region) {
                return cb(&region.mapping, guest_addr.offset_from(region.guest_base));
            }
        }
        Err(Error::InvalidGuestAddress(guest_addr))
    }
}

#[cfg(test)]
mod tests {
    extern crate tempfile;

    use self::tempfile::tempfile;
    use super::*;
    use std::fs::File;
    use std::mem;
    use std::os::unix::io::AsRawFd;
    use std::path::Path;

    #[test]
    fn test_regions() {
        // No regions provided should return error.
        assert_eq!(
            format!(
                "{:?}",
                GuestMemory::new_anon_from_tuples(&[]).err().unwrap()
            ),
            format!("{:?}", Error::NoMemoryRegions)
        );

        let start_addr1 = GuestAddress(0x0);
        let start_addr2 = GuestAddress(0x800);
        let guest_mem =
            GuestMemory::new_anon_from_tuples(&[(start_addr1, 0x400), (start_addr2, 0x400)])
                .unwrap();
        assert_eq!(guest_mem.num_regions(), 2);
        assert!(guest_mem.address_in_range(GuestAddress(0x200)));
        assert!(!guest_mem.address_in_range(GuestAddress(0x600)));
        assert!(guest_mem.address_in_range(GuestAddress(0xa00)));
        let end_addr = GuestAddress(0xc00);
        assert!(!guest_mem.address_in_range(end_addr));
        assert_eq!(guest_mem.end_addr(), end_addr);
        assert!(guest_mem.checked_offset(start_addr1, 0x900).is_some());
        assert!(guest_mem.checked_offset(start_addr1, 0x700).is_none());
        assert!(guest_mem.checked_offset(start_addr2, 0xc00).is_none());
        assert!(guest_mem.sync().is_ok());
    }

    #[test]
    fn overlap_memory() {
        let start_addr1 = GuestAddress(0x0);
        let start_addr2 = GuestAddress(0x1000);
        let res =
            GuestMemory::new_anon_from_tuples(&[(start_addr1, 0x2000), (start_addr2, 0x2000)]);
        assert_eq!(
            format!("{:?}", res.err().unwrap()),
            format!("{:?}", Error::MemoryRegionOverlap)
        );
    }

    #[test]
    fn test_read_u64() {
        let start_addr1 = GuestAddress(0x0);
        let start_addr2 = GuestAddress(0x1000);
        let bad_addr = GuestAddress(0x2001);
        let bad_addr2 = GuestAddress(0x1ffc);

        let gm = GuestMemory::new_anon_from_tuples(&[(start_addr1, 0x1000), (start_addr2, 0x1000)])
            .unwrap();

        let val1: u64 = 0xaa55_aa55_aa55_aa55;
        let val2: u64 = 0x55aa_55aa_55aa_55aa;
        assert_eq!(
            format!("{:?}", gm.write_obj_at_addr(val1, bad_addr).err().unwrap()),
            format!(
                "InvalidGuestAddressRange({:?}, {:?})",
                bad_addr,
                std::mem::size_of::<u64>()
            )
        );
        assert_eq!(
            format!("{:?}", gm.write_obj_at_addr(val1, bad_addr2).err().unwrap()),
            format!(
                "InvalidGuestAddressRange({:?}, {:?})",
                bad_addr2,
                std::mem::size_of::<u64>()
            )
        );

        gm.write_obj_at_addr(val1, GuestAddress(0x500)).unwrap();
        gm.write_obj_at_addr(val2, GuestAddress(0x1000 + 32))
            .unwrap();
        let num1: u64 = gm.read_obj_from_addr(GuestAddress(0x500)).unwrap();
        let num2: u64 = gm.read_obj_from_addr(GuestAddress(0x1000 + 32)).unwrap();
        assert_eq!(val1, num1);
        assert_eq!(val2, num2);
    }

    #[test]
    fn write_and_read_slice() {
        let mut start_addr = GuestAddress(0x1000);
        let gm = GuestMemory::new_anon_from_tuples(&[(start_addr, 0x400)]).unwrap();
        let sample_buf = &[1, 2, 3, 4, 5];

        assert_eq!(gm.write_slice_at_addr(sample_buf, start_addr).unwrap(), 5);

        let buf = &mut [0u8; 5];
        assert_eq!(gm.read_slice_at_addr(buf, start_addr).unwrap(), 5);
        assert_eq!(buf, sample_buf);

        start_addr = GuestAddress(0x13ff);
        assert_eq!(gm.write_slice_at_addr(sample_buf, start_addr).unwrap(), 1);
        assert_eq!(gm.read_slice_at_addr(buf, start_addr).unwrap(), 1);
        assert_eq!(buf[0], sample_buf[0]);
    }

    #[test]
    fn read_to_and_write_from_mem() {
        let gm = GuestMemory::new_anon_from_tuples(&[(GuestAddress(0x1000), 0x400)]).unwrap();
        let addr = GuestAddress(0x1010);
        gm.write_obj_at_addr(!0u32, addr).unwrap();
        gm.read_to_memory(
            addr,
            &mut File::open(Path::new("/dev/zero")).unwrap(),
            mem::size_of::<u32>(),
        )
        .unwrap();
        let value: u32 = gm.read_obj_from_addr(addr).unwrap();
        assert_eq!(value, 0);

        let mut sink = Vec::new();
        gm.write_from_memory(addr, &mut sink, mem::size_of::<u32>())
            .unwrap();
        assert_eq!(sink, vec![0; mem::size_of::<u32>()]);
    }

    #[test]
    fn create_vec_with_regions() {
        let region_size = 0x400;
        let regions = vec![
            (GuestAddress(0x0), region_size),
            (GuestAddress(0x1000), region_size),
        ];
        let mut iterated_regions = Vec::new();
        let gm = GuestMemory::new_anon_from_tuples(&regions).unwrap();

        let res: Result<()> = gm.with_regions(|_, _, size, _| {
            assert_eq!(size, region_size);
            Ok(())
        });
        assert!(res.is_ok());

        let res: Result<()> = gm.with_regions_mut(|_, guest_addr, size, _| {
            iterated_regions.push((guest_addr, size));
            Ok(())
        });
        assert!(res.is_ok());
        assert_eq!(regions, iterated_regions);
        assert_eq!(gm.clone().regions[0].guest_base, regions[0].0);
        assert_eq!(gm.clone().regions[1].guest_base, regions[1].0);
    }

    // Get the base address of the mapping for a GuestAddress.
    fn get_mapping(mem: &GuestMemory, addr: GuestAddress) -> Result<*const u8> {
        mem.do_in_region(addr, 1, |mapping, _| Ok(mapping.as_ptr() as *const u8))
    }

    #[test]
    fn guest_to_host() {
        let start_addr1 = GuestAddress(0x0);
        let start_addr2 = GuestAddress(0x100);
        let mem = GuestMemory::new_anon_from_tuples(&[(start_addr1, 0x100), (start_addr2, 0x400)])
            .unwrap();

        // Verify the host addresses match what we expect from the mappings.
        let addr1_base = get_mapping(&mem, start_addr1).unwrap();
        let addr2_base = get_mapping(&mem, start_addr2).unwrap();
        let host_addr1 = mem.get_host_address(start_addr1).unwrap();
        let host_addr2 = mem.get_host_address(start_addr2).unwrap();
        assert_eq!(host_addr1, addr1_base);
        assert_eq!(host_addr2, addr2_base);

        // Check that a bad address returns an error.
        let bad_addr = GuestAddress(0x12_3456);
        assert!(mem.get_host_address(bad_addr).is_err());
    }

    #[test]
    fn test_map_fold() {
        let start_addr1 = GuestAddress(0x0);
        let start_addr2 = GuestAddress(0x400);
        let mem =
            GuestMemory::new_anon_from_tuples(&[(start_addr1, 1024), (start_addr2, 2048)]).unwrap();

        assert_eq!(
            mem.map_and_fold(
                0,
                |(_, region)| region.size() / 1024,
                |acc, size| acc + size
            ),
            3
        );
    }

    #[test]
    fn test_memory_sync() {
        let file = tempfile().unwrap();
        let size = 0x100;
        let ranges = [FileMemoryDesc {
            gpa: GuestAddress(0),
            size,
            fd: file.as_raw_fd(),
            offset: 0,
            shared: false,
        }];
        let mem = GuestMemory::new_file_backed(&ranges).unwrap();
        assert!(mem.sync().is_ok());

        // Force munmap() memory so that sync() fails.
        let paddr = mem.get_host_address(GuestAddress(0)).unwrap();
        unsafe {
            libc::munmap(paddr as *mut libc::c_void, size);
        }
        assert!(mem.sync().is_err());

        // Don't run the destructor again as it is unsafe since we already `munmap`ed.
        std::mem::forget(mem);
    }

    #[test]
    fn test_invalid_fd_for_range_descriptors() {
        let mut ranges = Vec::new();
        let range = FileMemoryDesc {
            gpa: GuestAddress(0),
            size: 1024,
            fd: -1,
            offset: 0x0010_0000,
            shared: false,
        };
        ranges.push(range);
        assert!(GuestMemory::new_file_backed(&ranges).is_err());
    }

    #[test]
    fn test_invalid_offset_for_range_descriptors() {
        let file = tempfile().unwrap();
        let mut ranges = Vec::new();
        let range = FileMemoryDesc {
            gpa: GuestAddress(0),
            size: 1024,
            fd: file.as_raw_fd(),
            offset: 3, // mmap needs page size multiples.
            shared: false,
        };
        ranges.push(range);
        let mem = GuestMemory::new_file_backed(&ranges);
        assert!(mem.is_err());

        let mut ranges = Vec::new();
        let range = FileMemoryDesc {
            gpa: GuestAddress(0),
            size: std::usize::MAX / 2 + 3, // offset + size overflows.
            fd: file.as_raw_fd(),
            offset: std::usize::MAX / 2, // offset + size overflows.
            shared: false,
        };

        ranges.push(range);
        let mem = GuestMemory::new_file_backed(&ranges);
        assert!(mem.is_err());
    }

    #[test]
    fn test_valid_range_descriptors() {
        let file = tempfile().unwrap();
        let mut ranges = Vec::new();
        let chunk_size = 0x0010_0000;
        let num_ranges = 5;
        for i in { 0..num_ranges } {
            let range = FileMemoryDesc {
                gpa: GuestAddress(i * chunk_size),
                size: chunk_size,
                fd: file.as_raw_fd(),
                offset: i * chunk_size,
                shared: false,
            };
            ranges.push(range);
        }
        let mem = GuestMemory::new_file_backed(&ranges);
        assert!(mem.is_ok());
        assert_eq!(mem.unwrap().num_regions(), num_ranges);
    }

    #[test]
    fn test_overlapping_range_descriptors() {
        let file = tempfile().unwrap();
        let chunk_size = 0x0010_0000;
        let num_ranges = 5;
        let fd = file.as_raw_fd();
        let create_ranges = move || {
            let mut ranges = Vec::new();
            for i in { 0..num_ranges } {
                let range = FileMemoryDesc {
                    gpa: GuestAddress(i * chunk_size),
                    size: chunk_size,
                    fd,
                    offset: i * chunk_size,
                    shared: false,
                };
                ranges.push(range);
            }
            ranges
        };

        let mut invalid_gpa_ranges = create_ranges();
        let invalid_range = FileMemoryDesc {
            gpa: GuestAddress(num_ranges * chunk_size / 2), // Overlaps with previous ranges.
            size: chunk_size,
            fd,
            offset: num_ranges * chunk_size,
            shared: false,
        };
        invalid_gpa_ranges.push(invalid_range);
        let mem = GuestMemory::new_file_backed(&invalid_gpa_ranges);
        assert!(mem.is_err());
        assert_eq!(
            format!("{:?}", mem.err().unwrap()),
            format!("{:?}", Error::MemoryRegionOverlap)
        );

        let mut invalid_offset_ranges = create_ranges();
        let invalid_range = FileMemoryDesc {
            gpa: GuestAddress(num_ranges * chunk_size),
            size: chunk_size,
            fd,
            offset: num_ranges * chunk_size / 2, // Overlaps with previous range.
            shared: false,
        };
        invalid_offset_ranges.push(invalid_range);
        let mem = GuestMemory::new_file_backed(&invalid_offset_ranges);
        assert!(mem.is_err());
        assert_eq!(
            format!("{:?}", mem.err().unwrap()),
            format!("{:?}", Error::MemoryRegionOverlap)
        );
    }
}
