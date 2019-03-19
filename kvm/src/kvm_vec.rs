// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

use kvm_bindings::__IncompleteArrayField;
use std::mem;
use std::mem::size_of;

/// Errors associated with the [`KvmVec`](struct.KvmVec.html) struct.
pub enum Error {
    /// The max size has been exceeded
    SizeLimitExceeded,
}

/// Trait for accessing some properties of certain KVM structures that resemble an array.
///
#[allow(clippy::len_without_is_empty)]
pub trait KvmArray {
    /// The type of entries in the zero-sized array.
    type Entry: PartialEq;

    /// Get the array length.
    fn len(&self) -> usize;

    /// Get the array length as mut.
    fn set_len(&mut self, len: usize);

    /// Get max array length.
    fn max_len() -> usize;

    /// Get the array entries.
    fn entries(&self) -> &__IncompleteArrayField<Self::Entry>;

    /// Get the array entries as mut.
    fn entries_mut(&mut self) -> &mut __IncompleteArrayField<Self::Entry>;
}

/// An adapter that helps in treating a [`KvmArray`](trait.KvmArray.html) more like an actual `Vec`.
///
pub struct KvmVec<T: Default + KvmArray> {
    // This variable holds the `KvmArray` structure. We use a `Vec<T>` To make the allocation
    // large enough while still being aligned for `T`. Only the first element of `Vec<T>` will
    // actually be used as a `T`. The remaining memory in the `Vec<T>` is for `entries`, which
    // must be contiguous. Since the entries are of type `KvmArray::Entry` we must be careful to convert the
    // desired capacity of the `KvmVec` from `KvmArray::Entry` to `T` when reserving or releasing memory.
    kvm_array: Vec<T>,
    // The number of elements of type `KvmArray::Entry` currently in the vec.
    len: usize,
    // The capacity of the `KvmVec` measured in elements of type `KvmArray::Entry`.
    capacity: usize,
}

impl<T: Default + KvmArray> KvmVec<T> {
    /// Returns the capacity required by kvm_array in order to hold the provided number of  `KvmArray::Entry`
    ///
    fn num_elements_to_kvm_array_len(num_elements: usize) -> usize {
        let size_in_bytes = size_of::<T>() + num_elements * size_of::<T::Entry>();
        (size_in_bytes + size_of::<T>() - 1) / size_of::<T>()
    }

    /// Constructs a new [`KvmVec<T>`](struct.KvmVec.html) that contains `num_elements` empty
    /// elements of type [`KvmArray::Entry`](trait.KvmArray.html#associatedtype.Entry).
    ///
    pub fn new(num_elements: usize) -> KvmVec<T> {
        let required_kvm_array_capacity = KvmVec::<T>::num_elements_to_kvm_array_len(num_elements);

        let mut kvm_array = Vec::with_capacity(required_kvm_array_capacity);
        for _ in 0..required_kvm_array_capacity {
            kvm_array.push(T::default())
        }
        kvm_array[0].set_len(num_elements);

        KvmVec {
            kvm_array,
            len: num_elements,
            capacity: num_elements,
        }
    }

    /// Creates a new structure based on a supplied array of entry-type.
    ///
    pub fn from_entries(entries: &[<T as KvmArray>::Entry]) -> KvmVec<T>
    where
        <T as KvmArray>::Entry: Copy,
    {
        let nent = entries.len();
        let mut kvm_struct = Self::new(nent);
        kvm_struct.as_mut_entries_slice().copy_from_slice(entries);

        kvm_struct
    }

    /// Returns a reference to the actual KVM structure instance.
    ///
    pub fn as_original_struct(&self) -> &T {
        &self.kvm_array[0]
    }

    /// Returns a mut reference to the actual KVM structure instance.
    ///
    pub fn as_mut_original_struct(&mut self) -> &mut T {
        &mut self.kvm_array[0]
    }

    /// Get a pointer to the KVM struct so it can be passed to the kernel.
    ///
    pub fn as_ptr(&self) -> *const T {
        self.as_original_struct()
    }

    /// Get a mutable pointer to the KVM struct so it can be passed to the kernel.
    ///
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_original_struct()
    }

    /// Returns a mut `Vec<KvmArray::Entry>` that contains all the elements.
    /// It is important to call `mem::forget` after using this vector. Otherwise rust will destroy it.
    ///
    unsafe fn as_vec(&mut self) -> Vec<T::Entry> {
        let entries_ptr = self.as_mut_original_struct().entries_mut().as_mut_ptr();
        Vec::from_raw_parts(entries_ptr, self.len, self.capacity as usize)
    }

    /// Get the mutable elements slice so they can be modified before passing to the VCPU.
    ///
    pub fn as_entries_slice(&self) -> &[T::Entry] {
        let len = self.as_original_struct().len();
        unsafe { self.as_original_struct().entries().as_slice(len as usize) }
    }

    /// Get the mutable elements slice so they can be modified before passing to the VCPU.
    ///
    pub fn as_mut_entries_slice(&mut self) -> &mut [T::Entry] {
        let len = self.as_original_struct().len();
        unsafe {
            self.as_mut_original_struct()
                .entries_mut()
                .as_mut_slice(len as usize)
        }
    }

    /// Reserves capacity for at least `additional` more `KvmArray::Entry` elements.
    /// If the capacity is already reserved, this method doesn't do anything.
    ///
    fn reserve(&mut self, additional: usize) {
        let desired_capacity = self.len + additional;
        if desired_capacity <= self.capacity {
            return;
        }

        let current_kvm_array_len = self.kvm_array.len();
        let required_kvm_array_len = KvmVec::<T>::num_elements_to_kvm_array_len(desired_capacity);
        let additional_kvm_array_len = required_kvm_array_len - current_kvm_array_len;

        self.kvm_array.reserve(additional_kvm_array_len);
        self.capacity = desired_capacity;
    }

    /// Updates the length of `self` to the specified value.
    /// Also updates the length of the `T::Entry` structure and of `self.kvm_array` accordingly.
    ///
    fn update_len(&mut self, len: usize) {
        self.len = len;
        self.as_mut_original_struct().set_len(len);

        // We need to set the len of the kvm_array to be the number of T elements needed to fit
        // an array of `len` elements of type `T::Entry`. This way, when we call
        // `self.kvm_array.shrink_to_fit()` only the unnecessary memory will be released.
        let required_kvm_array_len = KvmVec::<T>::num_elements_to_kvm_array_len(len);
        unsafe {
            self.kvm_array.set_len(required_kvm_array_len);
        }
    }

    /// Appends an element to the end of the collection.
    ///
    pub fn push(&mut self, entry: T::Entry) -> Result<(), Error> {
        let desired_len = self.len + 1;
        if desired_len > T::max_len() {
            return Err(Error::SizeLimitExceeded);
        }

        self.reserve(1);

        let mut entries = unsafe { self.as_vec() };
        entries.push(entry);
        self.update_len(desired_len);

        mem::forget(entries);

        Ok(())
    }

    /// Retains only the elements specified by the predicate.
    ///
    pub fn retain<P>(&mut self, f: P)
    where
        P: FnMut(&T::Entry) -> bool,
    {
        let mut entries = unsafe { self.as_vec() };
        entries.retain(f);

        self.update_len(entries.len());
        self.capacity = entries.len();
        self.kvm_array.shrink_to_fit();

        mem::forget(entries);
    }
}

impl<T: Default + KvmArray> PartialEq for KvmVec<T> {
    fn eq(&self, other: &KvmVec<T>) -> bool {
        self.len == other.len && self.as_entries_slice() == other.as_entries_slice()
    }
}

impl<T: Default + KvmArray> Clone for KvmVec<T> {
    fn clone(&self) -> Self {
        let mut clone = KvmVec::<T>::new(self.capacity);

        let num_bytes = self.kvm_array.len() * size_of::<T>();

        let src_byte_slice =
            unsafe { std::slice::from_raw_parts(self.as_ptr() as *const u8, num_bytes) };

        let dst_byte_slice =
            unsafe { std::slice::from_raw_parts_mut(clone.as_mut_ptr() as *mut u8, num_bytes) };

        dst_byte_slice.copy_from_slice(src_byte_slice);

        clone
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvm_bindings::*;

    const MAX_LEN: usize = 100;

    #[repr(C)]
    #[derive(Default)]
    struct MockKvmArray {
        pub len: __u32,
        pub padding: __u32,
        pub entries: __IncompleteArrayField<__u32>,
    }

    impl KvmArray for MockKvmArray {
        type Entry = u32;

        fn len(&self) -> usize {
            self.len as usize
        }

        fn set_len(&mut self, len: usize) {
            self.len = len as u32
        }

        fn max_len() -> usize {
            MAX_LEN
        }

        fn entries(&self) -> &__IncompleteArrayField<u32> {
            &self.entries
        }

        fn entries_mut(&mut self) -> &mut __IncompleteArrayField<u32> {
            &mut self.entries
        }
    }

    type MockKvmVec = KvmVec<MockKvmArray>;

    const ENTRIES_OFFSET: usize = 2;

    const NUM_ELEMENTS_TO_KVM_ARRAY_LEN: &'static [(usize, usize)] = &[
        (0, 1),
        (1, 2),
        (2, 2),
        (3, 3),
        (4, 3),
        (5, 4),
        (10, 6),
        (50, 26),
        (100, 51),
    ];

    #[test]
    fn test_to_kvm_array_capacity() {
        for pair in NUM_ELEMENTS_TO_KVM_ARRAY_LEN {
            let num_elements = pair.0;
            let required_capacity = pair.1;
            assert_eq!(
                required_capacity,
                MockKvmVec::num_elements_to_kvm_array_len(num_elements)
            );
        }
    }

    #[test]
    fn test_new() {
        let num_entries = 10;

        let kvm_vec = MockKvmVec::new(num_entries);
        assert_eq!(num_entries, kvm_vec.capacity);

        let u32_slice = unsafe {
            std::slice::from_raw_parts(kvm_vec.as_ptr() as *const u32, num_entries + ENTRIES_OFFSET)
        };
        assert_eq!(num_entries, u32_slice[0] as usize);
        for entry in u32_slice[1..].iter() {
            assert_eq!(*entry, 0);
        }
    }

    #[test]
    fn test_entries_slice() {
        let num_entries = 10;
        let mut kvm_vec = MockKvmVec::new(num_entries);

        let expected_slice = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        {
            let mut_entries_slice = kvm_vec.as_mut_entries_slice();
            mut_entries_slice.copy_from_slice(expected_slice);
        }

        let u32_slice = unsafe {
            std::slice::from_raw_parts(kvm_vec.as_ptr() as *const u32, num_entries + ENTRIES_OFFSET)
        };
        assert_eq!(expected_slice, &u32_slice[ENTRIES_OFFSET..]);
        assert_eq!(expected_slice, kvm_vec.as_entries_slice());
    }

    #[test]
    fn test_reserve() {
        let mut kvm_vec = MockKvmVec::new(0);

        // test that the right capacity is reserved
        for pair in NUM_ELEMENTS_TO_KVM_ARRAY_LEN {
            let num_elements = pair.0;
            let required_kvm_array_len = pair.1;

            kvm_vec.reserve(num_elements);

            assert!(kvm_vec.kvm_array.capacity() >= required_kvm_array_len);
            assert_eq!(0, kvm_vec.len);
            assert_eq!(num_elements, kvm_vec.capacity);
        }

        // test that when the capacity is already reserved, the method doesn't do anything
        let current_capacity = kvm_vec.capacity;
        kvm_vec.reserve(current_capacity - 1);
        assert_eq!(current_capacity, kvm_vec.capacity);
    }

    #[test]
    fn test_push() {
        let mut kvm_vec = MockKvmVec::new(0);

        for i in 0..MAX_LEN {
            assert!(kvm_vec.push(i as u32).is_ok());
            assert_eq!(kvm_vec.as_entries_slice()[i], i as u32);
        }

        assert!(kvm_vec.push(0).is_err());
    }

    #[test]
    fn test_retain() {
        let mut kvm_vec = MockKvmVec::new(0);

        for i in 0..MAX_LEN {
            assert!(kvm_vec.push(i as u32).is_ok());
        }

        kvm_vec.retain(|entry| entry % 2 == 0);

        for entry in kvm_vec.as_entries_slice().iter() {
            assert_eq!(0, entry % 2);
        }
    }

    #[test]
    fn test_partial_eq() {
        let mut kvm_vec_1 = MockKvmVec::new(0);
        let mut kvm_vec_2 = MockKvmVec::new(0);
        let mut kvm_vec_3 = MockKvmVec::new(0);

        for i in 0..MAX_LEN {
            assert!(kvm_vec_1.push(i as u32).is_ok());
            assert!(kvm_vec_2.push(i as u32).is_ok());
            assert!(kvm_vec_3.push(0).is_ok());
        }

        assert!(kvm_vec_1 == kvm_vec_2);
        assert!(kvm_vec_1 != kvm_vec_3);
    }
}
