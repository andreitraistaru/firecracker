mod identity_snapshot_translator;

use snapshot::{MicrovmState, Version};
use std::fmt::{self, Display, Formatter};
use translator::identity_snapshot_translator::IdentitySnapshotTranslator;

#[derive(Debug)]
pub enum Error {
    Deserialize(bincode::Error),
    Serialize(bincode::Error),
    SnapshotTooBig(u64, u64),
    UnimplementedSnapshotTranslator((Version, Version)),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use self::Error::*;
        match *self {
            Deserialize(ref e) => write!(f, "Failed to deserialize: {}", e),
            Serialize(ref e) => write!(f, "Failed to serialize snapshot content. {}", e),
            UnimplementedSnapshotTranslator((from, to)) => write!(
                f,
                "Unimplemented snapshot translator between versions {} and {}.",
                from, to
            ),
            SnapshotTooBig(size, max_size) => {
                write!(f, "Snapshot too big: {} > {}.", size, max_size)
            }
        }
    }
}

pub trait SnapshotTranslator {
    fn serialize(&self, microvm_state: &MicrovmState) -> Result<Vec<u8>, Error>;
    fn deserialize(&self, input: Box<dyn std::io::Read>) -> Result<MicrovmState, Error>;
}

pub fn create_snapshot_translator(
    current_app_version: Version,
    other_app_version: Version,
) -> Result<Box<dyn SnapshotTranslator>, Error> {
    match current_app_version.major() {
        v if v == other_app_version.major() => Ok(Box::new(IdentitySnapshotTranslator {})),
        _ => Err(Error::UnimplementedSnapshotTranslator((
            current_app_version,
            other_app_version,
        ))),
    }
}

#[cfg(test)]
mod tests {
    extern crate tempfile;
    use self::tempfile::NamedTempFile;
    use snapshot::{MicrovmState, SnapshotHdr, Version, VmInfo};
    use std::io::{Seek, SeekFrom, Write};
    use std::path::PathBuf;
    use std::{fs, io};
    use translator::*;

    #[test]
    fn test_error_messages() {
        #[cfg(target_env = "musl")]
        let err0_str = "No error information (os error 0)";
        #[cfg(target_env = "gnu")]
        let err0_str = "Success (os error 0)";

        assert_eq!(
            format!(
                "{}",
                Error::Deserialize(bincode::Error::from(io::Error::from_raw_os_error(0)))
            ),
            format!("Failed to deserialize: io error: {}", err0_str)
        );
        assert_eq!(
            format!(
                "{}",
                Error::Serialize(bincode::Error::from(io::Error::from_raw_os_error(0)))
            ),
            format!(
                "Failed to serialize snapshot content. io error: {}",
                err0_str
            )
        );
        assert_eq!(
            format!(
                "{}",
                Error::UnimplementedSnapshotTranslator((
                    Version::new(0, 0, 0),
                    Version::new(1, 0, 0)
                ))
            ),
            "Unimplemented snapshot translator between versions 0.0.0 and 1.0.0."
        );
    }

    #[test]
    fn test_create_snapshot_translator() {
        assert!(create_snapshot_translator(Version::new(1, 0, 0), Version::new(1, 0, 0)).is_ok());

        let ret = create_snapshot_translator(Version::new(0, 0, 0), Version::new(1, 0, 0));
        assert!(ret.is_err());
        assert_eq!(
            format!("{}", ret.err().unwrap()),
            "Unimplemented snapshot translator between versions 0.0.0 and 1.0.0."
        );
    }

    /// In `vmm/src/translator/resources/current/` we should keep a json serialized
    /// snapshot  and a binary serialized compatible with the current version of MicrovmState.
    ///
    /// In `vmm/src/translator/resources/older` we keep a binary serialized snapshot for each
    /// older version of MicrovmState. This snapshots were generated using the latest
    /// `json-vm-snapshot` at that time.
    ///
    /// This test checks 3 things:
    ///
    /// 1. That `vmm/src/translator/resources/current/json-vm-snapshot` can be deserialized into
    /// a MicrovmState instance and that if we serialize this MicrovmState with bincode we get
    /// exactly `vmm/src/translator/resources/current/binary-vm-snapshot`. This confirms that we
    /// haven't modified the structure of the MicrovmState in the current firecracker build.
    ///
    /// 2. That if we deserialize each older snapshot from
    /// `vmm/src/translator/resources/older*/binary-vm-snapshot` into a MicrovmState instance
    /// and then serialize it to binary we get
    /// `vmm/src/translator/resources/current/binary-vm-snapshot`.
    /// This should confirm that all the translations from older versions to the current version
    /// work as expected.
    ///
    /// 3. That if we serialize the current snapshot from
    /// `vmm/src/translator/resources/current/binary-vm-snapshot` into each older version
    /// we get `vmm/src/translator/resources/older*/binary-vm-snapshot`.
    /// This should confirm that all the translations from the current version to older versions
    /// work as expected.
    ///
    /// When the structure of the MicrovmState changes we should move
    /// `vmm/src/translator/resources/current/binary-vm-snapshot` to
    /// `vmm/src/translator/resources/older/[version]/binary-vm-snapshot`
    /// and update `vmm/src/translator/resources/current/json-vm-snapshot` and
    /// `vmm/src/translator/resources/current/binary-vm-snapshot` accordingly.
    #[test]
    fn test_snapshot_translators() {
        let vmm_crate_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let resources_path = vmm_crate_path.join("src/translator/resources");

        // Read the current snapshot bytes
        let current_snapshot_bytes =
            fs::read(&resources_path.join("current/binary-vm-snapshot")).unwrap();
        let current_snapshot_hdr =
            bincode::deserialize::<SnapshotHdr>(&current_snapshot_bytes).unwrap();
        // Read the current snapshot from json
        let current_snapshot_json =
            fs::read_to_string(&resources_path.join("current/json-vm-snapshot")).unwrap();
        let maybe_current_snapshot = serde_json::from_str::<MicrovmState>(&current_snapshot_json);
        assert!(
            maybe_current_snapshot.is_ok(),
            "The MicrovmState structure has changed. \
             Make sure that the Firecracker version number has been updated \
             and the translation functions have been implemented."
        );
        let current_snapshot = maybe_current_snapshot.unwrap();
        // Check that if we deserialize the current snapshot from json we get the expected bytes
        assert_eq!(
            bincode::serialize(&current_snapshot).unwrap(),
            current_snapshot_bytes
        );

        // Read all the older snapshots from "firecracker/vmm/src/translator/resources/older".
        let dir = fs::read_dir(&resources_path.join("older")).unwrap();
        for subdir in dir {
            let subdir_path = subdir.unwrap().path();
            let snapshot_path = subdir_path.join("binary-vm-snapshot");
            let old_snapshot_bytes = fs::read(subdir_path.join("binary-vm-snapshot")).unwrap();

            let old_snapshot_hdr =
                bincode::deserialize::<SnapshotHdr>(&old_snapshot_bytes).unwrap();
            let translator = create_snapshot_translator(
                old_snapshot_hdr.app_version(),
                current_snapshot_hdr.app_version(),
            )
            .unwrap();

            // Try to deserialize each old snapshot to the current snapshot format
            // and check the result.
            let maybe_old_snapshot =
                translator.deserialize(Box::new(std::fs::File::open(snapshot_path).unwrap()));
            assert!(maybe_old_snapshot.is_ok());
            let old_snapshot = maybe_old_snapshot.unwrap();
            assert_eq!(
                bincode::serialize(&old_snapshot).unwrap(),
                current_snapshot_bytes
            );

            // Try to reserialize the result into the old format and check that it matches
            let maybe_reserialized_snapshot_bytes = translator.serialize(&old_snapshot);
            assert!(maybe_reserialized_snapshot_bytes.is_ok());
            let reserialized_snapshot_bytes = maybe_reserialized_snapshot_bytes.unwrap();
            assert_eq!(reserialized_snapshot_bytes, old_snapshot_bytes);

            // Try to serialize the current snapshot into each older snapshot format
            // and check the result.
            let maybe_current_snapshot_bytes = translator.serialize(&current_snapshot);
            assert!(maybe_current_snapshot_bytes.is_ok());
            let current_snapshot_bytes = maybe_current_snapshot_bytes.unwrap();
            assert_eq!(current_snapshot_bytes, old_snapshot_bytes);
        }
    }

    fn file_len(file: &mut fs::File) -> Result<u64, std::io::Error> {
        let len = file.seek(SeekFrom::End(0))?;
        file.seek(SeekFrom::Start(0))?;

        Ok(len)
    }

    /// This test generates invalid binary snapshots based on the current binary snapshot saved
    /// in the `resources` directory. The test verifies that deserialization of a corrupt
    /// snapshot fails gracefully.
    ///
    /// The valid snapshot is corrupted in several ways:
    /// 1. Truncated to s smaller size
    /// 2. Extended to a larger size
    /// 3. Serialized byte array corresponding to a FFI structure (`kvm_pit_state`) is modified
    ///    so as to have `length=0`
    ///
    #[test]
    fn test_invalid_binary_snapshots() {
        let vmm_crate_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let resources_path = vmm_crate_path.join("src/translator/resources");

        let mut tmp_snapshot = NamedTempFile::new().unwrap();
        fs::copy(
            &resources_path.join("current/binary-vm-snapshot"),
            &tmp_snapshot.path(),
        )
        .unwrap();

        // Build the translator. We'll use a single version.
        let current_snapshot_hdr: SnapshotHdr = bincode::deserialize_from(tmp_snapshot).unwrap();
        let translator = create_snapshot_translator(
            current_snapshot_hdr.app_version(),
            current_snapshot_hdr.app_version(),
        )
        .unwrap();

        // Attempt to deserialize a truncated snapshot.
        tmp_snapshot = NamedTempFile::new().unwrap();
        fs::copy(
            &resources_path.join("current/binary-vm-snapshot"),
            &tmp_snapshot.path(),
        )
        .unwrap();

        let mut truncated_snapshot = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&tmp_snapshot.path())
            .unwrap();
        let new_len = file_len(&mut truncated_snapshot).unwrap() / 2;
        truncated_snapshot.set_len(new_len).unwrap();

        let ret = translator.deserialize(Box::new(truncated_snapshot));
        assert!(ret.is_err());
        assert_eq!(
            format!("{}", ret.err().unwrap()),
            "Failed to deserialize: io error: failed to fill whole buffer"
        );

        // Attempt to deserialize from a larger snapshot.
        tmp_snapshot = NamedTempFile::new().unwrap();
        fs::copy(
            &resources_path.join("current/binary-vm-snapshot"),
            &tmp_snapshot.path(),
        )
        .unwrap();
        let mut extended_snapshot = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&tmp_snapshot.path())
            .unwrap();

        let new_len = file_len(&mut extended_snapshot).unwrap() + 100;
        extended_snapshot.set_len(new_len).unwrap();

        let ret = translator.deserialize(Box::new(extended_snapshot));
        assert!(ret.is_ok());

        // Corrupt a serialized array.

        // Read the current snapshot bytes.
        let mut current_snapshot_bytes =
            fs::read(&resources_path.join("current/binary-vm-snapshot")).unwrap();

        // Skip past the header and VM info.
        let serialized_snapshot_hdr_size =
            bincode::serialize(&SnapshotHdr::new(Version::new(0, 0, 0)))
                .unwrap()
                .len();
        let serialized_vm_info_size = bincode::serialize(&VmInfo::new(0)).unwrap().len();

        // `bincode` serializes data structures sequentially, so the `VmState` is next.
        {
            let vmstate_bytes = &mut current_snapshot_bytes
                [serialized_snapshot_hdr_size + serialized_vm_info_size..];
            let size_len = std::mem::size_of::<usize>();
            let kvm_pit_state2_len_bytes = &mut vmstate_bytes[..size_len];

            // First member is a `kvm_pit_state2`. This is a FFI object; to serialize it, the bytes
            // that compose it are directly passed to `bincode` as a `&[u8]`. `bincode` encodes slices
            // by saving the length first, as an `usize`, then each element.
            // Replacing the length of `kvm_pit_state2`'s byte representation should result in an error.
            kvm_pit_state2_len_bytes.copy_from_slice(vec![0u8; size_len].as_slice());
        }
        tmp_snapshot = NamedTempFile::new().unwrap();
        fs::copy(
            &resources_path.join("current/binary-vm-snapshot"),
            &tmp_snapshot.path(),
        )
        .unwrap();
        let mut corrupted_snapshot = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&tmp_snapshot.path())
            .unwrap();

        corrupted_snapshot.set_len(0).unwrap();
        corrupted_snapshot
            .write_all(&current_snapshot_bytes)
            .unwrap();
        corrupted_snapshot.seek(SeekFrom::Start(0)).unwrap();

        let ret = translator.deserialize(Box::new(corrupted_snapshot));
        assert_eq!(
            format!("{}", ret.err().unwrap()),
            format!(
                "Failed to deserialize: {}",
                bincode::Error::from(bincode::ErrorKind::Custom(
                    "Incomplete buffer: size 0".to_string()
                ))
            )
        );
    }
}
