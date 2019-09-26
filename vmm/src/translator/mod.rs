mod identity_snapshot_translator;

use snapshot::{MicrovmState, Version};
use std::fmt::{self, Display, Formatter};
use translator::identity_snapshot_translator::IdentitySnapshotTranslator;

#[derive(Debug)]
pub enum Error {
    Deserialize(bincode::Error),
    Serialize(bincode::Error),
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
                "Unimplemented snapshot translator from version {} to version {}.",
                from, to
            ),
        }
    }
}

pub trait SnapshotTranslator {
    fn serialize(&self, microvm_state: &MicrovmState) -> Result<Vec<u8>, Error>;

    fn deserialize(&self, bytes: &[u8]) -> Result<MicrovmState, Error>;
}

pub fn create_snapshot_translator(
    from: Version,
    to: Version,
) -> Result<Box<SnapshotTranslator>, Error> {
    match from.major() {
        v if v == to.major() => Ok(Box::new(IdentitySnapshotTranslator {})),
        _ => Err(Error::UnimplementedSnapshotTranslator((from, to))),
    }
}

#[cfg(test)]
mod tests {
    use snapshot::Version;
    use std::io;
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
            "Unimplemented snapshot translator from version 0.0.0 to version 1.0.0."
        );
    }

    #[test]
    fn test_create_snapshot_translator() {
        assert!(create_snapshot_translator(Version::new(1, 0, 0), Version::new(1, 0, 0)).is_ok());

        let ret = create_snapshot_translator(Version::new(0, 0, 0), Version::new(1, 0, 0));
        assert!(ret.is_err());
        assert_eq!(
            format!("{}", ret.err().unwrap()),
            "Unimplemented snapshot translator from version 0.0.0 to version 1.0.0."
        );
    }
}
