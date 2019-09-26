use snapshot::MicrovmState;
use translator::{Error, SnapshotTranslator};

pub struct IdentitySnapshotTranslator {}

impl SnapshotTranslator for IdentitySnapshotTranslator {
    fn serialize(&self, microvm_state: &MicrovmState) -> Result<Vec<u8>, Error> {
        Ok(bincode::serialize(microvm_state).map_err(Error::Serialize)?)
    }

    fn deserialize(&self, bytes: &[u8]) -> Result<MicrovmState, Error> {
        Ok(bincode::deserialize(&bytes).map_err(Error::Deserialize)?)
    }
}
