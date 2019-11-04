use snapshot::MicrovmState;
use translator::{Error, SnapshotTranslator};

pub struct IdentitySnapshotTranslator {}

impl IdentitySnapshotTranslator {
    // Returns the maximum serialized MicrovmState size.
    pub fn get_snapshot_size_limit(&self) -> u64 {
        // TODO: Implement a way to actually compute an accurate value.
        // https://github.com/firecracker-microvm/wisp/issues/144

        // For a VM with 32 vcpus, 4 network interfaces and 7 block devices
        // the serialized MicrovmState is 436571 bytes.
        // The current value of 2MiB is more than actually needed for Wisp
        // and Stronghold usecases.
        1024 * 1024 * 2
    }
}
impl SnapshotTranslator for IdentitySnapshotTranslator {
    fn serialize(&self, microvm_state: &MicrovmState) -> Result<Vec<u8>, Error> {
        Ok(bincode::serialize(microvm_state).map_err(Error::Serialize)?)
    }

    fn deserialize(&self, input: Box<dyn std::io::Read>) -> Result<MicrovmState, Error> {
        Ok(bincode::config()
            .limit(self.get_snapshot_size_limit())
            .deserialize_from(input)
            .map_err(Error::Deserialize)?)
    }
}
