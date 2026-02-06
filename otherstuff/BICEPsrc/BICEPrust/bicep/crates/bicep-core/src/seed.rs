use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Identity describing a stochastic path/sequence.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SeedIdentity {
    instrument_id: String,
    date_bucket: String,
    ensemble_id: u64,
    path_or_sequence_id: u64,
}

impl SeedIdentity {
    pub fn new(
        instrument_id: impl Into<String>,
        date_bucket: impl Into<String>,
        ensemble_id: u64,
        path_or_sequence_id: u64,
    ) -> Self {
        Self {
            instrument_id: instrument_id.into().to_lowercase(),
            date_bucket: date_bucket.into(),
            ensemble_id,
            path_or_sequence_id,
        }
    }

    pub fn legacy_from_path(path_id: u64) -> Self {
        Self::new("legacy-path", "1970-01-01", 0, path_id)
    }

    pub fn canonical_json_bytes(&self) -> Vec<u8> {
        let mut map = BTreeMap::new();
        map.insert("date_bucket", self.date_bucket.clone());
        map.insert("ensemble_id", self.ensemble_id.to_string());
        map.insert("instrument_id", self.instrument_id.clone());
        map.insert("path_or_sequence_id", self.path_or_sequence_id.to_string());
        serde_json::to_vec(&map).expect("SeedIdentity canonical serialization")
    }
}

#[derive(Clone, Debug)]
pub struct SeedSpec {
    key_id: String,
    key_bytes: [u8; 32],
}

impl SeedSpec {
    pub fn new(key_id: impl Into<String>, key_bytes: [u8; 32]) -> Self {
        Self {
            key_id: key_id.into(),
            key_bytes,
        }
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub fn derive_seed(&self, identity: &SeedIdentity) -> u64 {
        let canonical = identity.canonical_json_bytes();
        let hash = blake3::keyed_hash(&self.key_bytes, &canonical);
        let bytes = hash.as_bytes();
        let mut out = [0u8; 8];
        out.copy_from_slice(&bytes[..8]);
        u64::from_le_bytes(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_is_stable() {
        let identity = SeedIdentity::new("AAPL", "2025-01-01", 7, 42);
        let json = identity.canonical_json_bytes();
        let expected = b"{\"date_bucket\":\"2025-01-01\",\"ensemble_id\":\"7\",\"instrument_id\":\"aapl\",\"path_or_sequence_id\":\"42\"}";
        assert_eq!(json, expected);
    }

    #[test]
    fn derive_seed_changes_with_identity() {
        let spec = SeedSpec::new("test", [1u8; 32]);
        let id_a = SeedIdentity::new("AAPL", "2025-01-01", 0, 1);
        let id_b = SeedIdentity::new("AAPL", "2025-01-02", 0, 1);
        assert_ne!(spec.derive_seed(&id_a), spec.derive_seed(&id_b));
    }
}
