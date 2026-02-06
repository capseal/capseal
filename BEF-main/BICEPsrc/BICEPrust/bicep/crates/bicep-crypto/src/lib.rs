use bicep_core::{Ensemble, Path, Time};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Light-weight field element placeholder. Swap this for a proper finite field when needed.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FieldElement(pub u64);

impl FieldElement {
    pub const MODULUS: u64 = 0x1fffffffffffffff; // 2^61 - 1

    pub fn new(value: u64) -> Self {
        FieldElement(value % Self::MODULUS)
    }

    pub fn zero() -> Self {
        FieldElement(0)
    }

    pub fn one() -> Self {
        FieldElement(1)
    }

    pub fn add(self, other: Self) -> Self {
        FieldElement::new(self.0.wrapping_add(other.0))
    }

    pub fn mul(self, other: Self) -> Self {
        let prod = (self.0 as u128) * (other.0 as u128);
        FieldElement::new((prod % Self::MODULUS as u128) as u64)
    }

    pub fn from_le_bytes(bytes: [u8; 8]) -> Self {
        FieldElement::new(u64::from_le_bytes(bytes))
    }
}

/// Public commitment emitted after streaming through the trace.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceCommitment {
    pub len: u64,
    pub root: [u8; 32],
    pub challenges: Vec<FieldElement>,
    pub sketches: Vec<FieldElement>,
}

/// Internal mutable state used during streaming (keeps challenge powers).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceCommitState {
    len: u64,
    root: [u8; 32],
    challenges: Vec<FieldElement>,
    sketches: Vec<FieldElement>,
    powers: Vec<FieldElement>,
}

#[derive(Clone, Debug)]
pub struct TraceCommitParams {
    pub num_challenges: usize,
}

impl Default for TraceCommitParams {
    fn default() -> Self {
        Self { num_challenges: 2 }
    }
}

impl TraceCommitParams {
    pub fn initial_root(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"bef-init");
        let digest = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        out
    }
}

impl TraceCommitState {
    pub fn new(params: &TraceCommitParams) -> Self {
        let root = params.initial_root();
        let challenges: Vec<FieldElement> = (0..params.num_challenges)
            .map(|j| derive_challenge_for_index(&root, j as u64))
            .collect();
        let sketches = vec![FieldElement::zero(); params.num_challenges];
        let powers = vec![FieldElement::one(); params.num_challenges];
        Self {
            len: 0,
            root,
            challenges,
            sketches,
            powers,
        }
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn root(&self) -> &[u8; 32] {
        &self.root
    }

    pub fn challenges(&self) -> &[FieldElement] {
        &self.challenges
    }

    pub fn sketches(&self) -> &[FieldElement] {
        &self.sketches
    }

    pub fn update_with_chunk(&mut self, chunk_root: &[u8; 32], values: &[FieldElement]) {
        self.root = hash_root_update(self.root, self.len, *chunk_root);
        for idx in 0..self.challenges.len() {
            let mut s = self.sketches[idx];
            let mut pow = self.powers[idx];
            let challenge = self.challenges[idx];
            for value in values {
                s = s.add(value.mul(pow));
                pow = pow.mul(challenge);
            }
            self.sketches[idx] = s;
            self.powers[idx] = pow;
        }
        self.len += values.len() as u64;
    }

    pub fn finalize(&self) -> TraceCommitment {
        TraceCommitment {
            len: self.len,
            root: self.root,
            challenges: self.challenges.clone(),
            sketches: self.sketches.clone(),
        }
    }
}

fn derive_challenge_for_index(root: &[u8; 32], idx: u64) -> FieldElement {
    let mut hasher = Sha256::new();
    hasher.update(root);
    hasher.update(&idx.to_be_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    FieldElement::from_le_bytes(bytes)
}

pub fn hash_root_update(root: [u8; 32], len: u64, chunk_root: [u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(&root);
    hasher.update(&len.to_be_bytes());
    hasher.update(&chunk_root);
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Summary of chunk commitment/sketch data for fast verification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkSketch {
    pub chunk_index: u64,
    pub offset: u64,
    pub length: u64,
    pub root: [u8; 32],
    pub sketch_vec: Vec<FieldElement>,
}

/// Fast deterministic verification operating on chunk metadata.
pub fn bef_verify_fast(
    params: &TraceCommitParams,
    commitment: &TraceCommitment,
    chunk_summaries: &[ChunkSketch],
) -> bool {
    if commitment.len == 0 {
        return chunk_summaries.is_empty();
    }
    if chunk_summaries.is_empty() {
        return false;
    }
    let num_challenges = commitment.sketches.len();
    if num_challenges == 0 || commitment.challenges.len() != num_challenges {
        return false;
    }

    let mut chunks = chunk_summaries.to_vec();
    chunks.sort_by_key(|c| c.offset);

    let mut total_len = 0u64;
    for (idx, chunk) in chunks.iter().enumerate() {
        if chunk.offset != total_len || chunk.length == 0 {
            return false;
        }
        total_len = match total_len.checked_add(chunk.length) {
            Some(v) => v,
            None => return false,
        };
        if chunk.chunk_index != idx as u64 {
            return false;
        }
        if chunk.sketch_vec.len() != num_challenges {
            return false;
        }
    }
    if total_len != commitment.len {
        return false;
    }

    let mut root_acc = params.initial_root();
    total_len = 0;
    for chunk in &chunks {
        if chunk.offset != total_len {
            return false;
        }
        root_acc = hash_root_update(root_acc, chunk.offset, chunk.root);
        total_len += chunk.length;
    }
    if root_acc != commitment.root {
        return false;
    }

    let mut aggregates = vec![FieldElement::zero(); num_challenges];
    for chunk in &chunks {
        for (acc, contrib) in aggregates.iter_mut().zip(chunk.sketch_vec.iter()) {
            *acc = acc.add(*contrib);
        }
    }
    if aggregates != commitment.sketches {
        return false;
    }

    true
}

/// Crypto accumulator state at a particular step (legacy single-challenge prototype).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CryptoState {
    pub step: u64,
    pub root: [u8; 32],
    pub sketch: FieldElement,
    pub challenge: FieldElement,
}

impl CryptoState {
    pub fn genesis() -> Self {
        let root_bytes = TraceCommitParams::default().initial_root();
        let challenge = TemporalPCS::derive_challenge(&root_bytes);
        Self {
            step: 0,
            root: root_bytes,
            sketch: FieldElement::zero(),
            challenge,
        }
    }
}

/// Events that mutate the crypto state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum UpdateEvent {
    AppendChunk(Vec<FieldElement>),
    RotateChallenge,
}

/// Optional per-step artifact (e.g., Merkle proofs) that accompanies transitions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofSnippet {
    pub placeholder: u8,
}

/// Trait for objects that know how to advance the crypto state for each event.
pub trait CryptoTransition: Send + Sync {
    fn apply(&self, state: &CryptoState, event: &UpdateEvent) -> (CryptoState, ProofSnippet);
}

/// Temporal PCS accumulator (hash + algebraic sketch) transition logic.
pub struct TemporalPCS;

impl TemporalPCS {
    pub fn new() -> Self {
        Self
    }

    fn derive_challenge(root: &[u8; 32]) -> FieldElement {
        let mut hasher = Sha256::new();
        hasher.update(root);
        let digest = hasher.finalize();
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&digest[..8]);
        FieldElement::new(u64::from_le_bytes(bytes))
    }

    fn update_sketch(
        old_sketch: FieldElement,
        challenge: FieldElement,
        start_power: FieldElement,
        values: &[FieldElement],
    ) -> FieldElement {
        let mut acc = old_sketch;
        let mut power = start_power;
        for value in values {
            acc = acc.add(value.mul(power));
            power = power.mul(challenge);
        }
        acc
    }
}

impl CryptoTransition for TemporalPCS {
    fn apply(&self, state: &CryptoState, event: &UpdateEvent) -> (CryptoState, ProofSnippet) {
        match event {
            UpdateEvent::AppendChunk(values) => {
                let next_step = state.step + 1;

                let mut hasher = Sha256::new();
                hasher.update(&state.root);
                hasher.update(&next_step.to_be_bytes());
                for value in values {
                    hasher.update(&value.0.to_le_bytes());
                }
                let digest = hasher.finalize();
                let mut new_root = [0u8; 32];
                new_root.copy_from_slice(&digest);

                let challenge = Self::derive_challenge(&new_root);
                let start_power = FieldElement::one();
                let new_sketch = Self::update_sketch(state.sketch, challenge, start_power, values);

                (
                    CryptoState {
                        step: next_step,
                        root: new_root,
                        sketch: new_sketch,
                        challenge,
                    },
                    ProofSnippet { placeholder: 0 },
                )
            }
            UpdateEvent::RotateChallenge => {
                let challenge = Self::derive_challenge(&state.root);
                (
                    CryptoState {
                        challenge,
                        ..state.clone()
                    },
                    ProofSnippet { placeholder: 0 },
                )
            }
        }
    }
}

/// Convenience aliases so callers can reuse bicep-core temporal utilities.
pub type CryptoPath = Path<CryptoState>;
pub type CryptoEnsemble = Ensemble<CryptoState>;

/// Simple helper for simulating a run of the accumulator over update events.
pub fn apply_events<T: CryptoTransition>(
    transition: &T,
    events: &[UpdateEvent],
) -> (CryptoPath, Vec<ProofSnippet>) {
    let mut state = CryptoState::genesis();
    let mut path = Path::new();
    let mut proofs = Vec::with_capacity(events.len());

    // treat discrete steps as integer times for now
    for (idx, event) in events.iter().enumerate() {
        let (next_state, proof) = transition.apply(&state, event);
        state = next_state;
        path.push(idx as Time, state.clone());
        proofs.push(proof);
    }

    (path, proofs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::fs;
    use std::path::{Path, PathBuf};

    #[derive(Deserialize)]
    struct TraceCommitmentJson {
        len: u64,
        root_hex: String,
        challenges: Vec<u64>,
        sketches: Vec<u64>,
    }

    #[derive(Deserialize)]
    struct ChunkJson {
        chunk_index: u64,
        offset: u64,
        length: u64,
        root_hex: String,
        sketch_vec: Vec<u64>,
    }

    #[derive(Deserialize)]
    struct SketchFileJson {
        trace_commitment: TraceCommitmentJson,
        chunks: Vec<ChunkJson>,
    }

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("..")
            .join("..")
    }

    fn parse_hex32(hex: &str) -> [u8; 32] {
        assert_eq!(hex.len(), 64, "hex string must be 32 bytes");
        let mut out = [0u8; 32];
        for i in 0..32 {
            let byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("invalid hex");
            out[i] = byte;
        }
        out
    }

    fn load_trace_demo() -> (TraceCommitment, Vec<ChunkSketch>) {
        let path = repo_root().join("code/sketches/trace_demo_sketch.json");
        let data = fs::read_to_string(path).expect("failed to read trace demo sketch");
        let json: SketchFileJson = serde_json::from_str(&data).expect("invalid json");

        let commit = TraceCommitment {
            len: json.trace_commitment.len,
            root: parse_hex32(&json.trace_commitment.root_hex),
            challenges: json
                .trace_commitment
                .challenges
                .iter()
                .map(|&v| FieldElement::new(v))
                .collect(),
            sketches: json
                .trace_commitment
                .sketches
                .iter()
                .map(|&v| FieldElement::new(v))
                .collect(),
        };

        let chunks = json
            .chunks
            .into_iter()
            .map(|chunk| ChunkSketch {
                chunk_index: chunk.chunk_index,
                offset: chunk.offset,
                length: chunk.length,
                root: parse_hex32(&chunk.root_hex),
                sketch_vec: chunk
                    .sketch_vec
                    .into_iter()
                    .map(FieldElement::new)
                    .collect(),
            })
            .collect();

        (commit, chunks)
    }

    fn assert_metadata_consistency(
        commitment: &TraceCommitment,
        chunks: &[ChunkSketch],
        params: &TraceCommitParams,
    ) {
        let mut sorted = chunks.to_vec();
        sorted.sort_by_key(|c| c.offset);
        let mut total_len = 0u64;
        for (idx, chunk) in sorted.iter().enumerate() {
            assert_eq!(chunk.offset, total_len, "chunk offset mismatch");
            assert!(chunk.length > 0, "chunk length must be > 0");
            total_len = total_len.checked_add(chunk.length).expect("len overflow");
            assert_eq!(chunk.chunk_index, idx as u64, "chunk index mismatch");
            assert_eq!(chunk.sketch_vec.len(), commitment.challenges.len());
        }
        assert_eq!(total_len, commitment.len, "total length mismatch");

        let mut root = params.initial_root();
        for chunk in &sorted {
            root = hash_root_update(root, chunk.offset, chunk.root);
        }
        assert_eq!(root, commitment.root, "root mismatch");

        let mut aggregates = vec![FieldElement::zero(); commitment.challenges.len()];
        for chunk in &sorted {
            for (acc, contrib) in aggregates.iter_mut().zip(chunk.sketch_vec.iter()) {
                *acc = acc.add(*contrib);
            }
        }
        assert_eq!(aggregates, commitment.sketches, "sketch aggregate mismatch");
    }

    #[test]
    fn bef_verify_fast_accepts_trace_demo() {
        let (commitment, chunks) = load_trace_demo();
        let params = TraceCommitParams {
            num_challenges: commitment.challenges.len(),
        };
        assert_metadata_consistency(&commitment, &chunks, &params);
        assert!(bef_verify_fast(&params, &commitment, &chunks));
    }

    #[test]
    fn bef_verify_fast_rejects_sketch_tamper() {
        let (commitment, mut chunks) = load_trace_demo();
        chunks[0].sketch_vec[0] = chunks[0].sketch_vec[0].add(FieldElement::one());
        let params = TraceCommitParams {
            num_challenges: commitment.challenges.len(),
        };
        assert!(!bef_verify_fast(&params, &commitment, &chunks));
    }

    #[test]
    fn bef_verify_fast_rejects_root_tamper() {
        let (commitment, mut chunks) = load_trace_demo();
        chunks[0].root[0] ^= 1;
        let params = TraceCommitParams {
            num_challenges: commitment.challenges.len(),
        };
        assert!(!bef_verify_fast(&params, &commitment, &chunks));
    }
}
