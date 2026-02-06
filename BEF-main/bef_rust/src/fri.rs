use pyo3::prelude::*;
use rayon::prelude::*;
use sha2::{Sha256, Digest};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const MODULUS: u64 = (1 << 61) - 1;

// Binary archive constants (matching Python BinaryChunkArchive)
const BINARY_MAGIC: &[u8; 4] = b"STCA";
const BINARY_VERSION: u32 = 1;

fn add_mod(a: u64, b: u64) -> u64 {
    let s = a + b;
    if s >= MODULUS { s - MODULUS } else { s }
}

fn mul_mod(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % (MODULUS as u128)) as u64
}

// Simple Merkle Tree helpers (re-implemented local to avoid pub crate issues for now)
fn hash_pair(left: &[u8], right: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

/// Hash multiple children together (for k-ary Merkle trees)
fn hash_children(children: &[[u8; 32]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for child in children {
        hasher.update(child);
    }
    hasher.finalize().into()
}

/// Compute Merkle root with configurable arity (default 2 = binary)
fn compute_kary_merkle_root(leaves: &[[u8; 32]], arity: usize) -> [u8; 32] {
    if leaves.is_empty() { return [0; 32]; }
    if arity < 2 { return [0; 32]; }

    let mut current = leaves.to_vec();
    while current.len() > 1 {
        let mut next = Vec::with_capacity((current.len() + arity - 1) / arity);
        for block in current.chunks(arity) {
            // ALWAYS hash, even single-element blocks (pad with duplicate)
            // Matches Python: right = current[i+1] if i+1 < len else current[i]
            let mut padded = block.to_vec();
            while padded.len() < arity {
                padded.push(*padded.last().unwrap());
            }
            next.push(hash_children(&padded));
        }
        current = next;
    }
    current[0]
}

fn compute_merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    compute_kary_merkle_root(leaves, 2)
}

#[allow(dead_code)]
struct FriLayer {
    values: Vec<u64>,
    leaves: Vec<[u8; 32]>,
    root: [u8; 32],
    levels: Vec<Vec<[u8; 32]>>,
}

impl FriLayer {
    fn new(values: Vec<u64>) -> Self {
        // Parallel leaf hashing
        let leaves: Vec<[u8; 32]> = values.par_iter().map(|&v| {
            let mut hasher = Sha256::new();
            // Matching Python _int_to_bytes(v, 32)
            let val_bytes = v.to_be_bytes();
            let mut padded = [0u8; 32];
            padded[24..32].copy_from_slice(&val_bytes);
            hasher.update(&padded);
            hasher.finalize().into()
        }).collect();

        // Build full Merkle tree levels for O(log N) path generation
        let mut levels = vec![leaves.clone()];
        let mut current = leaves.clone();
        while current.len() > 1 {
            let mut next = Vec::with_capacity((current.len() + 1) / 2);
            for i in (0..current.len()).step_by(2) {
                let left = current[i];
                let right = if i + 1 < current.len() { current[i + 1] } else { left };
                next.push(hash_pair(&left, &right));
            }
            levels.push(next.clone());
            current = next;
        }

        let root = if current.is_empty() { [0; 32] } else { current[0] };
        
        FriLayer { values, leaves, root, levels }
    }

    #[allow(dead_code)]
    fn get_path(&self, index: usize) -> Vec<[u8; 32]> {
        if self.levels.is_empty() { return vec![]; }
        
        let mut path = Vec::new();
        let mut idx = index;
        
        // Use cached levels (O(log N))
        for level in self.levels.iter().take(self.levels.len() - 1) {
            let sibling_idx = idx ^ 1;
            if sibling_idx < level.len() {
                path.push(level[sibling_idx]);
            } else {
                path.push(level[idx]); // Duplication for odd edges
            }
            idx /= 2;
        }
        path
    }
}

#[pyclass]
pub struct FriProver {
    #[allow(dead_code)]
    layers: Vec<FriLayer>,
    #[allow(dead_code)]
    final_poly: Vec<u64>,
}

#[pymethods]
impl FriProver {
    #[new]
    fn new(initial_values: Vec<u64>, final_degree: usize) -> Self {
        let mut layers = Vec::new();
        let current_values = initial_values;

        // Commit phase loop
        while current_values.len() > final_degree + 1 {
            // 1. Commit current layer
            let layer = FriLayer::new(current_values.clone());
            
            // 2. Derive challenge (alpha/beta) from root
            // Note: In a real non-interactive FRI, we'd hash the transcript.
            // For this optimization step, we accept the challenge from Python 
            // OR we implement the challenge derivation here. 
            // To simplify integration, let's assume we do the folding math here
            // but we might need the verifier challenges.
            //
            // Python implementation:
            //   transcript.absorb(root)
            //   alpha = transcript.challenge()
            //
            // If we move the LOOP to Rust, Rust needs to know how to derive challenges.
            // Or we pass a list of pre-determined challenges?
            //
            // Let's implement a `fold_step` method instead of a full loop constructor.
            // This allows Python to drive the transcript (keeping logic identical)
            // while Rust does the heavy lifting.
            
            // WAIT: The prompt said "move the entire loop".
            // So I should implement the transcript logic too.
            //
            // Let's defer challenge derivation. I'll implement `commit_layer` 
            // and `fold_next_layer` exposed to Python. 
            // This gives Python control over the protocol (challenges) but Rust speed.
            
            layers.push(layer);
            
            // Break loop - we will let Python drive the recursion via method calls.
            break; 
        }
        
        FriProver {
            layers: Vec::new(), // Initialized empty, populated by calls
            final_poly: Vec::new(),
        }
    }
    
    fn commit_layer(&mut self, values: Vec<u64>) -> String {
        let layer = FriLayer::new(values);
        let root_hex = hex::encode(layer.root);
        self.layers.push(layer);
        root_hex
    }
    
    fn fold_next_layer(&mut self, alpha: u64) -> Vec<u64> {
        // L_{i+1}[j] = L_i[2j] + alpha * L_i[2j+1]
        // This matches `fold_codeword` in bef_zk/fri/domain.py.
        let prev = &self.layers.last().expect("no layers").values;
        let n = prev.len();
        if n % 2 != 0 {
            // Should not happen for power-of-two domains
            return Vec::new(); 
        }
        let next_n = n / 2;
        
        // Use GPU folding if available
        let next_values: Vec<u64> = if next_n >= 4096 {
            crate::gpu_shim::fold_on_gpu(prev, alpha).unwrap_or_else(|| {
                (0..next_n).into_par_iter().map(|i| {
                    let a = prev[2 * i];
                    let b = prev[2 * i + 1];
                    add_mod(a, mul_mod(alpha, b))
                }).collect()
            })
        } else {
             (0..next_n).into_par_iter().map(|i| {
                let a = prev[2 * i];
                let b = prev[2 * i + 1];
                add_mod(a, mul_mod(alpha, b))
            }).collect()
        };
        
        // We do NOT commit here. The caller (Python) decides when to commit.
        // But to be useful as a "prover", we probably want to commit immediately?
        // Python `_build_layers`: folds, THEN commits `next_codeword`.
        // So we return the values, Python commits them (using `commit_trace_batch` or just passing them back to `commit_layer`).
        // For efficiency, we should probably commit internally.
        
        // Let's change this method to `commit_fold`.
        // But Python architecture separates "vc.commit" from "fold".
        // To support "move entire loop", we should encapsulate the loop here.
        
        next_values
    }
}

// Helper to expose FRI folding as a standalone function for Python to use with existing VC
#[pyfunction]
pub fn fold_fri_layer(values: Vec<u64>, alpha: u64) -> Vec<u64> {
    let n = values.len();
    let next_n = n / 2;
    // Use GPU if threshold met
    if next_n >= 4096 {
        if let Some(res) = crate::gpu_shim::fold_on_gpu(&values, alpha) {
            return res;
        }
    }
    (0..next_n).into_par_iter().map(|i| {
        let a = values[2 * i];
        let b = values[2 * i + 1];
        add_mod(a, mul_mod(alpha, b))
    }).collect()
}

// ============================================================================
// PyFriState: Stateful FRI prover with layer caching for efficient openings
// ============================================================================

/// Result of a fold_and_commit operation
#[pyclass]
#[derive(Clone)]
pub struct PyFriCommitResult {
    #[pyo3(get)]
    pub root_hex: String,
    #[pyo3(get)]
    pub length: usize,
    #[pyo3(get)]
    pub num_chunks: usize,
    #[pyo3(get)]
    pub chunk_roots_hex: Vec<String>,
    #[pyo3(get)]
    pub archive_dir: Option<String>,
    #[pyo3(get)]
    pub archive_format: String,
}

/// Merkle multiproof for k-ary trees (matches Python MerkleMultiProof)
#[pyclass]
#[derive(Clone)]
pub struct PyMerkleMultiProof {
    #[pyo3(get)]
    pub tree_size: usize,
    #[pyo3(get)]
    pub arity: usize,
    #[pyo3(get)]
    pub sibling_levels: Vec<Vec<String>>, // hex-encoded
}

/// Single entry in a batch proof (matches Python VCBatchEntry)
#[pyclass]
#[derive(Clone)]
pub struct PyBatchEntry {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub value: u64,
    #[pyo3(get)]
    pub chunk_index: usize,
    #[pyo3(get)]
    pub chunk_offset: usize,
    #[pyo3(get)]
    pub leaf_pos: usize,
}

/// Chunk leaf proof (matches Python ChunkLeafProof)
#[pyclass]
#[derive(Clone)]
pub struct PyChunkLeafProof {
    #[pyo3(get)]
    pub chunk_index: usize,
    #[pyo3(get)]
    pub chunk_offset: usize,
    #[pyo3(get)]
    pub leaf_positions: Vec<usize>,
    #[pyo3(get)]
    pub proof: PyMerkleMultiProof,
}

/// Batch proof for multiple indices (matches Python VCBatchProof)
#[pyclass]
#[derive(Clone)]
pub struct PyBatchProof {
    #[pyo3(get)]
    pub entries: Vec<PyBatchEntry>,
    #[pyo3(get)]
    pub chunk_positions: Vec<usize>,
    #[pyo3(get)]
    pub chunk_roots: Vec<String>, // hex-encoded
    #[pyo3(get)]
    pub chunk_proof: PyMerkleMultiProof,
    #[pyo3(get)]
    pub chunk_leaf_proofs: Vec<PyChunkLeafProof>,
}

/// Cached layer data for efficient openings
#[derive(Clone)]
struct LayerCache {
    values: Vec<u64>,
    chunk_len: usize,
    chunk_tree_arity: usize,
    chunk_roots: Vec<[u8; 32]>,
    chunk_root_levels: Vec<Vec<[u8; 32]>>,  // k-ary tree levels over chunk roots
    global_root: [u8; 32],
}

/// Stateful FRI prover with layer caching for efficient openings.
///
/// Usage:
///   state = PyFriState(base_evaluations)
///   state.commit_and_cache(chunk_len, arity)  # Initial layer
///   for round in range(num_rounds):
///       beta = derive_beta(...)  # Python does transcript
///       state.fold_commit_and_cache(beta, chunk_len, arity)
///   # Now open_batch works on any layer
///   proof = state.open_batch(layer_idx, indices)
#[pyclass]
pub struct PyFriState {
    current: Vec<u64>,
    layers: Vec<LayerCache>,
}

#[pymethods]
impl PyFriState {
    #[new]
    pub fn new(codeword: Vec<u64>) -> Self {
        PyFriState {
            current: codeword,
            layers: Vec::new(),
        }
    }

    /// Get current codeword length
    pub fn len(&self) -> usize {
        self.current.len()
    }

    /// Get number of cached layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get a copy of the current codeword (for debugging/parity checks)
    pub fn get_codeword(&self) -> Vec<u64> {
        self.current.clone()
    }

    /// Commit current codeword and cache for later openings (for initial layer)
    #[pyo3(signature = (chunk_len, chunk_tree_arity=None))]
    pub fn commit_and_cache(
        &mut self,
        chunk_len: usize,
        chunk_tree_arity: Option<usize>,
    ) -> PyResult<PyFriCommitResult> {
        if chunk_len == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "chunk_len must be > 0",
            ));
        }
        let arity = chunk_tree_arity.unwrap_or(16);
        let cache = build_layer_cache(&self.current, chunk_len, arity)?;
        let result = cache_to_result(&cache);
        self.layers.push(cache);
        Ok(result)
    }

    /// Fold current codeword by beta, commit, cache, and update state.
    #[pyo3(signature = (beta, chunk_len, chunk_tree_arity=None))]
    pub fn fold_commit_and_cache(
        &mut self,
        beta: u64,
        chunk_len: usize,
        chunk_tree_arity: Option<usize>,
    ) -> PyResult<PyFriCommitResult> {
        if self.current.len() % 2 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "FRI fold requires even length codeword",
            ));
        }
        if chunk_len == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "chunk_len must be > 0",
            ));
        }

        let beta = beta % MODULUS;
        let arity = chunk_tree_arity.unwrap_or(16);

        // 1) Fold codeword (GPU Optimized)
        let n = self.current.len();
        let next_n = n / 2;
        
        let folded: Vec<u64> = if next_n >= 4096 {
            crate::gpu_shim::fold_on_gpu(&self.current, beta).unwrap_or_else(|| {
                 (0..next_n).into_par_iter().map(|i| {
                    let a = self.current[2 * i];
                    let b = self.current[2 * i + 1];
                    add_mod(a, mul_mod(beta, b))
                }).collect()
            })
        } else {
             (0..next_n).into_par_iter().map(|i| {
                let a = self.current[2 * i];
                let b = self.current[2 * i + 1];
                add_mod(a, mul_mod(beta, b))
            }).collect()
        };

        // 2) Build cache (includes commitment)
        let cache = build_layer_cache(&folded, chunk_len, arity)?;
        let result = cache_to_result(&cache);

        // 3) Update state
        self.layers.push(cache);
        self.current = folded;

        Ok(result)
    }

    /// Run the entire FRI proving loop in Rust (Pure Rust Transcript)
    #[pyo3(signature = (num_rounds, chunk_len, chunk_tree_arity=None))]
    pub fn prove_all(
        &mut self,
        num_rounds: usize,
        chunk_len: usize,
        chunk_tree_arity: Option<usize>,
    ) -> PyResult<Vec<PyFriCommitResult>> {
        if self.layers.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("Base layer not committed"));
        }

        let mut results = Vec::new();
        
        for round_idx in 0..num_rounds {
             // Derive beta from last layer root (Python _derive_beta logic)
             let last_layer = self.layers.last().unwrap();
             let root = last_layer.global_root;
             
             let mut hasher = Sha256::new();
             hasher.update(&root);
             hasher.update(&(round_idx as u32).to_be_bytes());
             let digest = hasher.finalize();
             
             // Reduce digest to field element
             let mut beta: u128 = 0;
             let m = MODULUS as u128;
             for &b in digest.as_slice() {
                 beta = (beta * 256 + (b as u128)) % m;
             }
             let mut beta_u64 = beta as u64;
             if beta_u64 == 0 { beta_u64 = 1; }
             
             let res = self.fold_commit_and_cache(beta_u64, chunk_len, chunk_tree_arity)?;
             results.push(res);
        }
        
        Ok(results)
    }

    /// Open batch of indices for a specific layer (Milestone 2 core method)
    pub fn open_batch(&self, layer_idx: usize, indices: Vec<usize>) -> PyResult<PyBatchProof> {
        if layer_idx >= self.layers.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("layer_idx {} out of range (have {} layers)", layer_idx, self.layers.len())
            ));
        }
        if indices.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("indices cannot be empty"));
        }

        let layer = &self.layers[layer_idx];
        open_batch_from_cache(layer, &indices)
    }

    /// Legacy: Fold current codeword by beta, commit+archive, update state.
    /// Does NOT cache for openings. Use fold_commit_and_cache for full support.
    #[pyo3(signature = (beta, chunk_len, archive_dir=None, archive_format=None, chunk_tree_arity=None))]
    pub fn fold_and_commit(
        &mut self,
        beta: u64,
        chunk_len: usize,
        archive_dir: Option<String>,
        archive_format: Option<String>,
        chunk_tree_arity: Option<usize>,
    ) -> PyResult<PyFriCommitResult> {
        if self.current.len() % 2 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "FRI fold requires even length codeword",
            ));
        }
        if chunk_len == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "chunk_len must be > 0",
            ));
        }

        let beta = beta % MODULUS;
        let arity = chunk_tree_arity.unwrap_or(16);

        let n = self.current.len();
        let next_n = n / 2;
        let folded: Vec<u64> = (0..next_n).into_par_iter().map(|i| {
            let a = self.current[2 * i];
            let b = self.current[2 * i + 1];
            add_mod(a, mul_mod(beta, b))
        }).collect();

        let format = archive_format.unwrap_or_else(|| "json".to_string());
        let result = commit_codeword_chunked(&folded, chunk_len, &archive_dir, &format, arity)?;

        self.current = folded;
        Ok(result)
    }

    /// Legacy: Commit current codeword without folding (for initial layer)
    #[pyo3(signature = (chunk_len, archive_dir=None, archive_format=None, chunk_tree_arity=None))]
    pub fn commit_current(
        &self,
        chunk_len: usize,
        archive_dir: Option<String>,
        archive_format: Option<String>,
        chunk_tree_arity: Option<usize>,
    ) -> PyResult<PyFriCommitResult> {
        if chunk_len == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "chunk_len must be > 0",
            ));
        }
        let format = archive_format.unwrap_or_else(|| "json".to_string());
        let arity = chunk_tree_arity.unwrap_or(16);
        commit_codeword_chunked(&self.current, chunk_len, &archive_dir, &format, arity)
    }
}

// ============================================================================
// Layer caching and opening helpers
// ============================================================================

/// Build full layer cache including Merkle tree levels for efficient openings
fn build_layer_cache(
    values: &[u64],
    chunk_len: usize,
    chunk_tree_arity: usize,
) -> PyResult<LayerCache> {
    if values.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("empty codeword"));
    }

    let num_chunks = (values.len() + chunk_len - 1) / chunk_len;

    // Compute chunk roots in parallel
    let chunk_roots: Vec<[u8; 32]> = (0..num_chunks).into_par_iter().map(|idx| {
        let start = idx * chunk_len;
        let end = std::cmp::min(start + chunk_len, values.len());
        let chunk = &values[start..end];
        let offset = start as u64;
        compute_chunk_root_fri(chunk, offset)
    }).collect();

    // Build k-ary tree levels over chunk roots
    let chunk_root_levels = build_kary_levels(&chunk_roots, chunk_tree_arity);
    let global_root = if chunk_root_levels.is_empty() {
        [0u8; 32]
    } else {
        chunk_root_levels.last().unwrap()[0]
    };

    Ok(LayerCache {
        values: values.to_vec(),
        chunk_len,
        chunk_tree_arity,
        chunk_roots,
        chunk_root_levels,
        global_root,
    })
}

/// Convert layer cache to PyFriCommitResult
fn cache_to_result(cache: &LayerCache) -> PyFriCommitResult {
    PyFriCommitResult {
        root_hex: hex::encode(cache.global_root),
        length: cache.values.len(),
        num_chunks: cache.chunk_roots.len(),
        chunk_roots_hex: cache.chunk_roots.iter().map(|r| hex::encode(r)).collect(),
        archive_dir: None,
        archive_format: "none".to_string(),
    }
}

/// Build k-ary Merkle tree levels (not just root)
fn build_kary_levels(leaves: &[[u8; 32]], arity: usize) -> Vec<Vec<[u8; 32]>> {
    if leaves.is_empty() || arity < 2 {
        return vec![];
    }

    let mut levels: Vec<Vec<[u8; 32]>> = vec![leaves.to_vec()];
    let mut current = leaves.to_vec();

    while current.len() > 1 {
        let mut next = Vec::with_capacity((current.len() + arity - 1) / arity);
        for block in current.chunks(arity) {
            // ALWAYS hash, even single-element blocks (pad with duplicate)
            // Matches Python build_levels / build_kary_levels
            let mut padded = block.to_vec();
            while padded.len() < arity {
                padded.push(*padded.last().unwrap());
            }
            next.push(hash_children(&padded));
        }
        levels.push(next.clone());
        current = next;
    }

    levels
}

/// Build binary Merkle tree levels for a chunk
fn build_chunk_levels(values: &[u64], chunk_offset: u64) -> Vec<Vec<[u8; 32]>> {
    let leaves: Vec<[u8; 32]> = values.iter().enumerate().map(|(idx, val)| {
        let mut hasher = Sha256::new();
        hasher.update(&chunk_offset.to_be_bytes());
        hasher.update(&(idx as u64).to_be_bytes());
        let val_bytes = (*val % MODULUS).to_be_bytes();
        let mut padded = [0u8; 32];
        padded[24..32].copy_from_slice(&val_bytes);
        hasher.update(&padded);
        hasher.finalize().into()
    }).collect();

    build_kary_levels(&leaves, 2)  // Binary tree within chunks
}

/// Generate k-ary multiproof for given indices
fn kary_multiproof(
    levels: &[Vec<[u8; 32]>],
    indices: &[usize],
    arity: usize,
) -> PyMerkleMultiProof {
    if levels.is_empty() {
        return PyMerkleMultiProof {
            tree_size: 0,
            arity,
            sibling_levels: vec![],
        };
    }

    let tree_size = levels[0].len();
    let mut sibling_levels: Vec<Vec<String>> = Vec::new();
    let mut current_indices: Vec<usize> = indices.iter().cloned().collect();
    current_indices.sort();
    current_indices.dedup();

    for level in levels.iter().take(levels.len().saturating_sub(1)) {
        let mut layer_siblings: Vec<String> = Vec::new();
        let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut next_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();

        for &idx in &current_indices {
            if used.contains(&idx) {
                continue;
            }
            let group_start = (idx / arity) * arity;
            for offset in 0..arity {
                let mut child = group_start + offset;
                if child >= level.len() {
                    child = level.len() - 1;
                }
                if child == idx {
                    continue;
                }
                if current_indices.contains(&child) {
                    continue;
                }
                if !used.contains(&child) {
                    layer_siblings.push(hex::encode(level[child]));
                }
            }
            for offset in 0..arity {
                let mut child = group_start + offset;
                if child >= level.len() {
                    child = level.len() - 1;
                }
                used.insert(child);
            }
            next_indices.insert(group_start / arity);
        }

        sibling_levels.push(layer_siblings);
        current_indices = next_indices.into_iter().collect();
        current_indices.sort();
    }

    PyMerkleMultiProof {
        tree_size,
        arity,
        sibling_levels,
    }
}

/// Open batch of indices from cached layer
fn open_batch_from_cache(layer: &LayerCache, indices: &[usize]) -> PyResult<PyBatchProof> {
    use std::collections::{HashMap, BTreeSet, BTreeMap};

    let chunk_len = layer.chunk_len;
    let total_len = layer.values.len();

    // Group indices by chunk
    let mut chunk_to_indices: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for &idx in indices {
        if idx >= total_len {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("index {} out of range (length {})", idx, total_len)
            ));
        }
        let chunk_idx = idx / chunk_len;
        chunk_to_indices.entry(chunk_idx).or_default().insert(idx);
    }

    let chunk_positions: Vec<usize> = chunk_to_indices.keys().cloned().collect();

    // Build entries and chunk leaf proofs
    let mut entries: Vec<PyBatchEntry> = Vec::new();
    let mut chunk_leaf_proofs: Vec<PyChunkLeafProof> = Vec::new();
    let mut chunk_levels_cache: HashMap<usize, Vec<Vec<[u8; 32]>>> = HashMap::new();

    for &chunk_idx in &chunk_positions {
        let chunk_offset = chunk_idx * chunk_len;
        let chunk_end = std::cmp::min(chunk_offset + chunk_len, total_len);
        let chunk_values = &layer.values[chunk_offset..chunk_end];

        // Build chunk levels (binary tree)
        let levels = build_chunk_levels(chunk_values, chunk_offset as u64);
        chunk_levels_cache.insert(chunk_idx, levels.clone());

        // Get local positions within this chunk
        let global_indices = chunk_to_indices.get(&chunk_idx).unwrap();
        let local_positions: Vec<usize> = global_indices
            .iter()
            .map(|&gidx| gidx - chunk_offset)
            .collect();

        // Build entries for this chunk
        for &gidx in global_indices {
            let local_idx = gidx - chunk_offset;
            entries.push(PyBatchEntry {
                index: gidx,
                value: layer.values[gidx] % MODULUS,  // Normalize to field
                chunk_index: chunk_idx,
                chunk_offset,
                leaf_pos: local_idx,
            });
        }

        // Build chunk leaf proof (binary multiproof)
        let leaf_proof = kary_multiproof(&levels, &local_positions, 2);
        chunk_leaf_proofs.push(PyChunkLeafProof {
            chunk_index: chunk_idx,
            chunk_offset,
            leaf_positions: local_positions,
            proof: leaf_proof,
        });
    }

    // Build global chunk proof (k-ary multiproof over chunk roots)
    let chunk_proof = kary_multiproof(
        &layer.chunk_root_levels,
        &chunk_positions,
        layer.chunk_tree_arity,
    );

    let chunk_roots: Vec<String> = chunk_positions
        .iter()
        .map(|&idx| hex::encode(layer.chunk_roots[idx]))
        .collect();

    Ok(PyBatchProof {
        entries,
        chunk_positions,
        chunk_roots,
        chunk_proof,
        chunk_leaf_proofs,
    })
}

/// Commit a codeword using chunked Merkle tree (same as STC row commitment)
fn commit_codeword_chunked(
    values: &[u64],
    chunk_len: usize,
    archive_dir: &Option<String>,
    archive_format: &str,
    chunk_tree_arity: usize,
) -> PyResult<PyFriCommitResult> {
    if values.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("empty codeword"));
    }

    // Calculate number of chunks (allow partial final chunk)
    let num_chunks = (values.len() + chunk_len - 1) / chunk_len;

    // 1) Compute chunk roots in parallel
    let chunk_roots: Vec<[u8; 32]> = (0..num_chunks).into_par_iter().map(|idx| {
        let start = idx * chunk_len;
        let end = std::cmp::min(start + chunk_len, values.len());
        let chunk = &values[start..end];
        let offset = start as u64;
        compute_chunk_root_fri(chunk, offset)
    }).collect();

    let chunk_roots_hex: Vec<String> = chunk_roots.iter()
        .map(|r| hex::encode(r))
        .collect();

    // 2) Archive chunks if requested
    if let Some(dir) = archive_dir {
        std::fs::create_dir_all(dir).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to create archive dir: {}", e))
        })?;

        if archive_format == "binary" {
            write_binary_archive(values, chunk_len, num_chunks, dir)?;
        } else {
            // JSON format (per-chunk files)
            for idx in 0..num_chunks {
                let start = idx * chunk_len;
                let end = std::cmp::min(start + chunk_len, values.len());
                let chunk = &values[start..end];
                let path = Path::new(dir).join(format!("chunk_{}.json", idx));
                let file = File::create(&path).map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!("Failed to create chunk file: {}", e))
                })?;
                serde_json::to_writer(file, &chunk).map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!("Failed to write chunk: {}", e))
                })?;
            }
        }
    }

    // 3) Compute global Merkle root (k-ary tree over chunk roots)
    let global_root = compute_kary_merkle_root(&chunk_roots, chunk_tree_arity);
    let root_hex = hex::encode(global_root);

    Ok(PyFriCommitResult {
        root_hex,
        length: values.len(),
        num_chunks,
        chunk_roots_hex,
        archive_dir: archive_dir.clone(),
        archive_format: archive_format.to_string(),
    })
}

/// Compute chunk root for FRI layer (matches STC chunk root format)
fn compute_chunk_root_fri(values: &[u64], chunk_offset: u64) -> [u8; 32] {
    let leaves: Vec<[u8; 32]> = values.iter().enumerate().map(|(idx, val)| {
        let mut hasher = Sha256::new();
        hasher.update(&chunk_offset.to_be_bytes());
        hasher.update(&(idx as u64).to_be_bytes());
        let val_bytes = (*val % MODULUS).to_be_bytes();
        let mut padded = [0u8; 32];
        padded[24..32].copy_from_slice(&val_bytes);
        hasher.update(&padded);
        hasher.finalize().into()
    }).collect();

    if leaves.is_empty() {
        return [0u8; 32];
    }

    compute_merkle_root(&leaves)
}

/// Write binary archive (chunks.bin + chunks.idx) matching Python BinaryChunkArchive
fn write_binary_archive(
    values: &[u64],
    chunk_len: usize,
    num_chunks: usize,
    dir: &str,
) -> PyResult<()> {
    let data_path = Path::new(dir).join("chunks.bin");
    let idx_path = Path::new(dir).join("chunks.idx");

    // Write data file
    let data_file = File::create(&data_path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to create chunks.bin: {}", e))
    })?;
    let mut data_writer = BufWriter::new(data_file);

    // Header: magic (4) + version (4) + num_chunks (8) = 16 bytes
    data_writer.write_all(BINARY_MAGIC)?;
    data_writer.write_all(&BINARY_VERSION.to_le_bytes())?;
    data_writer.write_all(&(num_chunks as u64).to_le_bytes())?;

    // Write index file
    let idx_file = File::create(&idx_path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to create chunks.idx: {}", e))
    })?;
    let mut idx_writer = BufWriter::new(idx_file);

    let mut current_offset: u64 = 16; // After header

    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * chunk_len;
        let end = std::cmp::min(start + chunk_len, values.len());
        let chunk = &values[start..end];
        let chunk_len_actual = chunk.len();

        // Write index entry: offset (8) + length (4)
        idx_writer.write_all(&current_offset.to_le_bytes())?;
        idx_writer.write_all(&(chunk_len_actual as u32).to_le_bytes())?;

        // Write chunk data (u64 little-endian)
        for &val in chunk {
            data_writer.write_all(&val.to_le_bytes())?;
        }

        current_offset += (chunk_len_actual * 8) as u64;
    }

    data_writer.flush()?;
    idx_writer.flush()?;

    // Update header with final chunk count (already correct, but let's be safe)
    // Actually we wrote it correctly, so no need to seek back

    Ok(())
}
