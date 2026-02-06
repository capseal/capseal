use pyo3::prelude::*;
use sha2::{Sha256, Digest};
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;

mod fri;
pub mod capseal_core;

const MODULUS: u64 = (1 << 61) - 1;

fn hash_pair(left: &[u8], right: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

fn compute_chunk_root(values: &[u64], chunk_offset: u64) -> [u8; 32] {
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

    let mut current = leaves;
    while current.len() > 1 {
        let mut next = Vec::with_capacity((current.len() + 1) / 2);
        for i in (0..current.len()).step_by(2) {
            let left = current[i];
            let right = if i + 1 < current.len() { current[i + 1] } else { left };
            next.push(hash_pair(&left, &right));
        }
        current = next;
    }
    current[0]
}

#[pyclass]
#[derive(Clone)]
struct PyChunkRecord {
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    offset: usize,
    #[pyo3(get)]
    root: String,
    #[pyo3(get)]
    archive_handle: String,
}

#[pyclass]
struct PyCommitmentResult {
    #[pyo3(get)]
    chunks: Vec<PyChunkRecord>,
    #[pyo3(get)]
    global_root: String,
    #[pyo3(get)]
    s_hex: Vec<String>,
    #[pyo3(get)]
    pow_hex: Vec<String>,
}

mod gpu_shim {
    #[cfg(feature = "gpu")]
    extern "C" {
        fn gpu_compute_sketches(
            input_ptr: *const u64,
            input_len: usize,
            challenges_ptr: *const u64,
            num_challenges: usize,
            output_sketches: *mut u64
        ) -> i32;

        fn gpu_fold_fri(
            input_ptr: *const u64,
            input_len: usize,
            alpha: u64,
            output_ptr: *mut u64
        ) -> i32;
    }

    #[cfg(feature = "gpu")]
    pub fn compute_on_gpu(data: &[u64], challenges: &[u64]) -> Option<Vec<u64>> {
        println!("🚀 [bef_rust] Offloading sketch computation to GPU ({} elements)...", data.len());
        let mut out = vec![0u64; challenges.len()];
        unsafe {
            let res = gpu_compute_sketches(
                data.as_ptr(),
                data.len(),
                challenges.as_ptr(),
                challenges.len(),
                out.as_mut_ptr()
            );
            if res == 0 {
                Some(out)
            } else {
                eprintln!("GPU computation failed with code {}", res);
                None
            }
        }
    }

    #[cfg(feature = "gpu")]
    pub fn fold_on_gpu(data: &[u64], alpha: u64) -> Option<Vec<u64>> {
        println!("🚀 [bef_rust] Offloading FRI fold to GPU ({} elements)...", data.len());
        let next_n = data.len() / 2;
        let mut out = vec![0u64; next_n];
        unsafe {
            let res = gpu_fold_fri(
                data.as_ptr(),
                data.len(),
                alpha,
                out.as_mut_ptr()
            );
            if res == 0 {
                Some(out)
            } else {
                eprintln!("GPU FRI fold failed with code {}", res);
                None
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    pub fn compute_on_gpu(_data: &[u64], _challenges: &[u64]) -> Option<Vec<u64>> {
        None
    }

    #[cfg(not(feature = "gpu"))]
    pub fn fold_on_gpu(_data: &[u64], _alpha: u64) -> Option<Vec<u64>> {
        None
    }
}

#[pyclass]
struct PyStcParams {
    challenges: Vec<u64>,
    _chunk_len: usize,
}

#[pymethods]
impl PyStcParams {
    #[new]
    fn new(challenges: Vec<u64>, chunk_len: usize) -> Self {
        PyStcParams {
            challenges,
            _chunk_len: chunk_len,
        }
    }
}

#[derive(Clone)]
struct State {
    n: u64,
    root: [u8; 32],
    s: Vec<u64>,
    pow: Vec<u64>,
}

#[pyclass]
struct PyStcState {
    inner: State,
}

#[pymethods]
impl PyStcState {
    #[new]
    fn new(challenges_len: usize) -> Self {
        PyStcState {
            inner: State {
                n: 0,
                root: [0; 32],
                s: vec![0; challenges_len],
                pow: vec![1; challenges_len],
            },
        }
    }

    fn update(&mut self, params: &PyStcParams, chunk_values: Vec<u64>) {
        const GPU_THRESHOLD: usize = 1024 * 10; 
        let use_gpu = chunk_values.len() >= GPU_THRESHOLD;
        
        if use_gpu {
            if let Some(sketches) = gpu_shim::compute_on_gpu(&chunk_values, &params.challenges) {
                let mod_u128 = MODULUS as u128;
                for (idx, &sketch) in sketches.iter().enumerate() {
                    let pow_r = self.inner.pow[idx] as u128;
                    let term = (pow_r * (sketch as u128)) % mod_u128;
                    self.inner.s[idx] = ((self.inner.s[idx] as u128 + term) % mod_u128) as u64;
                    
                    let r = params.challenges[idx] as u128;
                    let mut step = 1u128;
                    let mut base = r;
                    let mut exp = chunk_values.len() as u64;
                    while exp > 0 {
                        if exp % 2 == 1 {
                            step = (step * base) % mod_u128;
                        }
                        base = (base * base) % mod_u128;
                        exp /= 2;
                    }
                    self.inner.pow[idx] = ((self.inner.pow[idx] as u128 * step) % mod_u128) as u64;
                }
                self.inner.n += chunk_values.len() as u64;
                return;
            }
        }

        let count = chunk_values.len();
        let mod_u128 = MODULUS as u128;
        for (idx, r) in params.challenges.iter().enumerate() {
            let mut acc: u128 = 0;
            let mut pow_r: u128 = self.inner.pow[idx] as u128;
            let r_u128 = *r as u128;
            for &val in &chunk_values {
                let term = (pow_r * (val as u128)) % mod_u128;
                acc = (acc + term) % mod_u128;
                pow_r = (pow_r * r_u128) % mod_u128;
            }
            self.inner.s[idx] = ((self.inner.s[idx] as u128 + acc) % mod_u128) as u64;
            self.inner.pow[idx] = pow_r as u64;
        }
        self.inner.n += count as u64;
    }

    fn root_hex(&self) -> String {
        hex::encode(self.inner.root)
    }

    fn get_n(&self) -> String {
        self.inner.n.to_string()
    }

    fn get_s_hex(&self) -> Vec<String> {
        self.inner.s.iter().map(|v| format!("{:x}", v)).collect()
    }

    fn get_pow_hex(&self) -> Vec<String> {
        self.inner.pow.iter().map(|v| format!("{:x}", v)).collect()
    }
}

#[pyfunction]
#[pyo3(signature = (params, flat_values, row_width, archive_dir=None))]
fn commit_trace_batch(
    params: &PyStcParams,
    flat_values: Vec<u64>,
    row_width: usize,
    archive_dir: Option<String>,
) -> PyResult<PyCommitmentResult> {
    if flat_values.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("empty trace"));
    }
    let num_chunks = flat_values.len() / row_width;
    if flat_values.len() % row_width != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("trace length not multiple of row width"));
    }

    // 1. Parallel Processing of Chunks (Merkle hashing)
    let records: Vec<PyChunkRecord> = (0..num_chunks).into_par_iter().map(|idx| {
        let start = idx * row_width;
        let end = start + row_width;
        let chunk = &flat_values[start..end];
        let offset = start as u64;
        let root = compute_chunk_root(chunk, offset);
        let root_hex = hex::encode(root);

        PyChunkRecord {
            index: idx,
            offset: start,
            root: root_hex,
            archive_handle: format!("chunk_{}.json", idx),
        }
    }).collect();

    // 2. Archive (IO) - Sequential to avoid FD pressure
    if let Some(ref dir) = archive_dir {
        for r in &records {
            let start = r.index * row_width;
            let end = start + row_width;
            let chunk = &flat_values[start..end];
            let path = Path::new(dir).join(&r.archive_handle);
            let file = File::create(&path).expect("failed to create chunk file");
            serde_json::to_writer(file, &chunk).expect("failed to write chunk file");
        }
    }

    // 3. Compute Sketches
    let mut state = PyStcState::new(params.challenges.len());
    state.update(params, flat_values);

    // 4. Compute Global Merkle Root (Binary Tree)
    let chunk_roots_bin: Vec<[u8; 32]> = records.iter().map(|r| {
        let b = hex::decode(&r.root).expect("failed to decode hex root");
        let mut out = [0u8; 32];
        out.copy_from_slice(&b);
        out
    }).collect();
    
    let mut current = chunk_roots_bin;
    while current.len() > 1 {
        let mut next = Vec::with_capacity((current.len() + 1) / 2);
        for i in (0..current.len()).step_by(2) {
            let left = current[i];
            let right = if i + 1 < current.len() { current[i + 1] } else { left };
            next.push(hash_pair(&left, &right));
        }
        current = next;
    }
    let global_root = if current.is_empty() { hex::encode([0u8; 32]) } else { hex::encode(current[0]) };

    Ok(PyCommitmentResult {
        chunks: records,
        global_root,
        s_hex: state.get_s_hex(),
        pow_hex: state.get_pow_hex(),
    })
}

#[pymodule]
fn bef_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStcParams>()?;
    m.add_class::<PyStcState>()?;
    m.add_class::<PyChunkRecord>()?;
    m.add_class::<PyCommitmentResult>()?;
    m.add_function(wrap_pyfunction!(commit_trace_batch, m)?)?;
    m.add_function(wrap_pyfunction!(fri::fold_fri_layer, m)?)?;
    // FRI stateful prover (Milestone 1)
    m.add_class::<fri::PyFriState>()?;
    m.add_class::<fri::PyFriCommitResult>()?;
    // FRI Milestone 2: batch opening proof types
    m.add_class::<fri::PyBatchProof>()?;
    m.add_class::<fri::PyBatchEntry>()?;
    m.add_class::<fri::PyChunkLeafProof>()?;
    m.add_class::<fri::PyMerkleMultiProof>()?;
    // CapSeal Core (high-level API matching contracts.ts)
    capseal_core::register_module(m)?;
    Ok(())
}