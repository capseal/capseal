use bicep_core::{State, Time, F, Drift, Diffusion, SdeIntegrator, Calc, NoiseGenerator};
use bicep_sampler::{PathSpec, Path, Ensemble, Boundary, Stopping, apply_boundary};
use nalgebra::{DVector, DMatrix};
use rayon::prelude::*;
use wide::f64x4;

/// SIMD-optimized batch sampler for CPU
pub struct CpuBatchSampler<I, D, S>
where
    I: SdeIntegrator,
    D: Drift,
    S: Diffusion,
{
    pub integrator: I,
    pub drift: D,
    pub diffusion: S,
}

impl<I, D, S> CpuBatchSampler<I, D, S>
where
    I: SdeIntegrator + Clone + Send + Sync,
    D: Drift + Clone + Send + Sync,
    S: Diffusion + Clone + Send + Sync,
{
    pub fn new(integrator: I, drift: D, diffusion: S) -> Self {
        Self { integrator, drift, diffusion }
    }
    
    /// Run paths with adaptive batching and SIMD optimization
    pub fn run_paths_optimized(
        &self,
        calc: Calc,
        spec: &PathSpec,
        x0s: &[State],
        boundary: &Boundary,
        stopping: &Stopping,
        global_seed: u64,
    ) -> Ensemble {
        let n_cores = rayon::current_num_threads();
        let batch_size = (x0s.len() + n_cores - 1) / n_cores;
        
        // For small problems or non-uniform models, fall back to standard approach
        if x0s.len() < 64 || !self.supports_simd_batch() {
            return self.run_paths_standard(calc, spec, x0s, boundary, stopping, global_seed);
        }
        
        // SIMD batch processing for large uniform problems
        self.run_paths_simd_batched(calc, spec, x0s, boundary, stopping, global_seed, batch_size)
    }
    
    /// Standard path execution (fallback)
    fn run_paths_standard(
        &self,
        calc: Calc,
        spec: &PathSpec,
        x0s: &[State],
        boundary: &Boundary,
        stopping: &Stopping,
        global_seed: u64,
    ) -> Ensemble {
        let paths: Vec<Path> = x0s
            .par_iter()
            .enumerate()
            .map(|(path_id, x0)| {
                let mut rng = NoiseGenerator::from_path_id(global_seed, path_id as u64);
                self.run_single_path(calc, spec, x0.clone(), boundary, stopping, &mut rng)
            })
            .collect();
        
        Ensemble {
            paths,
            spec: spec.clone(),
        }
    }
    
    /// SIMD-batched path execution for uniform models
    fn run_paths_simd_batched(
        &self,
        calc: Calc,
        spec: &PathSpec,
        x0s: &[State],
        boundary: &Boundary,
        stopping: &Stopping,
        global_seed: u64,
        batch_size: usize,
    ) -> Ensemble {
        let paths: Vec<Path> = x0s
            .par_chunks(batch_size)
            .enumerate()
            .flat_map(|(chunk_idx, chunk)| {
                let chunk_offset = chunk_idx * batch_size;
                self.run_simd_chunk(calc, spec, chunk, boundary, stopping, global_seed, chunk_offset)
            })
            .collect();
        
        Ensemble {
            paths,
            spec: spec.clone(),
        }
    }
    
    /// Process a chunk of paths with SIMD vectorization
    fn run_simd_chunk(
        &self,
        calc: Calc,
        spec: &PathSpec,
        x0s: &[State],
        boundary: &Boundary,
        stopping: &Stopping,
        global_seed: u64,
        chunk_offset: usize,
    ) -> Vec<Path> {
        let dim = if let Some(x0) = x0s.first() { x0.dim() } else { return vec![]; };
        
        // For 1D problems, use f64x4 SIMD (4 paths at once)
        if dim == 1 && x0s.len() >= 4 {
            self.run_simd_1d_batch(calc, spec, x0s, boundary, stopping, global_seed, chunk_offset)
        } else {
            // Fall back to individual path processing for higher dimensions
            x0s.iter()
                .enumerate()
                .map(|(local_idx, x0)| {
                    let path_id = chunk_offset + local_idx;
                    let mut rng = NoiseGenerator::from_path_id(global_seed, path_id as u64);
                    self.run_single_path(calc, spec, x0.clone(), boundary, stopping, &mut rng)
                })
                .collect()
        }
    }
    
    /// SIMD-optimized 1D path processing (4 paths at once)
    fn run_simd_1d_batch(
        &self,
        calc: Calc,
        spec: &PathSpec,
        x0s: &[State],
        boundary: &Boundary,
        stopping: &Stopping,
        global_seed: u64,
        chunk_offset: usize,
    ) -> Vec<Path> {
        let mut paths = Vec::new();
        
        // Process 4 paths at a time with SIMD
        for batch_start in (0..x0s.len()).step_by(4) {
            let batch_end = (batch_start + 4).min(x0s.len());
            let batch_size = batch_end - batch_start;
            
            if batch_size == 4 {
                // Full SIMD batch
                let simd_paths = self.run_simd_1d_quartet(
                    calc, spec, &x0s[batch_start..batch_end], 
                    boundary, stopping, global_seed, chunk_offset + batch_start
                );
                paths.extend(simd_paths);
            } else {
                // Partial batch - fall back to scalar
                for (local_idx, x0) in x0s[batch_start..batch_end].iter().enumerate() {
                    let path_id = chunk_offset + batch_start + local_idx;
                    let mut rng = NoiseGenerator::from_path_id(global_seed, path_id as u64);
                    paths.push(self.run_single_path(calc, spec, x0.clone(), boundary, stopping, &mut rng));
                }
            }
        }
        
        paths
    }
    
    /// Run exactly 4 1D paths with f64x4 SIMD
    fn run_simd_1d_quartet(
        &self,
        calc: Calc,
        spec: &PathSpec,
        x0s: &[State],
        boundary: &Boundary,
        stopping: &Stopping,
        global_seed: u64,
        chunk_offset: usize,
    ) -> Vec<Path> {
        assert_eq!(x0s.len(), 4, "SIMD quartet requires exactly 4 states");
        assert!(x0s.iter().all(|x| x.dim() == 1), "SIMD quartet requires 1D states");
        
        // Initialize SIMD state vector
        let mut x_simd = f64x4::new([
            x0s[0].0[0], x0s[1].0[0], x0s[2].0[0], x0s[3].0[0]
        ]);
        
        let mut t = 0.0;
        let sqrt_dt = spec.dt.sqrt();
        
        // Initialize RNGs for each path
        let mut rngs: Vec<_> = (0..4)
            .map(|i| NoiseGenerator::from_path_id(global_seed, (chunk_offset + i) as u64))
            .collect();
        
        // Initialize paths
        let mut paths: Vec<Path> = (0..4).map(|i| Path {
            times: Vec::new(),
            states: Vec::new(),
            stopped: false,
            stop_time: None,
            hit_set: None,
        }).collect();
        
        let noise_dim = 1; // For 1D problems
        
        for step in 0..spec.n_steps {
            // Save states if on stride boundary
            if step % spec.save_stride == 0 {
                let x_array = x_simd.to_array();
                for i in 0..4 {
                    paths[i].times.push(t);
                    paths[i].states.push(State::new(vec![x_array[i]]));
                }
            }
            
            // Check stopping conditions (scalar for now - could be SIMD-ized too)
            let mut active_mask = [true; 4];
            for i in 0..4 {
                if paths[i].stopped {
                    active_mask[i] = false;
                    continue;
                }
                
                let state_i = State::new(vec![x_simd.to_array()[i]]);
                
                if let Some(ref hit_fn) = stopping.first_hit {
                    if let Some(hit_label) = hit_fn(&state_i) {
                        paths[i].stopped = true;
                        paths[i].stop_time = Some(t);
                        paths[i].hit_set = Some(hit_label);
                        active_mask[i] = false;
                        continue;
                    }
                }
                
                if let Some(max_time) = stopping.max_time {
                    if t >= max_time {
                        paths[i].stopped = true;
                        paths[i].stop_time = Some(t);
                        active_mask[i] = false;
                        continue;
                    }
                }
            }
            
            // Early exit if all paths stopped
            if active_mask.iter().all(|&x| !x) {
                break;
            }
            
            // Generate SIMD noise
            let dw_array: [f64; 4] = core::array::from_fn(|i| {
                if active_mask[i] {
                    rngs[i].generate_dw(noise_dim, sqrt_dt).0[0]
                } else {
                    0.0
                }
            });
            let dw_simd = f64x4::from(dw_array);
            
            // SIMD integration step for active paths
            if active_mask.iter().any(|&x| x) {
                x_simd = self.simd_integration_step_1d(calc, t, x_simd, spec.dt, dw_simd, &active_mask);
            }
            
            t += spec.dt;
        }
        
        // Save final states
        let final_x = x_simd.to_array();
        for i in 0..4 {
            if paths[i].times.is_empty() || paths[i].times.last() != Some(&t) {
                paths[i].times.push(t);
                paths[i].states.push(State::new(vec![final_x[i]]));
            }
        }
        
        paths
    }
    
    /// SIMD-optimized integration step for 1D problems
    fn simd_integration_step_1d(
        &self,
        calc: Calc,
        t: Time,
        x_simd: f64x4,
        dt: F,
        dw_simd: f64x4,
        active_mask: &[bool; 4],
    ) -> f64x4 {
        // For simplicity, convert to scalar, compute, then back to SIMD
        // A full SIMD implementation would require SIMD-aware Drift/Diffusion traits
        let x_array = x_simd.to_array();
        let dw_array = dw_simd.to_array();
        
        let mut result = [0.0; 4];
        
        for i in 0..4 {
            if active_mask[i] {
                let state_i = State::new(vec![x_array[i]]);
                let dw_i = State::new(vec![dw_array[i]]);
                
                let next_state = self.integrator.step(
                    calc, t, &state_i, dt, &dw_i, &self.drift, &self.diffusion
                );
                
                result[i] = next_state.0[0];
            } else {
                result[i] = x_array[i]; // Keep inactive paths unchanged
            }
        }
        
        f64x4::from(result)
    }
    
    /// Check if this model combination supports SIMD batching
    fn supports_simd_batch(&self) -> bool {
        // For now, conservative check - could be expanded based on model traits
        true
    }
    
    /// Single path execution (standard algorithm)
    fn run_single_path(
        &self,
        calc: Calc,
        spec: &PathSpec,
        mut x: State,
        boundary: &Boundary,
        stopping: &Stopping,
        rng: &mut NoiseGenerator,
    ) -> Path {
        let mut path = Path {
            times: Vec::new(),
            states: Vec::new(),
            stopped: false,
            stop_time: None,
            hit_set: None,
        };
        
        let mut t = 0.0;
        let noise_dim = self.diffusion.noise_dim(t, &x);
        let sqrt_dt = spec.dt.sqrt();
        
        for step in 0..spec.n_steps {
            if step % spec.save_stride == 0 {
                path.times.push(t);
                path.states.push(x.clone());
            }
            
            // Check stopping conditions
            if let Some(ref hit_fn) = stopping.first_hit {
                if let Some(hit_label) = hit_fn(&x) {
                    path.stopped = true;
                    path.stop_time = Some(t);
                    path.hit_set = Some(hit_label);
                    break;
                }
            }
            
            if let Some(max_time) = stopping.max_time {
                if t >= max_time {
                    path.stopped = true;
                    path.stop_time = Some(t);
                    break;
                }
            }
            
            let dw = rng.generate_dw(noise_dim, sqrt_dt);
            x = self.integrator.step(calc, t, &x, spec.dt, &dw, &self.drift, &self.diffusion);
            x = apply_boundary(boundary, x);
            
            t += spec.dt;
        }
        
        if path.times.is_empty() || path.times.last() != Some(&t) {
            path.times.push(t);
            path.states.push(x);
        }
        
        path
    }
}

/// Memory-efficient batch matrix operations
pub struct BatchMatrixOps;

impl BatchMatrixOps {
    /// Batch matrix-vector multiplication for drift computation
    pub fn batch_matvec_f64x4(
        matrices: &[DMatrix<F>], 
        vectors: &f64x4
    ) -> Vec<f64x4> {
        matrices.iter().map(|mat| {
            let v_array = vectors.to_array();
            let mut result = [0.0; 4];
            
            for i in 0..4 {
                // Assuming 1D for simplicity - extend for higher dimensions
                result[i] = mat[(0, 0)] * v_array[i];
            }
            
            f64x4::from(result)
        }).collect()
    }
    
    /// Vectorized noise generation for batch processing
    pub fn generate_batch_noise_1d(
        rngs: &mut [NoiseGenerator],
        sqrt_dt: F,
        count: usize,
    ) -> Vec<f64x4> {
        let mut batches = Vec::new();
        
        for batch_start in (0..count).step_by(4) {
            let batch_end = (batch_start + 4).min(count);
            
            if batch_end - batch_start == 4 {
                let noise_array: [f64; 4] = core::array::from_fn(|i| {
                    rngs[batch_start + i].generate_dw(1, sqrt_dt).0[0]
                });
                batches.push(f64x4::from(noise_array));
            }
        }
        
        batches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bicep_core::{EulerMaruyama, Calc};
    use bicep_models::BrownianMotion;
    
    #[test]
    fn test_cpu_batch_sampler() {
        let model = BrownianMotion::new(1.0);
        let sampler = CpuBatchSampler::new(EulerMaruyama, model.clone(), model);
        
        let spec = PathSpec::new(100, 0.01, 1);
        let x0s = vec![State::new(vec![0.0]); 8];
        let boundary = Boundary::None;
        let stopping = bicep_sampler::Stopping::default();
        
        let ensemble = sampler.run_paths_optimized(
            Calc::Ito, &spec, &x0s, &boundary, &stopping, 42
        );
        
        assert_eq!(ensemble.paths.len(), 8);
        assert!(ensemble.paths.iter().all(|p| p.states.len() > 0));
    }
    
    #[test]
    fn test_simd_1d_quartet() {
        let model = BrownianMotion::new(1.0);
        let sampler = CpuBatchSampler::new(EulerMaruyama, model.clone(), model);
        
        let spec = PathSpec::new(10, 0.1, 1);
        let x0s = vec![
            State::new(vec![0.0]),
            State::new(vec![1.0]), 
            State::new(vec![-1.0]),
            State::new(vec![0.5])
        ];
        let boundary = Boundary::None;
        let stopping = bicep_sampler::Stopping::default();
        
        let paths = sampler.run_simd_1d_quartet(
            Calc::Ito, &spec, &x0s, &boundary, &stopping, 42, 0
        );
        
        assert_eq!(paths.len(), 4);
        for path in &paths {
            assert!(path.states.len() > 0);
            assert_eq!(path.states[0].dim(), 1);
        }
    }
}