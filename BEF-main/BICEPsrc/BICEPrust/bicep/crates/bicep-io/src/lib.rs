use bicep_core::{State, Time, F, Calc};
use bicep_sampler::{Ensemble, Path, PathSpec};
use arrow::array::{Array, Float64Array, UInt64Array, UInt32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::sync::Arc;
use uuid::Uuid;

pub mod cli;
pub use cli::*;

/// Run manifest for complete reproducibility
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunManifest {
    pub run_id: String,
    pub timestamp: String,
    pub seed: u64,
    pub calc: String,         // "ito" | "stratonovich"
    pub integrator: String,   // "euler_maruyama" | "milstein" | "heun_stratonovich"
    pub dt: F,
    pub model_name: String,
    pub model_params: serde_json::Value,
    pub n_paths: usize,
    pub n_steps: usize,
    pub save_stride: usize,
    pub total_time: F,
    pub commit_hash: Option<String>,
    pub rust_version: String,
}

/// Parquet schema for path data
pub struct ParquetWriter {
    writer: ArrowWriter<File>,
    schema: Arc<Schema>,
    state_dim: usize,
}

/// Single row in the path table
#[derive(Clone, Debug)]
pub struct PathRow {
    pub run_id: String,
    pub path_id: u64,
    pub step: u32,
    pub time: F,
    pub state: Vec<F>,
    pub hit_set: Option<String>,
}

impl RunManifest {
    pub fn new(
        seed: u64,
        calc: Calc,
        integrator: &str,
        dt: F,
        model_name: &str,
        model_params: serde_json::Value,
        spec: &PathSpec,
    ) -> Self {
        let run_id = Uuid::new_v4().to_string();
        let timestamp = chrono::Utc::now().to_rfc3339();
        let calc_str = match calc {
            Calc::Ito => "ito",
            Calc::Stratonovich => "stratonovich",
        };
        
        Self {
            run_id,
            timestamp,
            seed,
            calc: calc_str.to_string(),
            integrator: integrator.to_string(),
            dt,
            model_name: model_name.to_string(),
            model_params,
            n_paths: 0,  // Will be set when writing
            n_steps: spec.n_steps,
            save_stride: spec.save_stride,
            total_time: spec.total_time(),
            commit_hash: get_git_commit(),
            rust_version: get_rust_version(),
        }
    }
    
    pub fn save_to_file(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
    
    pub fn load_from_file(path: &str) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let manifest = serde_json::from_str(&json)?;
        Ok(manifest)
    }
}

impl ParquetWriter {
    pub fn new(file_path: &str, state_dim: usize) -> anyhow::Result<Self> {
        let file = File::create(file_path)?;
        
        // Build schema: (run_id, path_id, step, time, x0, x1, ..., hit_set)
        let mut fields = vec![
            Field::new("run_id", DataType::Utf8, false),
            Field::new("path_id", DataType::UInt64, false),
            Field::new("step", DataType::UInt32, false),
            Field::new("time", DataType::Float64, false),
        ];
        
        // Add state dimensions
        for i in 0..state_dim {
            fields.push(Field::new(&format!("x{}", i), DataType::Float64, false));
        }
        
        fields.push(Field::new("hit_set", DataType::Utf8, true));  // Optional
        
        let schema = Arc::new(Schema::new(fields));
        let writer = ArrowWriter::try_new(file, schema.clone(), None)?;
        
        Ok(Self {
            writer,
            schema,
            state_dim,
        })
    }
    
    pub fn write_ensemble(
        &mut self, 
        ensemble: &Ensemble,
        manifest: &RunManifest
    ) -> anyhow::Result<()> {
        let mut rows = Vec::new();
        
        // Extract all path data into rows
        for (path_id, path) in ensemble.paths.iter().enumerate() {
            for (step_idx, (time, state)) in path.times.iter().zip(path.states.iter()).enumerate() {
                rows.push(PathRow {
                    run_id: manifest.run_id.clone(),
                    path_id: path_id as u64,
                    step: step_idx as u32,
                    time: *time,
                    state: state.0.data.as_vec().clone(),
                    hit_set: path.hit_set.clone(),
                });
            }
        }
        
        if rows.is_empty() {
            return Ok(());
        }
        
        // Build Arrow arrays
        let run_ids: Vec<String> = rows.iter().map(|r| r.run_id.clone()).collect();
        let path_ids: Vec<u64> = rows.iter().map(|r| r.path_id).collect();
        let steps: Vec<u32> = rows.iter().map(|r| r.step).collect();
        let times: Vec<f64> = rows.iter().map(|r| r.time).collect();
        
        let mut arrays: Vec<Arc<dyn Array>> = vec![
            Arc::new(StringArray::from(run_ids)),
            Arc::new(UInt64Array::from(path_ids)),
            Arc::new(UInt32Array::from(steps)),
            Arc::new(Float64Array::from(times)),
        ];
        
        // Add state dimension arrays
        for i in 0..self.state_dim {
            let state_vals: Vec<f64> = rows.iter()
                .map(|r| r.state.get(i).copied().unwrap_or(0.0))
                .collect();
            arrays.push(Arc::new(Float64Array::from(state_vals)));
        }
        
        // Add hit_set array
        let hit_sets: Vec<Option<String>> = rows.iter().map(|r| r.hit_set.clone()).collect();
        arrays.push(Arc::new(StringArray::from(hit_sets)));
        
        // Create record batch and write
        let batch = RecordBatch::try_new(self.schema.clone(), arrays)?;
        
        self.writer.write(&batch)?;
        Ok(())
    }
    
    pub fn close(self) -> anyhow::Result<()> {
        self.writer.close()?;
        Ok(())
    }
}

/// Write ensemble to Parquet with manifest
pub fn write_ensemble_with_manifest(
    ensemble: &Ensemble,
    manifest: &RunManifest,
    parquet_path: &str,
    manifest_path: &str,
) -> anyhow::Result<()> {
    // Determine state dimension from first path
    let state_dim = if let Some(path) = ensemble.paths.first() {
        if let Some(state) = path.states.first() {
            state.dim()
        } else {
            1
        }
    } else {
        1
    };
    
    // Write Parquet
    let mut writer = ParquetWriter::new(parquet_path, state_dim)?;
    writer.write_ensemble(ensemble, manifest)?;
    writer.close()?;
    
    // Write manifest
    let mut manifest_with_paths = manifest.clone();
    manifest_with_paths.n_paths = ensemble.n_paths();
    manifest_with_paths.save_to_file(manifest_path)?;
    
    println!("Wrote {} paths to {}", ensemble.n_paths(), parquet_path);
    println!("Wrote manifest to {}", manifest_path);
    
    Ok(())
}

/// Get git commit hash for reproducibility

fn get_git_commit() -> Option<String> {
    std::process::Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
}

fn get_rust_version() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}