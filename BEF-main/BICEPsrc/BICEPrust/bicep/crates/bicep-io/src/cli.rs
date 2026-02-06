use crate::{RunManifest, write_ensemble_with_manifest};
use bicep_core::{State, Calc, EulerMaruyama, Milstein, HeunStratonovich, NoiseConfig, ShockType};
use bicep_models::{BrownianMotion, GeometricBrownianMotion, OrnsteinUhlenbeck, DoubleWell};
use bicep_sampler::{Sampler, PathSpec, Boundary, Stopping};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::json;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "bicep")]
#[command(about = "BICEP - Brownian-Inspired Computation Engine for Paths")]
#[command(long_about = "High-performance SDE simulation with Itô/Stratonovich calculus support")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Sample SDE paths and write to Parquet
    Sample {
        /// Model type
        #[arg(long, value_enum)]
        model: ModelType,
        
        /// Calculation method
        #[arg(long, value_enum, default_value = "ito")]
        calc: CalcType,
        
        /// Integrator type
        #[arg(long, value_enum, default_value = "euler-maruyama")]
        integrator: IntegratorType,
        
        /// Time step size
        #[arg(long)]
        dt: f64,
        
        /// Number of time steps
        #[arg(long)]
        steps: usize,
        
        /// Number of paths to simulate
        #[arg(long)]
        paths: usize,
        
        /// Save every nth step (default: save all)
        #[arg(long, default_value = "1")]
        save_stride: usize,
        
        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
        
        /// Output Parquet file
        #[arg(long)]
        out: PathBuf,
        
        /// Model-specific parameters (JSON)
        #[arg(long)]
        params: Option<String>,
    },
}

#[derive(Clone, Debug, ValueEnum)]
pub enum ModelType {
    #[value(name = "brownian")]
    Brownian,
    #[value(name = "gbm")]
    GeometricBrownianMotion,
    #[value(name = "ou")]
    OrnsteinUhlenbeck,
    #[value(name = "double-well")]
    DoubleWell,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum CalcType {
    #[value(name = "ito")]
    Ito,
    #[value(name = "stratonovich")]
    Stratonovich,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum IntegratorType {
    #[value(name = "euler-maruyama")]
    EulerMaruyama,
    #[value(name = "milstein")]
    Milstein,
    #[value(name = "heun-stratonovich")]
    HeunStratonovich,
}

impl From<CalcType> for Calc {
    fn from(calc_type: CalcType) -> Self {
        match calc_type {
            CalcType::Ito => Calc::Ito,
            CalcType::Stratonovich => Calc::Stratonovich,
        }
    }
}

pub async fn run_sample_command(
    model: ModelType,
    calc: CalcType,
    integrator: IntegratorType,
    dt: f64,
    steps: usize,
    paths: usize,
    save_stride: usize,
    seed: u64,
    out: PathBuf,
    params: Option<String>,
) -> anyhow::Result<()> {
    println!("BICEP Sampling");
    println!("==============");
    println!("Model: {:?}", model);
    println!("Calculus: {:?}", calc);
    println!("Integrator: {:?}", integrator);
    println!("dt: {:.6}", dt);
    println!("Steps: {}", steps);
    println!("Paths: {}", paths);
    println!("Save stride: {}", save_stride);
    println!("Seed: {}", seed);
    println!("Output: {:?}", out);
    
    // Parse model parameters
    let model_params = if let Some(params_str) = params {
        serde_json::from_str(&params_str)?
    } else {
        json!({})
    };

    // Phase 1/2 modelling controls (recognized but not all used yet in core engine)
    let returns_space = model_params.get("returns_space").and_then(|v| v.as_bool());
    let ewma_alpha = model_params.get("ewma_alpha").and_then(|v| v.as_f64());
    let shock_type = model_params.get("shock_type").and_then(|v| v.as_str()).unwrap_or("gaussian");
    let nu = model_params.get("nu").and_then(|v| v.as_f64());
    let bootstrap_block = model_params.get("bootstrap_block").and_then(|v| v.as_u64()).map(|u| u as usize);
    let regime_percentile = model_params.get("regime_percentile").and_then(|v| v.as_f64());

    println!("\nModelling controls (requested):");
    println!("  returns_space:    {:?}", returns_space);
    println!("  ewma_alpha:       {:?}", ewma_alpha);
    println!("  shock_type:       {}", shock_type);
    println!("  nu:               {:?}", nu);
    println!("  bootstrap_block:  {:?}", bootstrap_block);
    println!("  regime_percentile:{:?}", regime_percentile);
    println!("  note: Student‑t noise and EWMA scaling are active; block bootstrap/regime split are reserved for future patches.");
    
    // Create path specification
    let spec = PathSpec::new(steps, dt, save_stride);
    
    // Create manifest
    let calc_enum = Calc::from(calc);
    let integrator_str = match integrator {
        IntegratorType::EulerMaruyama => "euler_maruyama",
        IntegratorType::Milstein => "milstein", 
        IntegratorType::HeunStratonovich => "heun_stratonovich",
    };
    
    let model_name = match model {
        ModelType::Brownian => "brownian",
        ModelType::GeometricBrownianMotion => "gbm",
        ModelType::OrnsteinUhlenbeck => "ou",
        ModelType::DoubleWell => "double_well",
    };
    
    let manifest = RunManifest::new(
        seed,
        calc_enum,
        integrator_str,
        dt,
        model_name,
        model_params.clone(),
        &spec,
    );
    
    // Build optional noise config
    let noise_cfg = {
        let shock = match shock_type.to_lowercase().as_str() {
            "student_t" | "student-t" | "studentt" => ShockType::StudentT,
            _ => ShockType::Gaussian,
        };
        Some(NoiseConfig { shock_type: shock, nu, ewma_alpha })
    };

    // Run simulation based on model type
    let ensemble = match model {
        ModelType::Brownian => {
            let sigma = model_params.get("sigma")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
            let model = BrownianMotion::new(sigma);
            run_simulation(integrator, model.clone(), model, calc_enum, &spec, paths, seed, noise_cfg.clone())?
        },
        
        ModelType::GeometricBrownianMotion => {
            let mu = model_params.get("mu").and_then(|v| v.as_f64()).unwrap_or(0.05);
            let sigma = model_params.get("sigma").and_then(|v| v.as_f64()).unwrap_or(0.2);
            let s0 = model_params.get("s0").and_then(|v| v.as_f64()).unwrap_or(100.0);
            
            let model = GeometricBrownianMotion::new(mu, sigma);
            let x0s = vec![State::new(vec![s0]); paths];
            run_simulation_with_x0s(integrator, model.clone(), model, calc_enum, &spec, &x0s, seed, noise_cfg.clone())?
        },
        
        ModelType::OrnsteinUhlenbeck => {
            let theta = model_params.get("theta").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let mu = model_params.get("mu").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let sigma = model_params.get("sigma").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let x0 = model_params.get("x0").and_then(|v| v.as_f64()).unwrap_or(0.0);
            
            let model = OrnsteinUhlenbeck::new(theta, mu, sigma);
            let x0s = vec![State::new(vec![x0]); paths];
            run_simulation_with_x0s(integrator, model.clone(), model, calc_enum, &spec, &x0s, seed, noise_cfg.clone())?
        },
        
        ModelType::DoubleWell => {
            let a = model_params.get("a").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let b = model_params.get("b").and_then(|v| v.as_f64()).unwrap_or(2.0);
            let temperature = model_params.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.1);
            let x0 = model_params.get("x0").and_then(|v| v.as_f64()).unwrap_or(-1.0);
            
            let model = DoubleWell::new(a, b, temperature);
            let x0s = vec![State::new(vec![x0]); paths];
            run_simulation_with_x0s(integrator, model.clone(), model, calc_enum, &spec, &x0s, seed, noise_cfg.clone())?
        },
    };
    
    // Write output files
    let parquet_path = out.to_str().unwrap();
    let manifest_path = out.with_extension("manifest.json");
    let manifest_path_str = manifest_path.to_str().unwrap();
    
    write_ensemble_with_manifest(&ensemble, &manifest, parquet_path, manifest_path_str)?;
    
    // Print summary statistics
    let stats = ensemble.final_statistics();
    println!("");
    println!("Summary Statistics:");
    println!("==================");
    println!("Paths completed: {}", stats.n_paths);
    
    if !stats.means.is_empty() {
        println!("Final state means: {:?}", stats.means.data.as_vec());
        println!("Final state stds: {:?}", 
                 stats.variances.data.as_vec().iter().map(|v| v.sqrt()).collect::<Vec<_>>());
    }
    
    if !stats.first_passage_times.is_empty() {
        let mean_fpt = stats.first_passage_times.iter().sum::<f64>() / stats.first_passage_times.len() as f64;
        println!("Transitions: {} / {} ({:.1}%)", 
                 stats.first_passage_times.len(), 
                 ensemble.n_paths(),
                 stats.first_passage_times.len() as f64 / ensemble.n_paths() as f64 * 100.0);
        println!("Mean first passage time: {:.4}", mean_fpt);
    }
    
    println!("✓ Simulation completed successfully!");
    
    Ok(())
}

/// Generic simulation runner
fn run_simulation<D, S>(
    integrator: IntegratorType,
    drift: D,
    diffusion: S,
    calc: Calc,
    spec: &PathSpec,
    n_paths: usize,
    seed: u64,
    noise_cfg: Option<NoiseConfig>,
) -> anyhow::Result<bicep_sampler::Ensemble>
where
    D: bicep_core::Drift + Clone,
    S: bicep_core::Diffusion + Clone,
{
    let x0s = vec![State::new(vec![0.0]); n_paths];
    run_simulation_with_x0s(integrator, drift, diffusion, calc, spec, &x0s, seed, noise_cfg)
}

/// Simulation runner with custom initial conditions
fn run_simulation_with_x0s<D, S>(
    integrator: IntegratorType,
    drift: D,
    diffusion: S,
    calc: Calc,
    spec: &PathSpec,
    x0s: &[State],
    seed: u64,
    noise_cfg: Option<NoiseConfig>,
) -> anyhow::Result<bicep_sampler::Ensemble>
where
    D: bicep_core::Drift + Clone,
    S: bicep_core::Diffusion + Clone,
{
    let boundary = Boundary::None;
    let stopping = Stopping::default();
    
    match integrator {
        IntegratorType::EulerMaruyama => {
            let sampler = Sampler::new(EulerMaruyama, drift, diffusion).with_noise_config(noise_cfg);
            Ok(sampler.run_paths(calc, spec, x0s, &boundary, &stopping, seed))
        },
        
        IntegratorType::Milstein => {
            let sampler = Sampler::new(Milstein, drift, diffusion).with_noise_config(noise_cfg);
            Ok(sampler.run_paths(calc, spec, x0s, &boundary, &stopping, seed))
        },
        
        IntegratorType::HeunStratonovich => {
            let sampler = Sampler::new(HeunStratonovich, drift, diffusion).with_noise_config(noise_cfg);
            Ok(sampler.run_paths(calc, spec, x0s, &boundary, &stopping, seed))
        },
    }
}
