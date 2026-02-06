use bicep_core::noise::NoiseGenerator;
use bicep_core::{integrators::EulerMaruyama, Calc, SdeIntegrator, State};
use bicep_models::GeometricBrownianMotion;
use serde_json::json;
use std::env;
use std::fs::File;
use std::io::Write;

#[test]
fn determinism() {
    // Test parameters
    let mu = 0.15;
    let sigma = 0.25;
    let gbm = GeometricBrownianMotion::new(mu, sigma);
    let integrator = EulerMaruyama;

    let x0 = 1.0;
    let dt: f64 = 1e-3;
    let steps = 1000;
    let n_paths = 10000;

    // Save current thread count
    let original_threads = env::var("RAYON_NUM_THREADS").ok();

    // Run with single thread
    env::set_var("RAYON_NUM_THREADS", "1");
    let (mean_single, var_single, ci_width_single) =
        run_simulation(&gbm, &integrator, x0, dt, steps, n_paths);

    // Run with max threads
    env::remove_var("RAYON_NUM_THREADS"); // Let rayon use all available threads
    let (mean_multi, var_multi, ci_width_multi) =
        run_simulation(&gbm, &integrator, x0, dt, steps, n_paths);

    // Restore original thread count
    if let Some(threads) = original_threads {
        env::set_var("RAYON_NUM_THREADS", threads);
    }

    // Compute differences
    let mean_drift = (mean_multi - mean_single).abs();
    let var_change = (var_multi - var_single).abs() / var_single;
    let ci_change = (ci_width_multi - ci_width_single).abs() / ci_width_single;

    // Write results
    let results = json!({
        "single_thread": {
            "mean": mean_single,
            "var": var_single,
            "ci_width": ci_width_single
        },
        "multi_thread": {
            "mean": mean_multi,
            "var": var_multi,
            "ci_width": ci_width_multi
        },
        "differences": {
            "mean_drift": mean_drift,
            "var_change_pct": var_change * 100.0,
            "ci_change_pct": ci_change * 100.0
        }
    });

    std::fs::create_dir_all("runs").ok();
    let mut file = File::create("runs/determinism.json").unwrap();
    write!(file, "{}", serde_json::to_string(&results).unwrap()).unwrap();

    println!("Determinism Test Results:");
    println!(
        "Single thread - Mean: {:.6}, Var: {:.6}",
        mean_single, var_single
    );
    println!(
        "Multi thread  - Mean: {:.6}, Var: {:.6}",
        mean_multi, var_multi
    );
    println!("Mean drift: {:.2e}", mean_drift);
    println!("Variance change: {:.2}%", var_change * 100.0);
    println!("CI width change: {:.2}%", ci_change * 100.0);

    // Assertions
    assert!(
        mean_drift < 1e-12,
        "Mean drift {} exceeds 1e-12",
        mean_drift
    );
    assert!(
        var_change < 0.01,
        "Variance change {:.2}% exceeds 1%",
        var_change * 100.0
    );
    assert!(
        ci_change < 0.01,
        "CI width change {:.2}% exceeds 1%",
        ci_change * 100.0
    );
}

fn run_simulation(
    model: &GeometricBrownianMotion,
    integrator: &EulerMaruyama,
    x0: f64,
    dt: f64,
    steps: usize,
    n_paths: usize,
) -> (f64, f64, f64) {
    let mut final_values = Vec::with_capacity(n_paths);

    // Use single-threaded simulation for determinism test
    for path_id in 0..n_paths {
        final_values.push(simulate_path(model, integrator, x0, dt, steps, path_id));
    }

    // Compute statistics
    let mean = final_values.iter().sum::<f64>() / n_paths as f64;
    let var = final_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n_paths - 1) as f64;

    // 95% confidence interval width
    let stderr = (var / n_paths as f64).sqrt();
    let ci_width = 2.0 * 1.96 * stderr;

    (mean, var, ci_width)
}

fn simulate_path(
    model: &GeometricBrownianMotion,
    integrator: &EulerMaruyama,
    x0: f64,
    dt: f64,
    steps: usize,
    path_id: usize,
) -> f64 {
    let mut rng = NoiseGenerator::from_path_id(42, path_id as u64);
    let mut state = State::new(vec![x0]);
    let mut t = 0.0;

    for _ in 0..steps {
        let dw = rng.generate_dw(1, dt.sqrt());
        state = integrator.step(Calc::Ito, t, &state, dt, &dw, model, model);
        t += dt;
    }

    state.0[0]
}
