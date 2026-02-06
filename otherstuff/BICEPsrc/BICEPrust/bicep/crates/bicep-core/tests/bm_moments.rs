use bicep_core::noise::NoiseGenerator;
use bicep_core::{integrators::EulerMaruyama, Calc, SdeIntegrator, State};
use bicep_models::BrownianMotion;
use serde_json::json;
use std::fs::File;
use std::io::Write;

#[test]
fn brownian_moments() {
    // Test parameters
    let n_paths = 10_000; // Reduced for faster testing
    let steps = 1000;
    let dt: f64 = 1e-3;
    let t_final = steps as f64 * dt; // T = 1.0

    // Create BM model
    let bm = BrownianMotion::standard();
    let integrator = EulerMaruyama;

    // Run simulation
    let mut final_values = Vec::with_capacity(n_paths);

    for path_id in 0..n_paths {
        let mut rng = NoiseGenerator::from_path_id(42, path_id as u64);
        let mut state = State::new(vec![0.0]); // X_0 = 0
        let mut t = 0.0;

        for _ in 0..steps {
            let dw = rng.generate_dw(1, dt.sqrt());
            state = integrator.step(Calc::Ito, t, &state, dt, &dw, &bm, &bm);
            t += dt;
        }

        final_values.push(state.0[0]);
    }

    // Compute statistics
    let mean = final_values.iter().sum::<f64>() / n_paths as f64;
    let var = final_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n_paths - 1) as f64;

    // Standard error for mean
    let stderr = (t_final / n_paths as f64).sqrt();

    // Write results
    let results = json!({
        "mean": mean,
        "var": var,
        "stderr": stderr,
        "expected_mean": 0.0,
        "expected_var": t_final,
        "n_paths": n_paths,
        "t_final": t_final
    });

    std::fs::create_dir_all("runs").ok();
    let mut file = File::create("runs/bm_moments.json").unwrap();
    write!(file, "{}", serde_json::to_string(&results).unwrap()).unwrap();

    // Assertions
    println!("BM Test Results:");
    println!("Mean: {:.6} (expected: 0)", mean);
    println!("Variance: {:.6} (expected: {})", var, t_final);
    println!("Standard error: {:.6}", stderr);

    // |mean| < 4 * stderr
    assert!(
        mean.abs() < 4.0 * stderr,
        "Mean {} exceeds 4 standard errors ({})",
        mean,
        4.0 * stderr
    );

    // |var - T| / T < 0.05
    let var_rel_error = (var - t_final).abs() / t_final;
    assert!(
        var_rel_error < 0.05,
        "Variance relative error {} exceeds 5%",
        var_rel_error
    );
}
