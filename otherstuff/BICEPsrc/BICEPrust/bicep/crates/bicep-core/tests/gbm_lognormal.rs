use bicep_core::noise::NoiseGenerator;
use bicep_core::{integrators::EulerMaruyama, Calc, SdeIntegrator, State};
use bicep_models::GeometricBrownianMotion;
use serde_json::json;
use std::fs::File;
use std::io::Write;

#[test]
fn gbm_lognormal() {
    // GBM parameters
    let mu = 0.2;
    let sigma = 0.35;
    let x0 = 1.0;
    let t_final = 1.0;
    let n_paths = 10_000; // Reduced for faster testing
    let dt: f64 = 1e-3;
    let steps = (t_final / dt) as usize;

    let gbm = GeometricBrownianMotion::new(mu, sigma);
    let integrator = EulerMaruyama;

    // Collect log(X_T) values
    let mut log_final_values = Vec::with_capacity(n_paths);

    for path_id in 0..n_paths {
        let mut rng = NoiseGenerator::from_path_id(42, path_id as u64);
        let mut state = State::new(vec![x0]);
        let mut t = 0.0;

        for _ in 0..steps {
            let dw = rng.generate_dw(1, dt.sqrt());
            state = integrator.step(Calc::Ito, t, &state, dt, &dw, &gbm, &gbm);
            t += dt;
        }

        log_final_values.push(state.0[0].ln());
    }

    // Theoretical distribution: log(X_T) ~ N(mu_theory, sigma2_theory)
    let mu_theory = x0.ln() + (mu - sigma.powi(2) / 2.0) * t_final;
    let sigma2_theory = sigma.powi(2) * t_final;

    // Sample statistics
    let mu_hat = log_final_values.iter().sum::<f64>() / n_paths as f64;
    let sigma2_hat = log_final_values
        .iter()
        .map(|x| (x - mu_hat).powi(2))
        .sum::<f64>()
        / (n_paths - 1) as f64;

    // Kolmogorov-Smirnov test
    let ks_pvalue = compute_ks_test(&log_final_values, mu_theory, sigma2_theory.sqrt());

    // Write results
    let results = json!({
        "mu_hat": mu_hat,
        "sigma2_hat": sigma2_hat,
        "ks_pvalue": ks_pvalue,
        "mu_theory": mu_theory,
        "sigma2_theory": sigma2_theory,
        "n_paths": n_paths
    });

    std::fs::create_dir_all("runs").ok();
    let mut file = File::create("runs/gbm_lognormal.json").unwrap();
    write!(file, "{}", serde_json::to_string(&results).unwrap()).unwrap();

    println!("GBM Log-Normal Test Results:");
    println!("Sample mean: {:.6} (expected: {:.6})", mu_hat, mu_theory);
    println!(
        "Sample variance: {:.6} (expected: {:.6})",
        sigma2_hat, sigma2_theory
    );
    println!("KS p-value: {:.6}", ks_pvalue);

    // Assert KS p-value > 0.05
    assert!(
        ks_pvalue > 0.05,
        "KS test failed: p-value {} <= 0.05",
        ks_pvalue
    );
}

fn compute_ks_test(samples: &[f64], mean: f64, std: f64) -> f64 {
    // Simple KS test implementation
    let n = samples.len();
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut d_max: f64 = 0.0;

    for (i, &x) in sorted.iter().enumerate() {
        let z = (x - mean) / std;
        let cdf_theoretical = normal_cdf(z);
        let cdf_empirical = (i + 1) as f64 / n as f64;

        let d1 = (cdf_empirical - cdf_theoretical).abs();
        let d2 = (i as f64 / n as f64 - cdf_theoretical).abs();

        d_max = d_max.max(d1).max(d2);
    }

    // Simplified p-value approximation using Kolmogorov distribution
    let sqrt_n = (n as f64).sqrt();
    let lambda = sqrt_n * d_max;

    // Kolmogorov distribution approximation
    let p_value = 2.0 * (-2.0 * lambda.powi(2)).exp();
    p_value.min(1.0)
}

fn normal_cdf(z: f64) -> f64 {
    // Standard normal CDF using error function approximation
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}
