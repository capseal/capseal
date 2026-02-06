use bicep_core::noise::NoiseGenerator;
use bicep_core::path::{Path, PathSpec};
use bicep_core::{integrators::EulerMaruyama, Calc, SdeIntegrator, State};
use bicep_models::GeometricBrownianMotion;

fn main() {
    // GBM parameters (e.g., stock price model)
    let mu = 0.05; // 5% annual drift
    let sigma = 0.2; // 20% annual volatility
    let gbm = GeometricBrownianMotion::new(mu, sigma);

    // Simulation parameters
    let x0 = 100.0; // Initial price
    let dt = 1.0 / 252.0; // Daily steps (252 trading days/year)
    let n_steps = 252; // One year
    let n_paths = 5; // Generate 5 sample paths

    // Path specification
    let path_spec = PathSpec::new(n_steps, dt);

    println!("Simulating {} GBM paths for {} days", n_paths, n_steps);
    println!("Initial value: {}, μ={}, σ={}", x0, mu, sigma);
    println!();

    // Generate paths
    let integrator = EulerMaruyama;

    for path_id in 0..n_paths {
        let mut rng = NoiseGenerator::from_path_id(42, path_id as u64);
        let mut path = Path::with_capacity(n_steps + 1);
        let mut state = State::new(vec![x0]);
        let mut t = 0.0;

        // Save initial state
        path.push(t, state.clone());

        // Simulate path
        for _ in 0..n_steps {
            let dw = rng.generate_dw(1, dt.sqrt());
            state = integrator.step(Calc::Ito, t, &state, dt, &dw, &gbm, &gbm);
            t += dt;
            path.push(t, state.clone());
        }

        let final_value = path.final_state().unwrap().0[0];
        let return_pct = (final_value / x0 - 1.0) * 100.0;

        println!(
            "Path {}: Final value = {:.2}, Return = {:.2}%",
            path_id, final_value, return_pct
        );
    }

    // Theoretical statistics
    let expected_final = gbm.exact_mean(x0, path_spec.final_time());
    let variance_final = gbm.exact_variance(x0, path_spec.final_time());
    let std_final = variance_final.sqrt();

    println!("\nTheoretical statistics at T={}:", path_spec.final_time());
    println!("Expected value: {:.2}", expected_final);
    println!("Standard deviation: {:.2}", std_final);
    println!(
        "95% confidence interval: [{:.2}, {:.2}]",
        expected_final - 1.96 * std_final,
        expected_final + 1.96 * std_final
    );
}
