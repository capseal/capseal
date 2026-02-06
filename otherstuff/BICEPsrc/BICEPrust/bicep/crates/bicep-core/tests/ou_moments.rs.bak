use bicep_core::{State, Calc, integrators::EulerMaruyama, SdeIntegrator};
use bicep_core::noise::NoiseGenerator;
use bicep_models::OrnsteinUhlenbeck;
use approx::{assert_relative_eq, assert_abs_diff_eq};

#[test]
fn test_ou_moments() {
    // OU parameters
    let theta = 2.0;  // Mean reversion rate
    let mu = 5.0;     // Long-term mean
    let sigma = 1.5;  // Volatility
    let ou = OrnsteinUhlenbeck::new(theta, mu, sigma);
    
    // Simulation parameters
    let x0 = 10.0;  // Start away from mean
    let dt = 0.0001;
    let n_steps = 10000;
    let n_paths = 10000;
    let final_time = dt * n_steps as f64;
    
    // Expected moments
    let expected_mean = ou.exact_mean(x0, final_time);
    let expected_var = ou.exact_variance(final_time);
    
    // Run Monte Carlo simulation
    let integrator = EulerMaruyama;
    let mut final_values = Vec::with_capacity(n_paths);
    
    for path_id in 0..n_paths {
        let mut rng = NoiseGenerator::from_path_id(42, path_id as u64);
        let mut state = State::new(vec![x0]);
        let mut t = 0.0;
        
        for _ in 0..n_steps {
            let dw = rng.generate_dw(1, dt.sqrt());
            state = integrator.step(Calc::Ito, t, &state, dt, &dw, &ou, &ou);
            t += dt;
        }
        
        final_values.push(state.0[0]);
    }
    
    // Compute sample statistics
    let sample_mean = final_values.iter().sum::<f64>() / n_paths as f64;
    let sample_var = final_values.iter()
        .map(|x| (x - sample_mean).powi(2))
        .sum::<f64>() / (n_paths - 1) as f64;
    
    // Check moments (allow 2% relative error)
    println!("OU Test Results:");
    println!("Expected mean: {:.4}, Sample mean: {:.4}", expected_mean, sample_mean);
    println!("Expected var: {:.4}, Sample var: {:.4}", expected_var, sample_var);
    println!("Stationary var: {:.4}", ou.stationary_variance());
    
    assert_relative_eq!(sample_mean, expected_mean, max_relative = 0.02);
    assert_relative_eq!(sample_var, expected_var, max_relative = 0.05);
}

#[test]
fn test_ou_stationary_distribution() {
    // Test that OU process reaches stationary distribution
    let theta = 1.0;
    let mu = 0.0;
    let sigma = 1.0;
    let ou = OrnsteinUhlenbeck::new(theta, mu, sigma);
    
    // Start from various initial conditions
    let initial_values = vec![-10.0, -5.0, 0.0, 5.0, 10.0];
    let dt: f64 = 0.01;   // Larger time step for faster simulation
    let n_steps = 1000;  // Reduced steps but still long enough
    let n_paths_per_x0 = 200;  // Fewer paths for speed
    
    let integrator = EulerMaruyama;
    let mut all_final_values = Vec::new();
    
    for &x0 in &initial_values {
        for path_id in 0..n_paths_per_x0 {
            let mut rng = NoiseGenerator::from_path_id(123, (x0 as i64 * 1000 + path_id) as u64);
            let mut state = State::new(vec![x0]);
            let mut t = 0.0;
            
            for _ in 0..n_steps {
                let dw = rng.generate_dw(1, dt.sqrt());
                state = integrator.step(Calc::Ito, t, &state, dt, &dw, &ou, &ou);
                t += dt;
            }
            
            all_final_values.push(state.0[0]);
        }
    }
    
    // Check that we've reached stationary distribution
    let sample_mean = all_final_values.iter().sum::<f64>() / all_final_values.len() as f64;
    let sample_var = all_final_values.iter()
        .map(|x| (x - sample_mean).powi(2))
        .sum::<f64>() / (all_final_values.len() - 1) as f64;
    
    let stationary_mean = mu;
    let stationary_var = ou.stationary_variance();
    
    println!("OU Stationary Test:");
    println!("Stationary mean: {:.4}, Sample mean: {:.4}", stationary_mean, sample_mean);
    println!("Stationary var: {:.4}, Sample var: {:.4}", stationary_var, sample_var);
    
    assert_abs_diff_eq!(sample_mean, stationary_mean, epsilon = 0.02);
    assert_relative_eq!(sample_var, stationary_var, max_relative = 0.05);
}