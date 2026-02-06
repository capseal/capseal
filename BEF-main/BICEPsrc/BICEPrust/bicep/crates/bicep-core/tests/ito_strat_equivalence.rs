use approx::assert_relative_eq;
use bicep_core::integrators::{EulerMaruyama, HeunStratonovich};
use bicep_core::noise::NoiseGenerator;
use bicep_core::{Calc, SdeIntegrator, State};
use bicep_models::GeometricBrownianMotion;

#[test]
fn test_ito_stratonovich_equivalence() {
    // Use GBM as test case since it has analytical Jacobian
    let mu = 0.1;
    let sigma = 0.3;
    let gbm = GeometricBrownianMotion::new(mu, sigma);

    // Test parameters
    let x0 = 1.0;
    let dt_values: Vec<f64> = vec![0.1, 0.01, 0.001, 0.0001];
    let n_paths = 1000;
    let n_steps = 100;

    println!("\nItô-Stratonovich Equivalence Test:");
    println!("dt\t\tMean Error\tStd Error");
    println!("{}", "-".repeat(40));

    for &dt in &dt_values {
        let mut ito_finals = Vec::with_capacity(n_paths);
        let mut strat_em_finals = Vec::with_capacity(n_paths);
        let mut strat_heun_finals = Vec::with_capacity(n_paths);

        for path_id in 0..n_paths {
            // Use same random numbers for all methods
            let seed = 12345 + path_id as u64;

            // Itô with Euler-Maruyama
            let mut rng_ito = NoiseGenerator::new(seed);
            let mut state_ito = State::new(vec![x0]);
            let mut t = 0.0;

            for _ in 0..n_steps {
                let dw = rng_ito.generate_dw(1, dt.sqrt());
                state_ito = EulerMaruyama.step(Calc::Ito, t, &state_ito, dt, &dw, &gbm, &gbm);
                t += dt;
            }
            ito_finals.push(state_ito.0[0]);

            // Stratonovich with Euler-Maruyama (using drift correction)
            let mut rng_strat_em = NoiseGenerator::new(seed);
            let mut state_strat_em = State::new(vec![x0]);
            t = 0.0;

            for _ in 0..n_steps {
                let dw = rng_strat_em.generate_dw(1, dt.sqrt());
                state_strat_em =
                    EulerMaruyama.step(Calc::Stratonovich, t, &state_strat_em, dt, &dw, &gbm, &gbm);
                t += dt;
            }
            strat_em_finals.push(state_strat_em.0[0]);

            // Stratonovich with Heun midpoint
            let mut rng_strat_heun = NoiseGenerator::new(seed);
            let mut state_strat_heun = State::new(vec![x0]);
            t = 0.0;

            for _ in 0..n_steps {
                let dw = rng_strat_heun.generate_dw(1, dt.sqrt());
                state_strat_heun = HeunStratonovich.step(
                    Calc::Stratonovich,
                    t,
                    &state_strat_heun,
                    dt,
                    &dw,
                    &gbm,
                    &gbm,
                );
                t += dt;
            }
            strat_heun_finals.push(state_strat_heun.0[0]);
        }

        // Compare distributions
        let ito_mean = ito_finals.iter().sum::<f64>() / n_paths as f64;
        let strat_em_mean = strat_em_finals.iter().sum::<f64>() / n_paths as f64;
        let strat_heun_mean = strat_heun_finals.iter().sum::<f64>() / n_paths as f64;

        let em_error = (strat_em_mean - ito_mean).abs();
        let heun_error = (strat_heun_mean - ito_mean).abs();

        // Compute standard deviations
        let ito_std = (ito_finals
            .iter()
            .map(|x| (x - ito_mean).powi(2))
            .sum::<f64>()
            / (n_paths - 1) as f64)
            .sqrt();
        let strat_em_std = (strat_em_finals
            .iter()
            .map(|x| (x - strat_em_mean).powi(2))
            .sum::<f64>()
            / (n_paths - 1) as f64)
            .sqrt();

        let std_error = (strat_em_std - ito_std).abs();

        println!("{:.4}\t\t{:.6}\t{:.6}", dt, em_error, std_error);

        // For small dt, the methods should converge
        if dt <= 0.001 {
            assert!(
                em_error < 0.01,
                "EM drift correction error too large for dt={}",
                dt
            );
            assert!(heun_error < 0.01, "Heun error too large for dt={}", dt);
        }
    }
}

#[test]
fn test_stratonovich_chain_rule() {
    // Test that Stratonovich preserves ordinary chain rule
    // Use f(X) = X² and check that df = 2X dX in Stratonovich
    let sigma = 0.5;
    let gbm = GeometricBrownianMotion::new(0.0, sigma); // Zero drift for simplicity

    let x0 = 1.0;
    let dt: f64 = 0.0001;
    let n_steps = 10000;
    let n_paths = 1000;

    let mut strat_f_finals = Vec::with_capacity(n_paths);
    let mut computed_f_finals = Vec::with_capacity(n_paths);

    for path_id in 0..n_paths {
        let mut rng = NoiseGenerator::from_path_id(99, path_id as u64);
        let mut x = State::new(vec![x0]);
        let mut f_x = x0 * x0; // f(X) = X²
        let mut t = 0.0;

        for _ in 0..n_steps {
            let dw = rng.generate_dw(1, dt.sqrt());
            let x_old = x.0[0];

            // Update X using Stratonovich
            x = HeunStratonovich.step(Calc::Stratonovich, t, &x, dt, &dw, &gbm, &gbm);

            // Update f(X) using chain rule: df = 2X ∘ dX (Stratonovich)
            let dx = x.0[0] - x_old;
            // For Stratonovich, use midpoint rule: 2 * (X_old + X_new)/2 * dX
            f_x += 2.0 * 0.5 * (x_old + x.0[0]) * dx;

            t += dt;
        }

        strat_f_finals.push(f_x);
        computed_f_finals.push(x.0[0] * x.0[0]);
    }

    // Compare E[f(X)] computed via chain rule vs direct computation
    let chain_rule_mean = strat_f_finals.iter().sum::<f64>() / n_paths as f64;
    let direct_mean = computed_f_finals.iter().sum::<f64>() / n_paths as f64;

    println!("\nStratonovich Chain Rule Test:");
    println!("E[X²] via chain rule: {:.4}", chain_rule_mean);
    println!("E[X²] direct: {:.4}", direct_mean);

    assert_relative_eq!(chain_rule_mean, direct_mean, max_relative = 0.02);
}
