use bicep_core::noise::NoiseGenerator;
use bicep_core::{integrators::EulerMaruyama, Calc, SdeIntegrator, State};
use bicep_models::BrownianMotion;

#[test]
fn reflecting_boundaries() {
    // Simple reflecting boundary implementation
    let lower_bound = -2.0;
    let upper_bound = 2.0;
    let model = BrownianMotion::standard();
    let integrator = EulerMaruyama;

    let x0 = 0.0;
    let dt: f64 = 1e-3;
    let steps = 10000;
    let n_paths = 1000;

    let tolerance = 1e-12;

    for path_id in 0..n_paths {
        let mut rng = NoiseGenerator::from_path_id(42, path_id as u64);
        let mut state = State::new(vec![x0]);
        let mut t = 0.0;

        for _ in 0..steps {
            let dw = rng.generate_dw(1, dt.sqrt());
            state = integrator.step(Calc::Ito, t, &state, dt, &dw, &model, &model);

            // Apply reflecting boundary manually
            if state.0[0] < lower_bound {
                state.0[0] = 2.0 * lower_bound - state.0[0];
            } else if state.0[0] > upper_bound {
                state.0[0] = 2.0 * upper_bound - state.0[0];
            }

            // Assert within bounds (with tolerance)
            assert!(
                state.0[0] >= lower_bound - tolerance && state.0[0] <= upper_bound + tolerance,
                "State {} outside reflecting bounds [{}, {}]",
                state.0[0],
                lower_bound,
                upper_bound
            );

            t += dt;
        }
    }

    println!("Reflecting boundaries test passed: all states within bounds");
}

#[test]
fn periodic_boundaries() {
    // Simple periodic boundary implementation
    let period = 4.0; // Period length
    let lower = -2.0; // [-2, 2] interval
    let upper = 2.0;
    let model = BrownianMotion::standard();
    let integrator = EulerMaruyama;

    let x0 = 0.0;
    let dt: f64 = 1e-3;
    let steps = 10000;

    let mut rng = NoiseGenerator::new(42);
    let mut state = State::new(vec![x0]);
    let mut t = 0.0;
    let mut wrap_count = 0i32;

    for _ in 0..steps {
        let dw = rng.generate_dw(1, dt.sqrt());
        state = integrator.step(Calc::Ito, t, &state, dt, &dw, &model, &model);

        // Apply periodic boundary manually
        while state.0[0] > upper {
            state.0[0] -= period;
            wrap_count += 1;
        }
        while state.0[0] < lower {
            state.0[0] += period;
            wrap_count -= 1;
        }

        // Assert within bounds
        assert!(
            state.0[0] >= lower && state.0[0] <= upper,
            "Wrapped state {} outside [{}, {}]",
            state.0[0],
            lower,
            upper
        );

        t += dt;
    }

    println!("Periodic boundaries test:");
    println!(
        "Final position: {:.3}, Wrap count: {}",
        state.0[0], wrap_count
    );
}

#[test]
fn absorbing_boundaries() {
    // Test first-passage time recording
    let absorbing_level = 3.0; // Increased level to make it harder to reach
    let model = BrownianMotion::standard();
    let integrator = EulerMaruyama;

    let x0 = 0.0;
    let dt: f64 = 1e-3;
    let max_steps = 20000; // Reduced max steps
    let n_paths = 100;

    let mut first_passage_times = Vec::new();

    for path_id in 0..n_paths {
        let mut rng = NoiseGenerator::from_path_id(42, path_id as u64);
        let mut state = State::new(vec![x0]);
        let mut t = 0.0;
        for _ in 0..max_steps {
            let dw = rng.generate_dw(1, dt.sqrt());
            state = integrator.step(Calc::Ito, t, &state, dt, &dw, &model, &model);
            t += dt;

            // Check absorption
            if state.0[0].abs() >= absorbing_level {
                first_passage_times.push(t);
                break; // Stop simulation
            }
        }
    }

    // Statistics
    let n_absorbed = first_passage_times.len();
    let absorption_rate = n_absorbed as f64 / n_paths as f64;

    println!("Absorbing boundaries test:");
    println!("Absorption rate: {:.2}%", absorption_rate * 100.0);
    println!("Absorbed paths: {}/{}", n_absorbed, n_paths);

    assert!(
        absorption_rate > 0.05,
        "Too few paths absorbed (got {:.1}%)",
        absorption_rate * 100.0
    );
    assert!(
        absorption_rate < 0.95,
        "Too many paths absorbed (got {:.1}%)",
        absorption_rate * 100.0
    );
}

#[test]
fn first_hit_stopping() {
    // Test that simulation can detect first-passage times
    let model = BrownianMotion::standard();
    let integrator = EulerMaruyama;

    let x0 = 0.0;
    let dt: f64 = 1e-3;
    let barrier = 1.0;
    let max_steps = 20000;

    let mut rng = NoiseGenerator::new(42);
    let mut state = State::new(vec![x0]);
    let mut t = 0.0;
    let mut first_hit_time = None;

    for _ in 0..max_steps {
        let dw = rng.generate_dw(1, dt.sqrt());
        state = integrator.step(Calc::Ito, t, &state, dt, &dw, &model, &model);
        t += dt;

        if state.0[0] >= barrier && first_hit_time.is_none() {
            first_hit_time = Some(t);
            break; // Stop at first hit
        }
    }

    println!("First-hit stopping test:");
    if let Some(hit_time) = first_hit_time {
        println!("First hit time: {:.4}", hit_time);
        println!("Final state: {:.4}", state.0[0]);
        assert!(state.0[0] >= barrier, "First hit state below barrier");
    } else {
        println!("No hit detected in {} steps", max_steps);
    }
}
