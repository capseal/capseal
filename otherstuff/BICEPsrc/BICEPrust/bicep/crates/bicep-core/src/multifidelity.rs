use crate::{Calc, Diffusion, Drift, NoiseGenerator, SdeIntegrator, State};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct MfEstimate {
    pub value: f64,           // The final multifidelity estimate
    pub variance: f64,        // Estimated variance of the estimator
    pub q_high_naive: f64,    // Standard MC estimate (High-Fi only)
    pub q_low_naive: f64,     // Standard MC estimate (Low-Fi only)
    pub correlation: f64,     // Correlation between High and Low fidelity
    pub beta: f64,            // Optimal control variate coefficient
    pub n_high: usize,
    pub n_low: usize,
}

/// Run a simulation and return 1.0 if it hits A, 0.0 if it hits B.
/// If it hits neither by max_steps, we (for now) return 0.0 or handle as incomplete.
fn run_trajectory<D: Drift + Sync, S: Diffusion + Sync, I: SdeIntegrator + Sync>(
    drift: &D,
    diffusion: &S,
    integrator: &I,
    start: &State,
    dt: f64,
    calc: Calc,
    noise_gen: &mut NoiseGenerator,
    is_a: &impl Fn(&State) -> bool,
    is_b: &impl Fn(&State) -> bool,
    max_steps: usize,
) -> f64 {
    let mut x = start.clone();
    let dim = x.dim();
    let sqrt_dt = dt.sqrt();
    let mut t = 0.0;

    for _ in 0..max_steps {
        if is_a(&x) { return 1.0; }
        if is_b(&x) { return 0.0; }

        let dw = noise_gen.generate_dw(dim, sqrt_dt);
        x = integrator.step(calc, t, &x, dt, &dw, drift, diffusion);
        t += dt;
    }
    // Timeout implies failure to reach goal in time
    0.0
}

/// Run a PAIRED simulation (High-Fi and Low-Fi sharing noise).
/// Returns (y_high, y_low).
/// 
/// Note: This implementation assumes dt_low is a integer multiple of dt_high.
/// Ratio R = dt_low / dt_high.
fn run_paired_trajectory<
    D1: Drift + Sync, S1: Diffusion + Sync, I1: SdeIntegrator + Sync,
    D2: Drift + Sync, S2: Diffusion + Sync, I2: SdeIntegrator + Sync
>(
    high: (&D1, &S1, &I1, f64),
    low: (&D2, &S2, &I2, f64),
    start: &State,
    calc: Calc,
    noise_gen: &mut NoiseGenerator,
    is_a: &impl Fn(&State) -> bool,
    is_b: &impl Fn(&State) -> bool,
    max_steps_high: usize,
) -> (f64, f64) {
    let (d1, s1, i1, dt1) = high;
    let (d2, s2, i2, dt2) = low;

    // Check ratio
    let ratio = (dt2 / dt1).round() as usize;
    assert!(ratio >= 1, "Low-Fi dt must be >= High-Fi dt");
    
    let mut x1 = start.clone();
    let mut x2 = start.clone();
    let dim = start.dim();
    let sqrt_dt1 = dt1.sqrt();
    
    let mut t1 = 0.0;
    let mut t2 = 0.0;
    
    let mut finished1 = false;
    let mut finished2 = false;
    let mut res1 = 0.0;
    let mut res2 = 0.0;

    // Accumulator for Low-Fi noise
    // dW_L = sum(dW_H)
    let mut dw_accum = State::zeros(dim); 

    for step in 0..max_steps_high {
        // If both finished, break
        if finished1 && finished2 { break; }

        // 1. Check conditions
        if !finished1 {
            if is_a(&x1) { res1 = 1.0; finished1 = true; }
            else if is_b(&x1) { res1 = 0.0; finished1 = true; }
        }
        if !finished2 {
            if is_a(&x2) { res2 = 1.0; finished2 = true; }
            else if is_b(&x2) { res2 = 0.0; finished2 = true; }
        }

        if finished1 && finished2 { break; }

        // 2. Generate Fine Noise
        let dw_fine = noise_gen.generate_dw(dim, sqrt_dt1);

        // 3. Step High-Fi
        if !finished1 {
            // Fix: remove * dereference, pass references directly
            x1 = i1.step(calc, t1, &x1, dt1, &dw_fine, d1, s1);
            t1 += dt1;
        }

        // 4. Accumulate for Low-Fi
        if !finished2 {
            // Add dW components
            for k in 0..dim {
                dw_accum[k] += dw_fine[k];
            }

            // If we hit the ratio boundary, step Low-Fi
            if (step + 1) % ratio == 0 {
                // Fix: remove * dereference, pass references directly
                x2 = i2.step(calc, t2, &x2, dt2, &dw_accum, d2, s2);
                t2 += dt2;
                // Reset accumulator
                dw_accum = State::zeros(dim);
            }
        }
    }

    (res1, res2)
}

pub struct MultifidelityEstimator;

impl MultifidelityEstimator {
    pub fn estimate<
        D1: Drift + Sync, S1: Diffusion + Sync, I1: SdeIntegrator + Sync,
        D2: Drift + Sync, S2: Diffusion + Sync, I2: SdeIntegrator + Sync
    >(
        start: &State,
        is_a: &(impl Fn(&State) -> bool + Sync),
        is_b: &(impl Fn(&State) -> bool + Sync),
        high: (&D1, &S1, &I1, f64),
        low: (&D2, &S2, &I2, f64),
        n_high: usize,
        n_cheap_only: usize,
        seed: u64,
        max_steps_high: usize,
    ) -> MfEstimate {
        // 1. Run Paired Simulations (N_High)
        let results_paired: Vec<(f64, f64)> = (0..n_high).into_par_iter().map(|i| {
            // Deterministic seeding for parallel reproducibility
            let mut rng = NoiseGenerator::from_path_id(seed, i as u64);
            run_paired_trajectory(
                high, low, start, Calc::Ito, &mut rng, is_a, is_b, max_steps_high
            )
        }).collect();

        // 2. Run Cheap-Only Simulations (N_Cheap_Only)
        // Note: Use a different seed offset (n_high) to ensure independence
        let results_cheap_only: Vec<f64> = (0..n_cheap_only).into_par_iter().map(|i| {
            let mut rng = NoiseGenerator::from_path_id(seed, (n_high + i) as u64);
            let (d2, s2, i2, dt2) = low;
            // Adjust max steps for low fi
            let ratio = (dt2 / high.3).round() as usize;
            let max_steps_low = max_steps_high / ratio;
            
            // Fix: remove * dereference
            run_trajectory(
                d2, s2, i2, start, dt2, Calc::Ito, &mut rng, is_a, is_b, max_steps_low
            )
        }).collect();

        // 3. Compute Statistics from Paired Data
        let y_h_vec: Vec<f64> = results_paired.iter().map(|(h, _)| *h).collect();
        let y_l_paired_vec: Vec<f64> = results_paired.iter().map(|(_, l)| *l).collect();

        let mean_h = mean(&y_h_vec);
        let mean_l_paired = mean(&y_l_paired_vec);
        
        let var_h = variance(&y_h_vec, mean_h);
        let var_l = variance(&y_l_paired_vec, mean_l_paired);
        let cov_hl = covariance(&y_h_vec, &y_l_paired_vec, mean_h, mean_l_paired);
        
        // Beta* = Cov(H, L) / Var(L)
        let beta = if var_l > 1e-12 { cov_hl / var_l } else { 0.0 };

        // 4. Compute Full Low-Fi Mean (N_L = N_High + N_Cheap_Only)
        // Combine paired L samples and cheap-only L samples
        let sum_l_paired: f64 = y_l_paired_vec.iter().sum();
        let sum_l_cheap: f64 = results_cheap_only.iter().sum();
        let n_total_l = n_high + n_cheap_only;
        let mean_l_total = (sum_l_paired + sum_l_cheap) / (n_total_l as f64);

        // 5. Control Variate Estimate
        // q_MF = mean(H) + beta * (mean(L_total) - mean(L_paired))
        let q_mf = mean_h + beta * (mean_l_total - mean_l_paired);

        // 6. Variance Estimation
        // Var(q_MF) = (1/N_H) * Var(H - beta*L_paired) + (beta^2 / N_L) * Var(L)
        //           = (1/N_H) * (Var(H) + beta^2*Var(L) - 2*beta*Cov(H,L)) + (beta^2/N_L)*Var(L)
        
        let var_residual = var_h + beta.powi(2) * var_l - 2.0 * beta * cov_hl;
        let term1 = var_residual / (n_high as f64);
        let term2 = (beta.powi(2) * var_l) / (n_total_l as f64);
        
        let est_variance = term1 + term2;
        
        // For correlation reporting
        let correlation = if var_h > 1e-12 && var_l > 1e-12 {
            cov_hl / (var_h.sqrt() * var_l.sqrt())
        } else {
            0.0
        };

        MfEstimate {
            value: q_mf.clamp(0.0, 1.0), // Probabilities must be in [0,1]
            variance: est_variance,
            q_high_naive: mean_h,
            q_low_naive: mean_l_total,
            correlation,
            beta,
            n_high,
            n_low: n_total_l,
        }
    }
}

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

fn variance(data: &[f64], mean: f64) -> f64 {
    if data.len() < 2 { return 0.0; }
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

fn covariance(x: &[f64], y: &[f64], mx: f64, my: f64) -> f64 {
    if x.len() < 2 { return 0.0; }
    x.iter().zip(y.iter())
        .map(|(xi, yi)| (xi - mx) * (yi - my))
        .sum::<f64>() / (x.len() - 1) as f64
}