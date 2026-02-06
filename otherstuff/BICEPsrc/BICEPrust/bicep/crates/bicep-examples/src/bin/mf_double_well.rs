use std::fs::File;
use std::io::{BufWriter, Write};

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

// -----------------------------
// Double-well SDE (2D)
// U(x,y) = (x^2 - 1)^2 + y^2
// dX = -∇U(X) dt + sigma dW
// -----------------------------

#[derive(Clone, Copy, Debug)]
struct State {
    x: f64,
    y: f64,
}

fn drift(s: State) -> State {
    // U = (x^2 - 1)^2 + y^2
    // dU/dx = 4x(x^2 - 1)
    // dU/dy = 2y
    // drift = -∇U
    State {
        x: -4.0 * s.x * (s.x * s.x - 1.0),
        y: -2.0 * s.y,
    }
}

fn is_a(s: State) -> bool {
    // Goal set A: right well
    s.x > 0.9
}

fn is_b(s: State) -> bool {
    // Failure set B: left well
    s.x < -0.9
}

fn euler_maruyama_step(s: State, dt: f64, sigma: f64, dWx: f64, dWy: f64) -> State {
    let b = drift(s);
    State {
        x: s.x + b.x * dt + sigma * dWx,
        y: s.y + b.y * dt + sigma * dWy,
    }
}

// Run one trajectory, return hit indicator Y in {0,1}:
// 1 if hit A before B, 0 if hit B before A (or timed out -> treat as 0 by default)
fn rollout_hit(
    x0: State,
    dt: f64,
    t_max: f64,
    sigma: f64,
    rng: &mut ChaCha8Rng,
    shared_increments: Option<&[(f64, f64)]>, // if provided, use these dW increments (paired sim)
) -> u8 {
    let steps = (t_max / dt).ceil() as usize;
    let sqrt_dt = dt.sqrt();

    let mut s = x0;

    for k in 0..steps {
        if is_a(s) {
            return 1;
        }
        if is_b(s) {
            return 0;
        }

        let (z1, z2) = if let Some(incs) = shared_increments {
            incs[k]
        } else {
            (
                rng.sample::<f64, _>(rand_distr::StandardNormal),
                rng.sample::<f64, _>(rand_distr::StandardNormal),
            )
        };

        // Brownian increments: dW ~ N(0, dt) so we use sqrt(dt)*Z
        let dWx = sqrt_dt * z1;
        let dWy = sqrt_dt * z2;

        s = euler_maruyama_step(s, dt, sigma, dWx, dWy);
    }

    // Timeout handling:
    // For committor problems, you can either (a) discard timeouts, (b) treat as 0, (c) treat as 0.5.
    // We’ll treat as 0 here to be conservative.
    0
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / (xs.len() as f64)
}

fn var_unbiased(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let s2 = xs.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / ((xs.len() - 1) as f64);
    s2
}

fn cov_unbiased(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    if x.len() < 2 {
        return 0.0;
    }
    let mx = mean(x);
    let my = mean(y);
    let c = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (a - mx) * (b - my))
        .sum::<f64>()
        / ((x.len() - 1) as f64);
    c
}

// Multifidelity control variate estimator:
// q̂_MF = Ȳ_H + β (Ȳ_L - Ȳ_L^(H-paired))
// where:
// - paired batch i=1..N_H gives (Y_H^i, Y_L^i)
// - cheap-only batch j=1..N_L gives Y_L^j (independent)
// β* = Cov(Y_H, Y_L) / Var(Y_L)  (regularized)
fn multifidelity_estimate(
    x0: State,
    n_hi: usize,
    n_lo: usize,
    dt_hi: f64,
    dt_lo: f64,
    t_max: f64,
    sigma: f64,
    beta_l2: f64,
    rng: &mut ChaCha8Rng,
) -> (f64, f64) {
    // --- Paired batch with shared Brownian normals (Zs) ---
    let steps_hi = (t_max / dt_hi).ceil() as usize;
    let steps_lo = (t_max / dt_lo).ceil() as usize;

    // For pairing, we’ll generate normals at the *fine* resolution and reuse them.
    // Low-fidelity uses a coarser dt, so it consumes blocks of fine increments.
    // Simple approach: create fine Zs and for low-fidelity, sum the block increments.
    let mut y_h = Vec::with_capacity(n_hi);
    let mut y_l_paired = Vec::with_capacity(n_hi);

    for _ in 0..n_hi {
        // fine normal draws (z1,z2) per fine step
        let mut fine_normals = Vec::with_capacity(steps_hi);
        for _k in 0..steps_hi {
            let z1 = rng.sample::<f64, _>(rand_distr::StandardNormal);
            let z2 = rng.sample::<f64, _>(rand_distr::StandardNormal);
            fine_normals.push((z1, z2));
        }

        // High-fidelity uses fine normals directly
        let yh = rollout_hit(x0, dt_hi, t_max, sigma, rng, Some(&fine_normals)) as f64;
        y_h.push(yh);

        // Low-fidelity: aggregate fine normals into coarse normals
        let block = ((dt_lo / dt_hi).round() as usize).max(1);
        let mut coarse_normals = Vec::with_capacity(steps_lo);

        // Combine blocks: sum sqrt(dt_hi)*Z over block -> sqrt(dt_lo)*Z_coarse
        // Equivalent: Z_coarse = (1/sqrt(block)) * sum Z_i
        let mut idx = 0usize;
        while idx < fine_normals.len() && coarse_normals.len() < steps_lo {
            let end = (idx + block).min(fine_normals.len());
            let mut s1 = 0.0;
            let mut s2 = 0.0;
            for k in idx..end {
                s1 += fine_normals[k].0;
                s2 += fine_normals[k].1;
            }
            let norm = (end - idx) as f64;
            coarse_normals.push((s1 / norm.sqrt(), s2 / norm.sqrt()));
            idx = end;
        }

        let yl = rollout_hit(x0, dt_lo, t_max, sigma, rng, Some(&coarse_normals)) as f64;
        y_l_paired.push(yl);
    }

    // --- Cheap-only batch (independent low-fidelity) ---
    let mut y_l = Vec::with_capacity(n_lo);
    for _ in 0..n_lo {
        let yl = rollout_hit(x0, dt_lo, t_max, sigma, rng, None) as f64;
        y_l.push(yl);
    }

    // --- β* with regularization ---
    let var_l = var_unbiased(&y_l_paired);
    let cov_hl = cov_unbiased(&y_h, &y_l_paired);
    let beta = if var_l < 1e-12 {
        0.0
    } else {
        cov_hl / (var_l + beta_l2)
    };

    let y_h_bar = mean(&y_h);
    let y_l_bar = mean(&y_l);
    let y_lh_bar = mean(&y_l_paired);

    let q_hat = y_h_bar + beta * (y_l_bar - y_lh_bar);

    // --- Variance estimate (the “don’t lie to yourself” version) ---
    // Var(Ȳ_H - β Ȳ_L^(H)) + β^2 Var(Ȳ_L)
    // with unbiased sample estimates:
    let var_yh = var_unbiased(&y_h);
    let var_ylh = var_unbiased(&y_l_paired);
    let cov_yh_ylh = cov_hl;

    let nH = n_hi as f64;
    let nL = n_lo as f64;

    let term_paired = (var_yh + beta * beta * var_ylh - 2.0 * beta * cov_yh_ylh) / nH;
    let term_lo = (beta * beta * var_unbiased(&y_l)) / nL;

    let var_hat = (term_paired + term_lo).max(1e-12);

    (q_hat.clamp(0.0, 1.0), var_hat)
}

fn main() -> anyhow::Result<()> {
    // Minimal “edit constants then run” workflow.
    // If you want CLI args, I’ll add clap in 30 seconds.
    let out_path = "double_well_data.csv";

    let seed: u64 = 1337;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Dataset: sample initial points in [-2,2]^2
    let n_points: usize = 4000;
    let lo = -2.0;
    let hi = 2.0;

    // Simulation params
    let sigma = 0.7; // noise strength (tune this)
    let t_max = 6.0;

    // Multifidelity params
    let n_hi = 64; // expensive paired
    let n_lo = 512; // cheap-only
    let dt_hi = 0.0025;
    let dt_lo = 0.01;

    // β regularization
    let beta_l2 = 1e-6;

    // Weight clamp (prevents ENN from worshipping one point)
    let w_max = 1e6;
    let eps = 1e-9;

    let f = File::create(out_path)?;
    let mut w = BufWriter::new(f);

    writeln!(w, "x,y,q_hat,var,weight,n_hi,n_lo,dt_hi,dt_lo,sigma,t_max")?;

    for _ in 0..n_points {
        let x = rng.gen_range(lo..hi);
        let y = rng.gen_range(lo..hi);
        let x0 = State { x, y };

        let (q_hat, var_hat) = multifidelity_estimate(
            x0, n_hi, n_lo, dt_hi, dt_lo, t_max, sigma, beta_l2, &mut rng,
        );

        let weight = (1.0 / (var_hat + eps)).min(w_max);

        writeln!(
            w,
            "{:.6},{:.6},{:.6},{:.8e},{:.6e},{},{},{:.6},{:.6},{:.6},{:.6}",
            x, y, q_hat, var_hat, weight, n_hi, n_lo, dt_hi, dt_lo, sigma, t_max
        )?;
    }

    w.flush()?;
    eprintln!("Wrote {}", out_path);
    Ok(())
}
