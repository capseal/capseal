# BEF Pipeline Upgrade Plan

Author: Codex (world's greatest machine-learning research engineer)
Date: $(date)

This document captures the full understanding of the BEF (BICEP → ENN → FusionAlpha) pipeline, lessons extracted from `v46_submission`, and the exhaustive checklist for strengthening our Rust/C++ implementation. Implementation work starts **after** this plan is finalized.

---

## 1. Conceptual Understanding

### 1.1 BICEP (Stochastic Trajectory Generator)
- Simulates stochastic differential equations (SDEs) for input data (EEG in submission; parity data here).
- Generates multiple futures per sample; adaptive budgets prevent runaway cost.
- Provides path statistics (mean, std, quantiles) and uncertainty splits (aleatoric vs. epistemic) when using ensembles/adaptive methods.
- Plays upstream role: quality and diversity of trajectories dictate ENN performance and downstream committor priors.

### 1.2 ENN (Entangled Neural Network)
- Consumes trajectory statistics/path means to produce entangled latent states, attention weights, and committor predictions.
- Architecture combines temporal/spatial feature extractors with PSD-constrained entangled RNN cells and attention-based collapse.
- Outputs both predictions and uncertainty metrics (entropy of logits, attention temperature).
- Serves as core committor predictor; quality of entanglement/attention directly influences FusionAlpha priors.

### 1.3 FusionAlpha (Graph Reasoner)
- Builds sensor/state graphs (correlation + spatial) with caching to avoid repeated work.
- Propagates committor priors via GNN layers (GCN/GAT), severity-scheduled diffusion, and MC-dropout for epistemic estimation.
- Resolves contradictions between neighboring nodes, produces refined committor field and action suggestions.

### 1.4 Auxiliary Classical Models
- PCA/Ridge, Riemannian, stacked models operate on engineered features and metadata (age/sex predictors).
- Provide sanity checks, fast baselines, and potential ensemble boosts.

### 1.5 Uncertainty Plumbing
- Full pipeline tracks aleatoric vs. epistemic components: BICEP (within vs. between SDE variance), ENN (entropy), FusionAlpha (dropout variance).
- Downstream decision logic (severity, gating, risk) relies on these values.

---

## 2. Comprehensive Checklist

### 2.1 BICEP Enhancements
1. **Add OU/oscillatory SDE variants**
   - Implement Ornstein-Uhlenbeck process with learnable theta/sigma scales.
   - Add oscillatory extension with frequency components and learnable amplitudes.
   - Provide event-related potentials (ERP) via Poisson jumps with Gaussian temporal kernels.
2. **Adaptive BICEP module**
   - Ensemble multiple SDEs (configurable count) with learnable mixing weights.
   - Enforce element budget (env/config) and auto-adjust paths-per-SDE.
   - Produce weighted paths plus (aleatoric, epistemic) tensors.
3. **Noise/variance reduction**
   - Support antithetic sampling.
   - Optional low-rank correlated noise via projection matrices.
4. **Uncertainty export**
   - Write aleatoric/epistemic stats and path statistics into Parquet metadata or sidecar JSON for downstream consumption.
5. **Env knobs**
   - Support env vars to override n_paths, disable stochasticity, clamp budgets, or force deterministic runs for diagnostics.

### 2.2 ENN Upgrades
1. **Spatial CNN front-end**
   - Temporal conv → BatchNorm → ELU.
   - Spatial conv across channels.
   - Depthwise separable conv block + pooling + dropout.
   - Projection to embedding dimension.
2. **Entangled RNN cell**
   - PSD entanglement via E = L·Lᵀ.
   - Learnable decay λ (log parameterized, clamped for stability).
   - Optional LayerNorm before tanh.
3. **Collapse layer**
   - Attention scores via gated projection.
   - Learnable temperature (log-space) and ability to export attention weights.
   - Final projection to K → outputs.
4. **Multi-scale option**
   - Support multiple temporal scales (1,2,4) with shared entangled cell.
5. **Entropy + uncertainty outputs**
   - Provide entropy of logits, attention concentration, etc., for severity calculations.
6. **Torch interop**
   - If initial implementation stays PyTorch, wrap as TorchScript callable from Rust/C++ until full port is ready.

### 2.3 FusionAlpha Improvements
1. **Graph construction**
   - Build correlation-based k-NN adjacency per sample.
   - Optional spatial component (predefined channel positions) with exponential decay weighting.
   - Threshold tiny weights and add self-loops.
   - Symmetric normalization of adjacency.
2. **Graph cache**
   - Cache adjacency templates keyed by (C,k, correlation flag, spatial flag, hash of positions).
   - Provide hit/miss stats for diagnostics.
3. **GNN stack**
   - Implement GCN or GAT layers with dropout/activation options.
   - Support hierarchical fusion (local/global) similar to submission for multi-scale reasoning.
4. **Propagation severity**
   - Derive severity from ENN confidence/entropy as they do.
   - Schedule number of propagation steps based on severity.
5. **MC-dropout uncertainty**
   - Run multiple dropout passes to quantify epistemic uncertainty at graph level.
6. **Outputs**
   - Provide refined committor, attention maps, and uncertainty metrics.

### 2.4 Uncertainty Plumbing
1. **BICEP → ENN interface**
   - Pass path statistics and uncertainty splits to ENN.
2. **ENN → FusionAlpha interface**
   - Provide priors + confidence + attention weights per node.
3. **FusionAlpha severity gating**
   - Use the propagated uncertainty to gate action recommendations.
4. **Final outputs**
   - Expose aleatoric, epistemic, and total uncertainty per prediction for downstream consumers (CLI, APIs).

### 2.5 Classical Feature Models & Metadata
1. **Feature extraction**
   - Port the statistical feature logic (bandpower, Hjorth, line length, entropy, correlation stats) to Rust/Python utility.
2. **Linear models**
   - Load JSON configs (scaler mean/std, weights, biases) and implement safe inference (guard zero-variance features).
3. **Metadata predictors**
   - Provide quick outputs for metadata (e.g., age, sex) if relevant to our domain, mainly as sanity check or additional signal.
4. **Ensembling hooks**
   - Combine classical outputs with ENN/FusionAlpha predictions (stacked regression or simple weighted average).

### 2.6 Configuration & Diagnostics
1. **Environment overrides**
   - Mirror submission’s env toggles (channel selection, budgets, deterministic switches, bypass fusion, etc.).
2. **Logging**
   - Record budgets used, severity, cache hits, path counts, etc.
3. **Validation scripts**
   - Extend `run_full_pipeline.sh` to verify new artifacts (uncertainty JSON, fusion outputs).
4. **Visualization hooks**
   - Optionally produce quick plots or CSV summaries for key metrics.

### 2.7 Implementation Roadmap
1. Adaptive BICEP enhancements.
2. ENN architecture upgrade (TorchScript intermediate if needed).
3. FusionAlpha graph + GNN revamp.
4. Uncertainty plumbing integration.
5. Classical feature module and ensembling.
6. Config/logging improvements.
7. End-to-end validation + documentation updates.

---

## 3. Acceptance Criteria
- Pipeline emits enriched Parquet + uncertainty metadata.
- ENN outputs include attention/entropy; severity leverages them.
- FusionAlpha builds cached graphs, runs GNN propagation, outputs refined committor + uncertainty, and writes `fusion_alpha_results.csv` (already in place) with richer fields.
- Classical feature ensemble available as optional signal.
- All components configurable via env/CLI, with logging + validation scripts updated.

---

**Implementation begins only after this checklist is acknowledged.**
