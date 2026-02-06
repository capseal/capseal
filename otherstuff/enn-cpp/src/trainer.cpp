#include "enn/trainer.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace enn {

// Numerically stable sigmoid
inline F stable_sigmoid(F x) {
    if (x >= 0) {
        return 1.0 / (1.0 + std::exp(-x));
    } else {
        F ex = std::exp(x);
        return ex / (1.0 + ex);
    }
}

// Numerically stable BCE loss: -[y*log(p) + (1-y)*log(1-p)] where p = sigmoid(logit)
// Equivalent to: max(logit,0) - logit*y + log(1 + exp(-|logit|))
inline F stable_bce_loss(F logit, F target) {
    F abs_logit = std::abs(logit);
    return std::max(logit, F(0)) - logit * target + std::log1p(std::exp(-abs_logit));
}

// BCE gradient w.r.t. logit: sigmoid(logit) - target
inline F bce_grad(F logit, F target) {
    return stable_sigmoid(logit) - target;
}

// Vectorized BCE loss for multi-bit output: sum over bits
inline F stable_bce_loss_vec(const Vec& logits, const Vec& targets) {
    F total = 0.0;
    for (int i = 0; i < logits.size(); ++i) {
        total += stable_bce_loss(logits(i), targets(i));
    }
    return total;
}

// Vectorized BCE gradient: returns [output_dim] gradient vector
inline Vec bce_grad_vec(const Vec& logits, const Vec& targets) {
    Vec grad(logits.size());
    for (int i = 0; i < logits.size(); ++i) {
        grad(i) = stable_sigmoid(logits(i)) - targets(i);
    }
    return grad;
}

SequenceTrainer::SequenceTrainer(int k, int raw_input_dim, int embed_dim, int hidden_dim,
                                F lambda, const TrainConfig& config)
    : config_(config), embed_dim_(embed_dim), raw_input_dim_(raw_input_dim) {
    frontend_ = std::make_unique<SpatialTemporalCNN>(raw_input_dim_, embed_dim_,
                                                     config.frontend_filters,
                                                     config.frontend_temporal_kernel,
                                                     config.frontend_depth_kernel);
    cell_ = std::make_unique<EntangledCell>(k, embed_dim_, hidden_dim,
                                            lambda, config.use_layer_norm);
    collapse_ = std::make_unique<Collapse>(k, config.output_dim);
    optimizer_ = std::make_unique<AdamW>(config.learning_rate, 0.9, 0.999, 1e-8,
                                         config.weight_decay);
    opt_state_ = std::make_unique<OptimizerState>(k, embed_dim_, hidden_dim,
                                                  config.frontend_filters, raw_input_dim_,
                                                  config.frontend_temporal_kernel,
                                                  config.frontend_depth_kernel,
                                                  config.output_dim);
}

F SequenceTrainer::train_epoch(const SeqBatch& data) {
    F total_loss = 0.0;
    int num_batches = 0;

    for (size_t start = 0; start < data.batch_size(); start += config_.batch_size) {
        size_t end = std::min(start + config_.batch_size, data.batch_size());
        if (start >= end) break;

        EntangledCell::Grads cell_acc(cell_->k, embed_dim_, cell_->hidden_dim);
        cell_acc.zero();
        FrontendGrads front_acc(config_.frontend_filters, raw_input_dim_,
                                config_.frontend_temporal_kernel,
                                config_.frontend_depth_kernel, embed_dim_);
        front_acc.zero();
        Collapse::Grads collapse_acc(cell_->k, config_.output_dim);
        collapse_acc.zero();
        F batch_loss = 0.0;

        const bool is_multi = config_.output_dim > 1 && data.is_multi_bit();

        for (size_t i = start; i < end; ++i) {
            SequenceCache cache;
            F seq_loss;
            // Extract weights if available and enabled
            std::vector<F> w_seq;
            if (config_.use_weighted_loss && !data.weights.empty()) {
                w_seq = data.weights[i];
            }

            if (is_multi) {
                seq_loss = train_sequence_multi(data.sequences[i], data.multi_targets[i], cache);
            } else {
                seq_loss = train_sequence(data.sequences[i], data.targets[i], cache, w_seq);
            }
            batch_loss += seq_loss;

            EntangledCell::Grads seq_cell(cell_->k, embed_dim_, cell_->hidden_dim);
            seq_cell.zero();
            FrontendGrads seq_front(config_.frontend_filters, raw_input_dim_,
                                    config_.frontend_temporal_kernel,
                                    config_.frontend_depth_kernel, embed_dim_);
            seq_front.zero();
            Collapse::Grads seq_collapse(cell_->k, config_.output_dim);
            seq_collapse.zero();

            if (is_multi) {
                backward_through_time_multi(data.sequences[i], data.multi_targets[i], cache,
                                            seq_cell, seq_front, seq_collapse);
            } else {
                backward_through_time(data.sequences[i], data.targets[i], cache,
                                      seq_cell, seq_front, seq_collapse, w_seq);
            }

            cell_acc.add_scaled(seq_cell, 1.0);
            front_acc.add_scaled(seq_front, 1.0);
            collapse_acc.add_scaled(seq_collapse, 1.0);
        }

        F scale = 1.0 / static_cast<F>(end - start);
        cell_acc.scale(scale);
        front_acc.scale(scale);
        collapse_acc.scale(scale);

        F reg_loss = compute_regularization_loss();
        apply_gradients(cell_acc, front_acc, collapse_acc, reg_loss);

        total_loss += batch_loss * scale;
        num_batches++;
    }

    return (num_batches > 0) ? total_loss / num_batches : total_loss;
}

F SequenceTrainer::train_sequence(const std::vector<Vec>& inputs,
                                  const std::vector<F>& targets,
                                  SequenceCache& cache,
                                  const std::vector<F>& weights) {
    const int seq_len = static_cast<int>(inputs.size());

    cache.cell_caches.resize(seq_len);
    cache.collapse_caches.resize(seq_len);
    cache.psi_history.resize(seq_len);
    cache.h_history.resize(seq_len);
    cache.embeddings.resize(seq_len);
    cache.embed_grads.assign(seq_len, Vec::Zero(embed_dim_));
    cache.predictions.resize(seq_len);
    cache.initial_psi = Vec::Zero(cell_->k);
    cache.initial_h = Vec::Zero(cell_->hidden_dim);

    frontend_->forward_sequence(inputs, cache.embeddings, cache.frontend_cache);

    Vec psi = cache.initial_psi;
    Vec h = cache.initial_h;
    F total_loss = 0.0;

    for (int t = 0; t < seq_len; ++t) {
        psi = cell_->forward(cache.embeddings[t], h, psi, cache.cell_caches[t]);
        cache.psi_history[t] = psi;
        cache.h_history[t] = h;

        F pred = collapse_->forward(psi, cache.collapse_caches[t]);
        cache.predictions[t] = pred;
        // Only accumulate loss at final timestep if loss_final_only is set
        if (!config_.loss_final_only || t == seq_len - 1) {
            F step_loss = 0.0;
            if (config_.use_bce_loss) {
                // BCE loss: treat pred as logit
                step_loss = stable_bce_loss(pred, targets[t]);
            } else {
                // MSE loss
                F diff = pred - targets[t];
                step_loss = 0.5 * diff * diff;
            }

            // Apply weight if enabled
            if (config_.use_weighted_loss && !weights.empty() && static_cast<size_t>(t) < weights.size()) {
                step_loss *= weights[t];
            }
            total_loss += step_loss;
        }
    }

    return total_loss;
}

F SequenceTrainer::train_sequence_multi(const std::vector<Vec>& inputs,
                                        const std::vector<Vec>& targets,
                                        SequenceCache& cache) {
    const int seq_len = static_cast<int>(inputs.size());

    cache.cell_caches.resize(seq_len);
    cache.collapse_caches.resize(seq_len);
    cache.psi_history.resize(seq_len);
    cache.h_history.resize(seq_len);
    cache.embeddings.resize(seq_len);
    cache.embed_grads.assign(seq_len, Vec::Zero(embed_dim_));
    cache.multi_predictions.resize(seq_len);
    cache.initial_psi = Vec::Zero(cell_->k);
    cache.initial_h = Vec::Zero(cell_->hidden_dim);

    frontend_->forward_sequence(inputs, cache.embeddings, cache.frontend_cache);

    Vec psi = cache.initial_psi;
    Vec h = cache.initial_h;
    F total_loss = 0.0;

    for (int t = 0; t < seq_len; ++t) {
        psi = cell_->forward(cache.embeddings[t], h, psi, cache.cell_caches[t]);
        cache.psi_history[t] = psi;
        cache.h_history[t] = h;

        // Multi-bit forward: returns [output_dim] logits
        Vec pred = collapse_->forward_multi(psi, cache.collapse_caches[t]);
        cache.multi_predictions[t] = pred;

        // Only accumulate loss at final timestep if loss_final_only is set
        if (!config_.loss_final_only || t == seq_len - 1) {
            // BCE loss per bit, summed
            total_loss += stable_bce_loss_vec(pred, targets[t]);
        }
    }

    return total_loss;
}

void SequenceTrainer::backward_through_time(const std::vector<Vec>& inputs,
                                            const std::vector<F>& targets,
                                            SequenceCache& cache,
                                            EntangledCell::Grads& cell_grads,
                                            FrontendGrads& front_grads,
                                            Collapse::Grads& collapse_grads,
                                            const std::vector<F>& weights) {
    const int seq_len = static_cast<int>(targets.size());
    Vec dpsi_future = Vec::Zero(cell_->k);
    Vec dh_future = Vec::Zero(cell_->hidden_dim);

    for (int t = seq_len - 1; t >= 0; --t) {
        F dL_dpred;
        if (config_.use_bce_loss) {
            // BCE gradient: sigmoid(logit) - target
            dL_dpred = bce_grad(cache.predictions[t], targets[t]);
        } else {
            // MSE gradient: pred - target
            dL_dpred = cache.predictions[t] - targets[t];
        }

        // Apply weight if enabled
        if (config_.use_weighted_loss && !weights.empty() && static_cast<size_t>(t) < weights.size()) {
            dL_dpred *= weights[t];
        }

        // Zero out gradient for non-final timesteps if loss_final_only is set
        if (config_.loss_final_only && t != seq_len - 1) {
            dL_dpred = 0.0;
        }

        Vec dpsi_collapse;
        collapse_->backward(dL_dpred, cache.psi_history[t],
                            cache.collapse_caches[t], dpsi_collapse,
                            collapse_grads);

        Vec dpsi_total = dpsi_collapse + dpsi_future;
        Vec dpsi_in, dh, dx;
        cell_->backward(dpsi_total, cache.cell_caches[t], cell_grads,
                        dpsi_in, dh, dx);
        cache.embed_grads[t] = dx;

        dpsi_future = dpsi_in;
        dh_future = dh;
        (void)dh_future; // hidden state not yet recurrent, placeholder for future use
    }

    frontend_->backward_sequence(inputs, cache.frontend_cache,
                                 cache.embed_grads, front_grads);
}

void SequenceTrainer::backward_through_time_multi(const std::vector<Vec>& inputs,
                                                  const std::vector<Vec>& targets,
                                                  SequenceCache& cache,
                                                  EntangledCell::Grads& cell_grads,
                                                  FrontendGrads& front_grads,
                                                  Collapse::Grads& collapse_grads) {
    const int seq_len = static_cast<int>(targets.size());
    Vec dpsi_future = Vec::Zero(cell_->k);
    Vec dh_future = Vec::Zero(cell_->hidden_dim);

    for (int t = seq_len - 1; t >= 0; --t) {
        // BCE gradient: sigmoid(logit) - target, per bit
        Vec dL_dpred = bce_grad_vec(cache.multi_predictions[t], targets[t]);

        // Zero out gradient for non-final timesteps if loss_final_only is set
        if (config_.loss_final_only && t != seq_len - 1) {
            dL_dpred.setZero();
        }

        Vec dpsi_collapse;
        collapse_->backward_multi(dL_dpred, cache.psi_history[t],
                                  cache.collapse_caches[t], dpsi_collapse,
                                  collapse_grads);

        Vec dpsi_total = dpsi_collapse + dpsi_future;
        Vec dpsi_in, dh, dx;
        cell_->backward(dpsi_total, cache.cell_caches[t], cell_grads,
                        dpsi_in, dh, dx);
        cache.embed_grads[t] = dx;

        dpsi_future = dpsi_in;
        dh_future = dh;
        (void)dh_future; // hidden state not yet recurrent, placeholder for future use
    }

    frontend_->backward_sequence(inputs, cache.frontend_cache,
                                 cache.embed_grads, front_grads);
}

void SequenceTrainer::apply_gradients(const EntangledCell::Grads& cell_grads,
                                      const FrontendGrads& front_grads,
                                      const Collapse::Grads& collapse_grads,
                                      F reg_loss) {
    (void)reg_loss;  // Loss value used for logging; gradients computed below

    // Advance optimizer time step ONCE per batch (not per parameter)
    // This is critical for correct Adam/AdamW bias correction
    optimizer_->tick();

    // Copy gradients so we can modify them
    EntangledCell::Grads cell = cell_grads;
    FrontendGrads front = front_grads;
    Collapse::Grads collapse = collapse_grads;

    // =========================================================================
    // (A) GRADIENT CLIPPING - prevents optimization collapse / grad spikes
    // =========================================================================
    if (config_.grad_clip_norm > 0) {
        // Compute global gradient norm across all parameters
        F grad_norm_sq = 0.0;
        grad_norm_sq += cell.dWx.squaredNorm();
        grad_norm_sq += cell.dWh.squaredNorm();
        grad_norm_sq += cell.dL.squaredNorm();
        grad_norm_sq += cell.db.squaredNorm();
        grad_norm_sq += cell.dgamma.squaredNorm();
        grad_norm_sq += cell.dbeta.squaredNorm();
        grad_norm_sq += cell.dlog_lambda * cell.dlog_lambda;
        grad_norm_sq += front.dW_temporal.squaredNorm();
        grad_norm_sq += front.db_temporal.squaredNorm();
        grad_norm_sq += front.dW_spatial.squaredNorm();
        grad_norm_sq += front.db_spatial.squaredNorm();
        grad_norm_sq += front.dW_depthwise.squaredNorm();
        grad_norm_sq += front.db_depthwise.squaredNorm();
        grad_norm_sq += front.dW_proj.squaredNorm();
        grad_norm_sq += front.db_proj.squaredNorm();
        grad_norm_sq += collapse.dWq.squaredNorm();
        grad_norm_sq += collapse.dWout.squaredNorm();
        grad_norm_sq += collapse.dbias.squaredNorm();  // Vec, not scalar
        grad_norm_sq += collapse.dlog_temp * collapse.dlog_temp;

        F grad_norm = std::sqrt(grad_norm_sq);
        if (grad_norm > config_.grad_clip_norm) {
            F scale = config_.grad_clip_norm / grad_norm;
            cell.dWx *= scale;
            cell.dWh *= scale;
            cell.dL *= scale;
            cell.db *= scale;
            cell.dgamma *= scale;
            cell.dbeta *= scale;
            cell.dlog_lambda *= scale;
            front.dW_temporal *= scale;
            front.db_temporal *= scale;
            front.dW_spatial *= scale;
            front.db_spatial *= scale;
            front.dW_depthwise *= scale;
            front.db_depthwise *= scale;
            front.dW_proj *= scale;
            front.db_proj *= scale;
            collapse.dWq *= scale;
            collapse.dWout *= scale;
            collapse.dbias *= scale;
            collapse.dlog_temp *= scale;
        }
    }

    // =========================================================================
    // (B) EXPLICIT REGULARIZATION - only if NOT using AdamW weight_decay
    // =========================================================================
    // Don't double-regularize: pick AdamW decay OR explicit reg_eta, not both
    if (!config_.use_adamw_decay && config_.reg_eta > 0) {
        // L2 regularization: grad += reg_eta * W
        // (C) Only decay weight matrices, NOT biases or LayerNorm params
        cell.dWx += config_.reg_eta * cell_->Wx;
        cell.dWh += config_.reg_eta * cell_->Wh;
        cell.dL  += config_.reg_eta * cell_->L;

        if (!config_.decay_weights_only) {
            // Only if explicitly requested - usually destabilizes training
            cell.db     += config_.reg_eta * cell_->b;
            cell.dgamma += config_.reg_eta * cell_->ln_gamma;
            cell.dbeta  += config_.reg_eta * cell_->ln_beta;
        }
    }

    // PSD regularizer (additional L2 on L for entanglement norm control)
    if (config_.reg_beta > 0) {
        cell.dL += config_.reg_beta * 1e-6 * cell_->L;
    }

    // =========================================================================
    // OPTIMIZER STEPS
    // =========================================================================
    optimizer_->step(cell_->Wx, opt_state_->m_Wx, opt_state_->v_Wx, cell.dWx);
    optimizer_->step(cell_->Wh, opt_state_->m_Wh, opt_state_->v_Wh, cell.dWh);
    optimizer_->step(cell_->L, opt_state_->m_L, opt_state_->v_L, cell.dL);
    optimizer_->step(cell_->b, opt_state_->m_b, opt_state_->v_b, cell.db);
    optimizer_->step(cell_->ln_gamma, opt_state_->m_ln_gamma, opt_state_->v_ln_gamma, cell.dgamma);
    optimizer_->step(cell_->ln_beta, opt_state_->m_ln_beta, opt_state_->v_ln_beta, cell.dbeta);
    optimizer_->step(cell_->log_lambda, opt_state_->m_log_lambda, opt_state_->v_log_lambda,
                     cell.dlog_lambda);

    optimizer_->step(frontend_->W_temporal(), opt_state_->m_front_temporal,
                     opt_state_->v_front_temporal, front.dW_temporal);
    optimizer_->step(frontend_->b_temporal(), opt_state_->m_front_temporal_b,
                     opt_state_->v_front_temporal_b, front.db_temporal);
    optimizer_->step(frontend_->W_spatial(), opt_state_->m_front_spatial,
                     opt_state_->v_front_spatial, front.dW_spatial);
    optimizer_->step(frontend_->b_spatial(), opt_state_->m_front_spatial_b,
                     opt_state_->v_front_spatial_b, front.db_spatial);
    optimizer_->step(frontend_->W_depthwise(), opt_state_->m_front_depthwise,
                     opt_state_->v_front_depthwise, front.dW_depthwise);
    optimizer_->step(frontend_->b_depthwise(), opt_state_->m_front_depthwise_b,
                     opt_state_->v_front_depthwise_b, front.db_depthwise);
    optimizer_->step(frontend_->W_proj(), opt_state_->m_front_proj,
                     opt_state_->v_front_proj, front.dW_proj);
    optimizer_->step(frontend_->b_proj(), opt_state_->m_front_proj_b,
                     opt_state_->v_front_proj_b, front.db_proj);

    optimizer_->step(collapse_->Wq, opt_state_->m_Wq, opt_state_->v_Wq, collapse.dWq);
    optimizer_->step(collapse_->Wout, opt_state_->m_Wout, opt_state_->v_Wout, collapse.dWout);
    optimizer_->step(collapse_->bout, opt_state_->m_collapse_bias, opt_state_->v_collapse_bias,
                     collapse.dbias);
    optimizer_->step(collapse_->log_temp, opt_state_->m_log_temp, opt_state_->v_log_temp,
                     collapse.dlog_temp);
}

F SequenceTrainer::compute_regularization_loss() {
    F reg_loss = 0.0;
    if (config_.reg_beta > 0) {
        reg_loss += config_.reg_beta * cell_->compute_psd_regularizer_loss();
    }
    if (config_.reg_eta > 0) {
        reg_loss += config_.reg_eta * cell_->compute_param_l2_loss();
    }
    return reg_loss;
}

F SequenceTrainer::evaluate(const SeqBatch& data, Metrics& metrics) {
    metrics.reset();
    F total_loss = 0.0;
    const bool is_multi = config_.output_dim > 1 && data.is_multi_bit();

    for (size_t i = 0; i < data.batch_size(); ++i) {
        F seq_loss = 0.0;

        if (is_multi) {
            // Multi-bit evaluation
            std::vector<Vec> preds = forward_sequence_multi(data.sequences[i]);
            for (size_t t = 0; t < data.multi_targets[i].size(); ++t) {
                F loss = stable_bce_loss_vec(preds[t], data.multi_targets[i][t]);
                if (!config_.loss_final_only || t == data.multi_targets[i].size() - 1) {
                    seq_loss += loss;
                }
                if (t == data.multi_targets[i].size() - 1) {
                    // Final timestep: update multi-bit metrics
                    metrics.update_multi(preds[t], data.multi_targets[i][t], loss);
                }
            }
        } else {
            // Scalar evaluation (backwards compatible)
            std::vector<F> preds = forward_sequence(data.sequences[i]);
            for (size_t t = 0; t < data.targets[i].size(); ++t) {
                F loss;
                if (config_.use_bce_loss) {
                    loss = stable_bce_loss(preds[t], data.targets[i][t]);
                } else {
                    F diff = preds[t] - data.targets[i][t];
                    loss = 0.5 * diff * diff;
                }
                if (!config_.loss_final_only || t == data.targets[i].size() - 1) {
                    seq_loss += loss;
                }
                if (t == data.targets[i].size() - 1) {
                    F pred_for_metrics = config_.use_bce_loss ? stable_sigmoid(preds[t]) : preds[t];
                    metrics.update(pred_for_metrics, data.targets[i][t], loss);
                }
            }
        }
        total_loss += seq_loss;
    }

    metrics.finalize();
    return total_loss / std::max<size_t>(1, data.batch_size());
}

std::vector<F> SequenceTrainer::forward_sequence(const std::vector<Vec>& sequence,
                                                 Vec* final_psi, Vec* final_h,
                                                 Vec* final_alpha, F* final_temperature) const {
    const int seq_len = static_cast<int>(sequence.size());
    std::vector<Vec> embeddings(seq_len);
    FrontendCache front_cache;
    frontend_->forward_sequence(sequence, embeddings, front_cache);

    std::vector<F> predictions;
    predictions.reserve(seq_len);
    Vec psi = Vec::Zero(cell_->k);
    Vec h = Vec::Zero(cell_->hidden_dim);
    Vec last_alpha = Vec::Zero(cell_->k);
    F last_temp = std::exp(collapse_->log_temp);

    for (int t = 0; t < seq_len; ++t) {
        CellCache cache;
        psi = cell_->forward(embeddings[t], h, psi, cache);
        CollapseCache collapse_cache;
        F pred = collapse_->forward(psi, collapse_cache);
        predictions.push_back(pred);
        last_alpha = collapse_cache.alpha;
        last_temp = collapse_cache.temperature;
    }

    if (final_psi) *final_psi = psi;
    if (final_h) *final_h = h;
    if (final_alpha) *final_alpha = last_alpha;
    if (final_temperature) *final_temperature = last_temp;
    return predictions;
}

std::vector<Vec> SequenceTrainer::forward_sequence_multi(const std::vector<Vec>& sequence) const {
    const int seq_len = static_cast<int>(sequence.size());
    std::vector<Vec> embeddings(seq_len);
    FrontendCache front_cache;
    frontend_->forward_sequence(sequence, embeddings, front_cache);

    std::vector<Vec> predictions;
    predictions.reserve(seq_len);
    Vec psi = Vec::Zero(cell_->k);
    Vec h = Vec::Zero(cell_->hidden_dim);

    for (int t = 0; t < seq_len; ++t) {
        CellCache cache;
        psi = cell_->forward(embeddings[t], h, psi, cache);
        CollapseCache collapse_cache;
        Vec pred = collapse_->forward_multi(psi, collapse_cache);
        predictions.push_back(pred);
    }

    return predictions;
}

TrainerWithScheduler::TrainerWithScheduler(std::unique_ptr<SequenceTrainer> trainer,
                                           F base_lr, F min_lr, int total_steps)
    : trainer_(std::move(trainer)) {
    scheduler_ = std::make_unique<CosineScheduler>(base_lr, min_lr, total_steps);
}

F TrainerWithScheduler::train_epoch(const SeqBatch& data) {
    update_learning_rate();
    return trainer_->train_epoch(data);
}

F TrainerWithScheduler::evaluate(const SeqBatch& data, Metrics& metrics) {
    return trainer_->evaluate(data, metrics);
}

void TrainerWithScheduler::update_learning_rate() {
    F new_lr = (*scheduler_)(current_step_);
    trainer_->set_learning_rate(new_lr);
    current_step_++;
}

F TrainerWithScheduler::get_current_lr() const {
    return (*scheduler_)(current_step_);
}

} // namespace enn
