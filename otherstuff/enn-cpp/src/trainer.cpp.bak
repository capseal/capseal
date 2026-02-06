#include "enn/trainer.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>

namespace enn {

SequenceTrainer::SequenceTrainer(int k, int input_dim, int hidden_dim, F lambda, 
                                const TrainConfig& config)
    : config_(config) {
    
    cell_ = std::make_unique<EntangledCell>(k, input_dim, hidden_dim, lambda);
    collapse_ = std::make_unique<Collapse>(k);
    optimizer_ = std::make_unique<AdamW>(config.learning_rate, 0.9, 0.999, 1e-8, config.weight_decay);
    opt_state_ = std::make_unique<OptimizerState>(k, input_dim, hidden_dim);
}

F SequenceTrainer::train_epoch(const SeqBatch& data) {
    F total_loss = 0.0;
    int num_batches = 0;
    
    // Process sequences in batches
    for (size_t start = 0; start < data.batch_size(); start += config_.batch_size) {
        size_t end = std::min(start + config_.batch_size, data.batch_size());
        
        EntangledCell::Grads accumulated_cell_grads(cell_->k, cell_->input_dim, cell_->hidden_dim);
        Mat accumulated_collapse_grads = Mat::Zero(cell_->k, cell_->k);
        F batch_loss = 0.0;
        
        // Process each sequence in the batch
        for (size_t i = start; i < end; ++i) {
            SequenceCache cache;
            F seq_loss = train_sequence(data.sequences[i], data.targets[i], cache);
            
            EntangledCell::Grads seq_cell_grads(cell_->k, cell_->input_dim, cell_->hidden_dim);
            Mat seq_collapse_grads = Mat::Zero(cell_->k, cell_->k);
            
            // Get predictions for backprop
            std::vector<F> predictions;
            Vec psi = cache.initial_psi;
            Vec h = cache.initial_h;
            
            for (size_t t = 0; t < data.sequences[i].size(); ++t) {
                psi = cell_->forward(data.sequences[i][t], h, psi, cache.cell_caches[t]);
                cache.psi_history[t] = psi;
                cache.h_history[t] = h;
                
                F pred = collapse_->forward(psi, cache.collapse_caches[t]);
                predictions.push_back(pred);
            }
            
            // Backward through time for this sequence
            backward_through_time(data.targets[i], predictions, cache, 
                                seq_cell_grads, seq_collapse_grads);
            
            // Accumulate gradients
            accumulated_cell_grads.add_scaled(seq_cell_grads, 1.0);
            accumulated_collapse_grads += seq_collapse_grads;
            batch_loss += seq_loss;
        }
        
        // Scale by batch size
        F scale = 1.0 / (end - start);
        accumulated_cell_grads.dWx *= scale;
        accumulated_cell_grads.dWh *= scale;
        accumulated_cell_grads.dL *= scale;
        accumulated_cell_grads.db *= scale;
        accumulated_cell_grads.dlambda *= scale;
        accumulated_collapse_grads *= scale;
        
        // Add regularization
        F reg_loss = compute_regularization_loss();
        
        // Apply gradients
        apply_gradients(accumulated_cell_grads, accumulated_collapse_grads, reg_loss);
        
        total_loss += batch_loss * scale;
        num_batches++;
    }
    
    return total_loss / num_batches;
}

F SequenceTrainer::train_sequence(const std::vector<Vec>& inputs, const std::vector<F>& targets,
                                 SequenceCache& cache) {
    int seq_len = inputs.size();
    
    // Initialize cache
    cache.cell_caches.resize(seq_len);
    cache.collapse_caches.resize(seq_len);
    cache.psi_history.resize(seq_len);
    cache.h_history.resize(seq_len);
    cache.initial_psi = Vec::Zero(cell_->k);
    cache.initial_h = Vec::Zero(cell_->hidden_dim);
    
    // Forward pass
    Vec psi = cache.initial_psi;
    Vec h = cache.initial_h;
    F total_loss = 0.0;
    
    for (int t = 0; t < seq_len; ++t) {
        // Cell forward
        psi = cell_->forward(inputs[t], h, psi, cache.cell_caches[t]);
        cache.psi_history[t] = psi;
        cache.h_history[t] = h;
        
        // Collapse forward
        F pred = collapse_->forward(psi, cache.collapse_caches[t]);
        
        // Compute loss (MSE)
        F loss = 0.5 * (pred - targets[t]) * (pred - targets[t]);
        total_loss += loss;
    }
    
    return total_loss;
}

void SequenceTrainer::backward_through_time(const std::vector<F>& targets,
                                           const std::vector<F>& predictions,
                                           const SequenceCache& cache,
                                           EntangledCell::Grads& cell_grads,
                                           Mat& collapse_grads) {
    int seq_len = targets.size();
    
    // Initialize upstream gradients
    Vec dpsi_future = Vec::Zero(cell_->k);
    Vec dh_future = Vec::Zero(cell_->hidden_dim);
    
    // Backward pass through time
    for (int t = seq_len - 1; t >= 0; --t) {
        // Loss gradient
        F dL_dpred = predictions[t] - targets[t];
        
        // Collapse backward
        Vec dpsi_collapse;
        Mat dWg;
        collapse_->backward(dL_dpred, cache.psi_history[t], cache.collapse_caches[t], 
                           dpsi_collapse, dWg);
        collapse_grads += dWg;
        
        // Total gradient w.r.t. psi at this timestep
        Vec dpsi_total = dpsi_collapse + dpsi_future;
        
        // Cell backward
        Vec dpsi_in, dh;
        cell_->backward(dpsi_total, cache.cell_caches[t], cell_grads, dpsi_in, dh);
        
        // Prepare gradients for previous timestep
        dpsi_future = dpsi_in;
        dh_future = dh;
    }
}

void SequenceTrainer::apply_gradients(const EntangledCell::Grads& cell_grads,
                                     const Mat& collapse_grads, F /* reg_loss */) {
    // Apply gradients with AdamW
    optimizer_->step(cell_->Wx, opt_state_->m_Wx, opt_state_->v_Wx, cell_grads.dWx);
    optimizer_->step(cell_->Wh, opt_state_->m_Wh, opt_state_->v_Wh, cell_grads.dWh);
    optimizer_->step(cell_->L, opt_state_->m_L, opt_state_->v_L, cell_grads.dL);
    optimizer_->step(cell_->b, opt_state_->m_b, opt_state_->v_b, cell_grads.db);
    optimizer_->step(cell_->lambda, opt_state_->m_lambda, opt_state_->v_lambda, cell_grads.dlambda);
    optimizer_->step(collapse_->Wg, opt_state_->m_Wg, opt_state_->v_Wg, collapse_grads);
}

F SequenceTrainer::compute_regularization_loss() {
    F reg_loss = 0.0;
    
    // PSD regularizer (already enforced by L*L^T, just add small penalty)
    if (config_.reg_beta > 0) {
        reg_loss += config_.reg_beta * cell_->compute_psd_regularizer_loss();
    }
    
    // L2 parameter regularizer
    if (config_.reg_eta > 0) {
        reg_loss += config_.reg_eta * cell_->compute_param_l2_loss();
    }
    
    return reg_loss;
}

F SequenceTrainer::evaluate(const SeqBatch& data, Metrics& metrics) {
    metrics.reset();
    F total_loss = 0.0;
    
    for (size_t i = 0; i < data.batch_size(); ++i) {
        std::vector<F> predictions = forward_sequence(data.sequences[i]);
        
        F seq_loss = 0.0;
        for (size_t t = 0; t < data.targets[i].size(); ++t) {
            F loss = 0.5 * (predictions[t] - data.targets[i][t]) * (predictions[t] - data.targets[i][t]);
            seq_loss += loss;
            
            // Update metrics (only final timestep for classification tasks)
            if (t == data.targets[i].size() - 1) {
                metrics.update(predictions[t], data.targets[i][t], loss);
            }
        }
        total_loss += seq_loss;
    }
    
    metrics.finalize();
    return total_loss / data.batch_size();
}

std::vector<F> SequenceTrainer::forward_sequence(const std::vector<Vec>& sequence,
                                                Vec* final_psi, Vec* final_h) const {
    std::vector<F> predictions;
    predictions.reserve(sequence.size());
    
    Vec psi = Vec::Zero(cell_->k);
    Vec h = Vec::Zero(cell_->hidden_dim);
    
    for (const auto& input : sequence) {
        CellCache cache;
        psi = cell_->forward(input, h, psi, cache);
        
        CollapseCache collapse_cache;
        F pred = collapse_->forward(psi, collapse_cache);
        predictions.push_back(pred);
    }
    
    if (final_psi) *final_psi = psi;
    if (final_h) *final_h = h;
    
    return predictions;
}

// Batch trainer implementation
BatchTrainer::BatchTrainer(int k, int input_dim, int hidden_dim, F lambda,
                          const TrainConfig& config) : config_(config) {
    cell_ = std::make_unique<EntangledCell>(k, input_dim, hidden_dim, lambda);
    collapse_ = std::make_unique<Collapse>(k);
    optimizer_ = std::make_unique<AdamW>(config.learning_rate, 0.9, 0.999, 1e-8, config.weight_decay);
    opt_state_ = std::make_unique<OptimizerState>(k, input_dim, hidden_dim);
}

F BatchTrainer::train_epoch(const Batch& data) {
    F total_loss = 0.0;
    int num_batches = 0;
    
    for (size_t start = 0; start < data.size(); start += config_.batch_size) {
        size_t end = std::min(start + config_.batch_size, data.size());
        
        EntangledCell::Grads accumulated_grads(cell_->k, cell_->input_dim, cell_->hidden_dim);
        Mat accumulated_collapse_grads = Mat::Zero(cell_->k, cell_->k);
        F batch_loss = 0.0;
        
        for (size_t i = start; i < end; ++i) {
            Vec psi_in = Vec::Zero(cell_->k);
            Vec h = Vec::Zero(cell_->hidden_dim);
            
            // Forward pass
            CellCache cell_cache;
            Vec psi = cell_->forward(data.inputs[i], h, psi_in, cell_cache);
            
            CollapseCache collapse_cache;
            F pred = collapse_->forward(psi, collapse_cache);
            
            // Loss
            F target = data.targets[i];
            F loss = 0.5 * (pred - target) * (pred - target);
            batch_loss += loss;
            
            // Backward pass
            F dL_dpred = pred - target;
            
            Vec dpsi;
            Mat dWg;
            collapse_->backward(dL_dpred, psi, collapse_cache, dpsi, dWg);
            accumulated_collapse_grads += dWg;
            
            Vec dpsi_in_unused, dh_unused;
            cell_->backward(dpsi, cell_cache, accumulated_grads, dpsi_in_unused, dh_unused);
        }
        
        // Scale gradients
        F scale = 1.0 / (end - start);
        accumulated_grads.dWx *= scale;
        accumulated_grads.dWh *= scale;
        accumulated_grads.dL *= scale;
        accumulated_grads.db *= scale;
        accumulated_grads.dlambda *= scale;
        accumulated_collapse_grads *= scale;
        
        // Add regularization
        F reg_loss = 0.0;
        if (config_.reg_eta > 0) {
            reg_loss += config_.reg_eta * cell_->compute_param_l2_loss();
        }
        
        // Apply gradients
        apply_gradients(accumulated_grads, accumulated_collapse_grads, reg_loss);
        
        total_loss += batch_loss * scale;
        num_batches++;
    }
    
    return total_loss / num_batches;
}

F BatchTrainer::evaluate(const Batch& data, Metrics& metrics) {
    metrics.reset();
    
    for (size_t i = 0; i < data.size(); ++i) {
        F pred = forward(data.inputs[i]);
        F loss = 0.5 * (pred - data.targets[i]) * (pred - data.targets[i]);
        metrics.update(pred, data.targets[i], loss);
    }
    
    metrics.finalize();
    return metrics.loss;
}

F BatchTrainer::forward(const Vec& input, Vec* psi_out) {
    Vec psi_in = Vec::Zero(cell_->k);
    Vec h = Vec::Zero(cell_->hidden_dim);
    
    CellCache cache;
    Vec psi = cell_->forward(input, h, psi_in, cache);
    
    CollapseCache collapse_cache;
    F pred = collapse_->forward(psi, collapse_cache);
    
    if (psi_out) *psi_out = psi;
    return pred;
}

void BatchTrainer::apply_gradients(const EntangledCell::Grads& cell_grads,
                                  const Mat& collapse_grads, F /* reg_loss */) {
    optimizer_->step(cell_->Wx, opt_state_->m_Wx, opt_state_->v_Wx, cell_grads.dWx);
    optimizer_->step(cell_->Wh, opt_state_->m_Wh, opt_state_->v_Wh, cell_grads.dWh);
    optimizer_->step(cell_->L, opt_state_->m_L, opt_state_->v_L, cell_grads.dL);
    optimizer_->step(cell_->b, opt_state_->m_b, opt_state_->v_b, cell_grads.db);
    optimizer_->step(cell_->lambda, opt_state_->m_lambda, opt_state_->v_lambda, cell_grads.dlambda);
    optimizer_->step(collapse_->Wg, opt_state_->m_Wg, opt_state_->v_Wg, collapse_grads);
}

// Scheduler wrapper
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