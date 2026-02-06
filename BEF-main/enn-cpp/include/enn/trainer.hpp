#pragma once
#include "cell.hpp"
#include "collapse.hpp"
#include "frontend.hpp"
#include "optim.hpp"
#include "data.hpp"
#include "regularizers.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace enn {

// Training configuration
struct TrainConfig {
    F learning_rate = 1e-3;
    F weight_decay = 1e-4;
    int batch_size = 32;
    int epochs = 100;
    F reg_beta = 1e-3;        // PSD regularizer strength
    F reg_gamma = 0.0;        // KL collapse penalty (0 = disabled)
    F reg_eta = 1e-6;         // L2 parameter penalty
    bool verbose = true;
    int print_every = 10;
    int frontend_filters = 32;
    int frontend_temporal_kernel = 5;
    int frontend_depth_kernel = 3;
    int embed_dim = 32;
    bool use_layer_norm = true;

    // BPTT settings
    int bptt_length = -1;     // -1 = full sequence, >0 = truncated BPTT
    bool accumulate_grads = true;  // Accumulate gradients across timesteps

    // Loss shaping (critical for grokking experiments)
    bool loss_final_only = false;  // if true, compute loss only at final timestep
    bool use_bce_loss = false;     // if true, use BCE loss (treat output as logit); else MSE
    int output_dim = 1;           // 1 for scalar output, >1 for multi-bit output

    // Stability controls (fix optimization collapse before chasing grokking)
    F grad_clip_norm = 0.0;       // If > 0, clip global gradient norm to this value
    bool decay_weights_only = true; // If true, only decay Wx/Wh/L/Wq/Wout, skip biases and LN params
    bool use_adamw_decay = true;  // If true, use AdamW weight_decay; if false, use explicit reg_eta
};

// Optimizer state container
struct OptimizerState {
    // Cell parameter states
    Mat m_Wx, v_Wx, m_Wh, v_Wh, m_L, v_L;
    Vec m_b, v_b;
    Vec m_ln_gamma, v_ln_gamma;
    Vec m_ln_beta, v_ln_beta;
    F m_log_lambda = 0, v_log_lambda = 0;

    // Collapse parameter states (Wout is now [output_dim x k] matrix)
    Mat m_Wq, v_Wq;
    Mat m_Wout, v_Wout;           // [output_dim x k]
    Vec m_collapse_bias, v_collapse_bias;  // [output_dim]
    F m_log_temp = 0, v_log_temp = 0;

    // Frontend parameter states
    Mat m_front_temporal, v_front_temporal;
    Vec m_front_temporal_b, v_front_temporal_b;
    Mat m_front_spatial, v_front_spatial;
    Vec m_front_spatial_b, v_front_spatial_b;
    Mat m_front_depthwise, v_front_depthwise;
    Vec m_front_depthwise_b, v_front_depthwise_b;
    Mat m_front_proj, v_front_proj;
    Vec m_front_proj_b, v_front_proj_b;

    OptimizerState(int k, int embed_dim, int hidden_dim,
                   int frontend_filters, int raw_input_dim,
                   int temporal_kernel, int depth_kernel,
                   int output_dim = 1) {
        m_Wx = Mat::Zero(k, embed_dim); v_Wx = Mat::Zero(k, embed_dim);
        m_Wh = Mat::Zero(k, hidden_dim); v_Wh = Mat::Zero(k, hidden_dim);
        m_L = Mat::Zero(k, k); v_L = Mat::Zero(k, k);
        m_b = Vec::Zero(k); v_b = Vec::Zero(k);
        m_ln_gamma = Vec::Zero(k); v_ln_gamma = Vec::Zero(k);
        m_ln_beta = Vec::Zero(k); v_ln_beta = Vec::Zero(k);
        m_log_lambda = 0.0; v_log_lambda = 0.0;
        m_Wq = Mat::Zero(k, k); v_Wq = Mat::Zero(k, k);
        m_Wout = Mat::Zero(output_dim, k); v_Wout = Mat::Zero(output_dim, k);
        m_collapse_bias = Vec::Zero(output_dim); v_collapse_bias = Vec::Zero(output_dim);
        m_log_temp = 0.0; v_log_temp = 0.0;

        int temporal_cols = raw_input_dim * temporal_kernel;
        m_front_temporal = Mat::Zero(frontend_filters, temporal_cols);
        v_front_temporal = Mat::Zero(frontend_filters, temporal_cols);
        m_front_temporal_b = Vec::Zero(frontend_filters);
        v_front_temporal_b = Vec::Zero(frontend_filters);

        m_front_spatial = Mat::Zero(frontend_filters, frontend_filters);
        v_front_spatial = Mat::Zero(frontend_filters, frontend_filters);
        m_front_spatial_b = Vec::Zero(frontend_filters);
        v_front_spatial_b = Vec::Zero(frontend_filters);

        m_front_depthwise = Mat::Zero(frontend_filters, depth_kernel);
        v_front_depthwise = Mat::Zero(frontend_filters, depth_kernel);
        m_front_depthwise_b = Vec::Zero(frontend_filters);
        v_front_depthwise_b = Vec::Zero(frontend_filters);

        m_front_proj = Mat::Zero(embed_dim, frontend_filters);
        v_front_proj = Mat::Zero(embed_dim, frontend_filters);
        m_front_proj_b = Vec::Zero(embed_dim);
        v_front_proj_b = Vec::Zero(embed_dim);
    }
};

// Sequence trainer with proper BPTT
class SequenceTrainer {
public:
    // Cached computation results for BPTT (public for testing)
    struct SequenceCache {
        std::vector<CellCache> cell_caches;
        std::vector<CollapseCache> collapse_caches;
        std::vector<Vec> psi_history;
        std::vector<Vec> h_history;
        std::vector<Vec> embeddings;
        std::vector<Vec> embed_grads;
        FrontendCache frontend_cache;
        Vec initial_psi, initial_h;
        std::vector<F> predictions;           // scalar predictions (output_dim=1)
        std::vector<Vec> multi_predictions;   // multi-bit predictions (output_dim>1)
    };

private:
    std::unique_ptr<SpatialTemporalCNN> frontend_;
    std::unique_ptr<EntangledCell> cell_;
    std::unique_ptr<Collapse> collapse_;
    std::unique_ptr<AdamW> optimizer_;
    std::unique_ptr<OptimizerState> opt_state_;
    TrainConfig config_;
    int embed_dim_ = 0;
    int raw_input_dim_ = 0;
    
public:
    SequenceTrainer(int k, int raw_input_dim, int embed_dim, int hidden_dim, F lambda, 
                   const TrainConfig& config = TrainConfig{});
    
    // Train on sequence batch with full BPTT
    F train_epoch(const SeqBatch& data);
    
    // Evaluate on sequence batch (no gradient updates)
    F evaluate(const SeqBatch& data, Metrics& metrics);
    
    // Forward pass through a single sequence
    std::vector<F> forward_sequence(const std::vector<Vec>& sequence,
                                   Vec* final_psi = nullptr, Vec* final_h = nullptr,
                                   Vec* final_alpha = nullptr, F* final_temperature = nullptr) const;

    // Forward pass for multi-bit output
    std::vector<Vec> forward_sequence_multi(const std::vector<Vec>& sequence) const;
    
    // Train with proper BPTT backpropagation
    F train_sequence(const std::vector<Vec>& inputs, const std::vector<F>& targets,
                    SequenceCache& cache);

    // Train with multi-bit targets (for modular addition grokking)
    F train_sequence_multi(const std::vector<Vec>& inputs, const std::vector<Vec>& targets,
                          SequenceCache& cache);

    // Backprop through time
    void backward_through_time(const std::vector<Vec>& inputs,
                              const std::vector<F>& targets,
                              SequenceCache& cache,
                              EntangledCell::Grads& cell_grads,
                              FrontendGrads& front_grads,
                              Collapse::Grads& collapse_grads);

    // Backprop for multi-bit targets
    void backward_through_time_multi(const std::vector<Vec>& inputs,
                                    const std::vector<Vec>& targets,
                                    SequenceCache& cache,
                                    EntangledCell::Grads& cell_grads,
                                    FrontendGrads& front_grads,
                                    Collapse::Grads& collapse_grads);
    
    // Apply gradients with regularization
    void apply_gradients(const EntangledCell::Grads& cell_grads,
                        const FrontendGrads& front_grads,
                        const Collapse::Grads& collapse_grads,
                        F reg_loss);
    
    // Compute regularization loss
    F compute_regularization_loss();
    
    // Learning rate control
    void set_learning_rate(F lr) { optimizer_->lr = lr; }
    F get_learning_rate() const { return optimizer_->lr; }
    
    // Getters
    const EntangledCell& get_cell() const { return *cell_; }
    const Collapse& get_collapse() const { return *collapse_; }
    const TrainConfig& get_config() const { return config_; }
};

// Learning rate scheduler integration
class TrainerWithScheduler {
private:
    std::unique_ptr<SequenceTrainer> trainer_;
    std::unique_ptr<CosineScheduler> scheduler_;
    int current_step_ = 0;
    
public:
    TrainerWithScheduler(std::unique_ptr<SequenceTrainer> trainer,
                        F base_lr, F min_lr, int total_steps);
    
    F train_epoch(const SeqBatch& data);
    F evaluate(const SeqBatch& data, Metrics& metrics);
    
    void update_learning_rate();
    F get_current_lr() const;
    
    const SequenceTrainer& get_trainer() const { return *trainer_; }
};

} // namespace enn
