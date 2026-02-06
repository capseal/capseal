#pragma once
#include "cell.hpp"
#include "collapse.hpp"
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
    
    // BPTT settings
    int bptt_length = -1;     // -1 = full sequence, >0 = truncated BPTT
    bool accumulate_grads = true;  // Accumulate gradients across timesteps
};

// Optimizer state container
struct OptimizerState {
    // Cell parameter states
    Mat m_Wx, v_Wx, m_Wh, v_Wh, m_L, v_L;
    Vec m_b, v_b;
    F m_lambda = 0, v_lambda = 0;
    
    // Collapse parameter states  
    Mat m_Wg, v_Wg;
    
    OptimizerState(int k, int input_dim, int hidden_dim) {
        m_Wx = Mat::Zero(k, input_dim); v_Wx = Mat::Zero(k, input_dim);
        m_Wh = Mat::Zero(k, hidden_dim); v_Wh = Mat::Zero(k, hidden_dim);
        m_L = Mat::Zero(k, k); v_L = Mat::Zero(k, k);
        m_b = Vec::Zero(k); v_b = Vec::Zero(k);
        m_Wg = Mat::Zero(k, k); v_Wg = Mat::Zero(k, k);
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
        Vec initial_psi, initial_h;
    };

private:
    std::unique_ptr<EntangledCell> cell_;
    std::unique_ptr<Collapse> collapse_;
    std::unique_ptr<AdamW> optimizer_;
    std::unique_ptr<OptimizerState> opt_state_;
    TrainConfig config_;
    
public:
    SequenceTrainer(int k, int input_dim, int hidden_dim, F lambda, 
                   const TrainConfig& config = TrainConfig{});
    
    // Train on sequence batch with full BPTT
    F train_epoch(const SeqBatch& data);
    
    // Evaluate on sequence batch (no gradient updates)
    F evaluate(const SeqBatch& data, Metrics& metrics);
    
    // Forward pass through a single sequence
    std::vector<F> forward_sequence(const std::vector<Vec>& sequence, 
                                   Vec* final_psi = nullptr, Vec* final_h = nullptr) const;
    
    // Train with proper BPTT backpropagation
    F train_sequence(const std::vector<Vec>& inputs, const std::vector<F>& targets,
                    SequenceCache& cache);
    
    // Backprop through time  
    void backward_through_time(const std::vector<F>& targets,
                              const std::vector<F>& predictions,
                              const SequenceCache& cache,
                              EntangledCell::Grads& cell_grads,
                              Mat& collapse_grads);
    
    // Apply gradients with regularization
    void apply_gradients(const EntangledCell::Grads& cell_grads,
                        const Mat& collapse_grads, F reg_loss);
    
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

// Batch trainer for non-sequential data
class BatchTrainer {
private:
    std::unique_ptr<EntangledCell> cell_;
    std::unique_ptr<Collapse> collapse_;
    std::unique_ptr<AdamW> optimizer_;
    std::unique_ptr<OptimizerState> opt_state_;
    TrainConfig config_;
    
public:
    BatchTrainer(int k, int input_dim, int hidden_dim, F lambda,
                const TrainConfig& config = TrainConfig{});
    
    // Train on batch data
    F train_epoch(const Batch& data);
    
    // Evaluate on batch data
    F evaluate(const Batch& data, Metrics& metrics);
    
    // Single forward pass
    F forward(const Vec& input, Vec* psi_out = nullptr);
    
    // Apply gradients
    void apply_gradients(const EntangledCell::Grads& cell_grads,
                        const Mat& collapse_grads, F reg_loss);
    
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