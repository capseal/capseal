#include "enn/trainer.hpp"
#include <iostream>
#include <cmath>
#include <random>
#include <iomanip>

using namespace enn;

bool gradient_check_sequence(const std::function<F()>& f, const std::function<void(F)>& update_param,
                            F analytical_grad, const std::string& param_name, F h = 1e-5) {
    // Central difference for sequences (need larger h due to accumulated errors)
    update_param(h);
    F f_plus = f();
    
    update_param(-2*h);
    F f_minus = f();
    
    update_param(h);  // Restore
    
    F numerical_grad = (f_plus - f_minus) / (2.0 * h);
    F error = std::abs(analytical_grad - numerical_grad);
    F rel_error = error / (std::abs(numerical_grad) + 1e-8);
    
    bool passed = rel_error < 1e-3;  // Looser tolerance for BPTT
    
    std::cout << param_name << ": analytical=" << std::setprecision(6) << analytical_grad 
              << ", numerical=" << std::setprecision(6) << numerical_grad 
              << ", rel_error=" << std::setprecision(2) << std::scientific << rel_error;
    if (passed) {
        std::cout << " PASS" << std::endl;
    } else {
        std::cout << " FAIL" << std::endl;
    }
    
    return passed;
}

bool test_bptt_gradients() {
    std::cout << "Testing BPTT gradients..." << std::endl;
    
    // Small model for testing
    const int k = 3;
    const int input_dim = 2;
    const int hidden_dim = 4;
    const int seq_len = 3;
    
    TrainConfig config;
    config.learning_rate = 1e-3;
    config.batch_size = 1;
    config.reg_eta = 0.0;  // No regularization for clean gradients
    config.reg_beta = 0.0;
    
    SequenceTrainer trainer(k, input_dim, hidden_dim, 0.1, config);
    
    // Create a simple test sequence
    std::vector<Vec> inputs;
    std::vector<F> targets;
    
    std::mt19937 gen(123);
    std::normal_distribution<F> dist(0.0, 0.5);
    
    for (int t = 0; t < seq_len; ++t) {
        Vec input(input_dim);
        input << dist(gen), dist(gen);
        inputs.push_back(input);
        targets.push_back(0.5 + 0.3 * std::sin(t));  // Simple target pattern
    }
    
    // Compute loss and gradients via BPTT
    auto compute_loss = [&]() -> F {
        auto predictions = trainer.forward_sequence(inputs);
        F loss = 0.0;
        for (int t = 0; t < seq_len; ++t) {
            loss += 0.5 * (predictions[t] - targets[t]) * (predictions[t] - targets[t]);
        }
        return loss;
    };
    
    // Forward pass to get gradients
    typename SequenceTrainer::SequenceCache cache;
    F initial_loss = trainer.train_sequence(inputs, targets, cache);
    
    // Get predictions for backward pass
    std::vector<F> predictions = trainer.forward_sequence(inputs);
    
    // Do backward pass to get analytical gradients
    EntangledCell::Grads cell_grads(k, input_dim, hidden_dim);
    Mat collapse_grads = Mat::Zero(k, k);
    trainer.backward_through_time(targets, predictions, cache, cell_grads, collapse_grads);
    
    bool all_passed = true;
    
    // Test cell parameter gradients
    const auto& cell = trainer.get_cell();
    
    // Test a few Wx elements
    for (int i = 0; i < std::min(k, 2); ++i) {
        for (int j = 0; j < std::min(input_dim, 2); ++j) {
            auto update_Wx = [&](F delta) { 
                const_cast<EntangledCell&>(cell).Wx(i, j) += delta; 
            };
            std::string param_name = "Wx(" + std::to_string(i) + "," + std::to_string(j) + ")";
            all_passed &= gradient_check_sequence(compute_loss, update_Wx, 
                                                 cell_grads.dWx(i, j), param_name);
        }
    }
    
    // Test bias gradients
    for (int i = 0; i < std::min(k, 2); ++i) {
        auto update_b = [&](F delta) { 
            const_cast<EntangledCell&>(cell).b(i) += delta; 
        };
        std::string param_name = "b(" + std::to_string(i) + ")";
        all_passed &= gradient_check_sequence(compute_loss, update_b, 
                                             cell_grads.db(i), param_name);
    }
    
    // Test lambda
    auto update_lambda = [&](F delta) { 
        const_cast<EntangledCell&>(cell).lambda += delta; 
    };
    all_passed &= gradient_check_sequence(compute_loss, update_lambda, 
                                         cell_grads.dlambda, "lambda");
    
    // Test collapse gradients
    const auto& collapse = trainer.get_collapse();
    for (int i = 0; i < std::min(k, 2); ++i) {
        for (int j = 0; j < std::min(k, 2); ++j) {
            auto update_Wg = [&](F delta) { 
                const_cast<Collapse&>(collapse).Wg(i, j) += delta; 
            };
            std::string param_name = "Wg(" + std::to_string(i) + "," + std::to_string(j) + ")";
            all_passed &= gradient_check_sequence(compute_loss, update_Wg, 
                                                 collapse_grads(i, j), param_name);
        }
    }
    
    return all_passed;
}

bool test_simple_sequence_learning() {
    std::cout << "\nTesting simple sequence learning..." << std::endl;
    
    TrainConfig config;
    config.learning_rate = 1e-2;
    config.batch_size = 4;
    config.epochs = 50;
    config.verbose = false;
    
    SequenceTrainer trainer(8, 1, 16, 0.05, config);
    
    // Generate simple copy task: [1, 0] -> [0, 0, 1, 0]
    SeqBatch train_data;
    train_data.sequences.resize(20);
    train_data.targets.resize(20);
    
    for (int i = 0; i < 20; ++i) {
        train_data.sequences[i].resize(4);
        train_data.targets[i].resize(4);
        
        // Input: [1, 0, 0, 0] (marker at start)
        train_data.sequences[i][0] = (Vec(1) << 1.0).finished();
        train_data.sequences[i][1] = (Vec(1) << 0.0).finished();
        train_data.sequences[i][2] = (Vec(1) << 0.0).finished();
        train_data.sequences[i][3] = (Vec(1) << 0.0).finished();
        
        // Target: [0, 0, 1, 0] (recall marker with delay)
        train_data.targets[i][0] = 0.0;
        train_data.targets[i][1] = 0.0;
        train_data.targets[i][2] = 1.0;
        train_data.targets[i][3] = 0.0;
    }
    
    // Train for a few epochs
    F initial_loss = std::numeric_limits<F>::max();
    F final_loss = 0.0;
    
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        F loss = trainer.train_epoch(train_data);
        if (epoch == 0) initial_loss = loss;
        if (epoch == config.epochs - 1) final_loss = loss;
    }
    
    std::cout << "Initial loss: " << std::setprecision(4) << initial_loss << std::endl;
    std::cout << "Final loss: " << std::setprecision(4) << final_loss << std::endl;
    
    // Check that learning occurred
    bool learning_occurred = final_loss < initial_loss * 0.8;
    std::cout << "Learning occurred: " << (learning_occurred ? "Yes" : "No") << std::endl;
    
    return learning_occurred;
}

int main() {
    std::cout << "=== BPTT Gradient Check Tests ===" << std::endl;
    
    if (!test_bptt_gradients()) {
        std::cout << "FAIL: BPTT gradient check failed" << std::endl;
        return 1;
    }
    std::cout << "PASS: BPTT gradient tests" << std::endl;
    
    if (!test_simple_sequence_learning()) {
        std::cout << "FAIL: Simple sequence learning test failed" << std::endl;
        return 1;
    }
    std::cout << "PASS: Simple sequence learning test" << std::endl;
    
    std::cout << "\nAll BPTT tests passed!" << std::endl;
    return 0;
}