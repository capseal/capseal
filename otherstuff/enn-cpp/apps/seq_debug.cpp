#include "enn/trainer.hpp"
#include <iostream>
#include <iomanip>

using namespace enn;

int main() {
    std::cout << "=== ENN Sequence Learning Debug ===" << std::endl;
    
    // Test 1: Very simple copy task (much easier than parity)
    std::cout << "\n1. Testing Simple Copy Task [1,0,0] -> [0,0,1]" << std::endl;
    
    TrainConfig config;
    config.learning_rate = 1e-2;  // Higher learning rate
    config.weight_decay = 1e-6;   // Lower weight decay
    config.batch_size = 8;
    config.epochs = 100;
    config.reg_beta = 0.0;        // No regularization initially
    config.reg_eta = 0.0;
    config.verbose = true;
    config.print_every = 10;
    
    // Smaller architecture for debugging
    const int k = 16;
    const int input_dim = 1;
    const int hidden_dim = 32;
    const F lambda = 0.01;  // Much lower lambda
    const int seq_len = 3;
    
    SequenceTrainer trainer(k, input_dim, hidden_dim, lambda, config);
    
    // Create simple copy task: [1, 0, 0] -> [0, 0, 1]
    SeqBatch train_data;
    train_data.sequences.resize(100);  // More training examples
    train_data.targets.resize(100);
    
    for (int i = 0; i < 100; ++i) {
        train_data.sequences[i].resize(seq_len);
        train_data.targets[i].resize(seq_len);
        
        // Input: [1, 0, 0] (marker, then nothing)
        train_data.sequences[i][0] = (Vec(1) << 1.0).finished();
        train_data.sequences[i][1] = (Vec(1) << 0.0).finished();
        train_data.sequences[i][2] = (Vec(1) << 0.0).finished();
        
        // Target: [0, 0, 1] (nothing, nothing, then recall)
        train_data.targets[i][0] = 0.0;
        train_data.targets[i][1] = 0.0;
        train_data.targets[i][2] = 1.0;
    }
    
    // Training loop with detailed monitoring
    F initial_loss = std::numeric_limits<F>::max();
    
    for (int epoch = 1; epoch <= config.epochs; ++epoch) {
        F train_loss = trainer.train_epoch(train_data);
        
        if (epoch == 1) initial_loss = train_loss;
        
        if (epoch % config.print_every == 0) {
            // Test on first sequence
            auto predictions = trainer.forward_sequence(train_data.sequences[0]);
            
            std::cout << "Epoch " << std::setw(3) << epoch
                      << " | Loss: " << std::fixed << std::setprecision(6) << train_loss
                      << " | Lambda: " << std::setprecision(4) << trainer.get_cell().lambda
                      << " | Pred: [" << std::setprecision(3) 
                      << predictions[0] << "," << predictions[1] << "," << predictions[2] << "]"
                      << " | Target: [0.000,0.000,1.000]" << std::endl;
        }
        
        // Early success check
        auto predictions = trainer.forward_sequence(train_data.sequences[0]);
        if (predictions[2] > 0.8 && predictions[0] < 0.2 && predictions[1] < 0.2) {
            std::cout << "SUCCESS! Learned copy task at epoch " << epoch << std::endl;
            break;
        }
    }
    
    std::cout << "\nInitial loss: " << initial_loss << std::endl;
    F final_loss = trainer.train_epoch(train_data);
    std::cout << "Final loss: " << final_loss << std::endl;
    std::cout << "Loss reduction: " << (initial_loss - final_loss) / initial_loss * 100 << "%" << std::endl;
    
    // Test 2: Even simpler - constant prediction
    std::cout << "\n2. Testing Constant Prediction Task" << std::endl;
    
    SequenceTrainer trainer2(8, 1, 16, 0.0, config);  // No decoherence
    
    SeqBatch constant_data;
    constant_data.sequences.resize(50);
    constant_data.targets.resize(50);
    
    for (int i = 0; i < 50; ++i) {
        constant_data.sequences[i].resize(2);
        constant_data.targets[i].resize(2);
        
        // Input: [1, 0]
        constant_data.sequences[i][0] = (Vec(1) << 1.0).finished();
        constant_data.sequences[i][1] = (Vec(1) << 0.0).finished();
        
        // Target: [0.7, 0.7] (constant prediction)
        constant_data.targets[i][0] = 0.7;
        constant_data.targets[i][1] = 0.7;
    }
    
    F const_initial_loss = trainer2.train_epoch(constant_data);
    
    for (int epoch = 1; epoch <= 50; ++epoch) {
        F loss = trainer2.train_epoch(constant_data);
        if (epoch % 10 == 0) {
            auto preds = trainer2.forward_sequence(constant_data.sequences[0]);
            std::cout << "Epoch " << epoch << " | Loss: " << std::setprecision(6) << loss
                      << " | Pred: [" << std::setprecision(3) << preds[0] << "," << preds[1] << "]" << std::endl;
        }
    }
    
    F const_final_loss = trainer2.train_epoch(constant_data);
    std::cout << "Constant task - Initial: " << const_initial_loss << ", Final: " << const_final_loss << std::endl;
    
    // Test 3: Check if gradients are flowing
    std::cout << "\n3. Gradient Flow Analysis" << std::endl;
    
    // Single forward/backward pass
    auto sequence = train_data.sequences[0];
    auto targets = train_data.targets[0];
    
    SequenceTrainer::SequenceCache cache;
    F loss = trainer.train_sequence(sequence, targets, cache);
    
    auto predictions = trainer.forward_sequence(sequence);
    
    EntangledCell::Grads cell_grads(k, input_dim, hidden_dim);
    Mat collapse_grads = Mat::Zero(k, k);
    
    trainer.backward_through_time(targets, predictions, cache, cell_grads, collapse_grads);
    
    std::cout << "Loss: " << loss << std::endl;
    std::cout << "Wx gradient norm: " << cell_grads.dWx.norm() << std::endl;
    std::cout << "b gradient norm: " << cell_grads.db.norm() << std::endl;
    std::cout << "lambda gradient: " << cell_grads.dlambda << std::endl;
    std::cout << "Collapse gradient norm: " << collapse_grads.norm() << std::endl;
    
    if (cell_grads.dWx.norm() < 1e-10) {
        std::cout << "WARNING: Gradients are too small - possible vanishing gradient problem!" << std::endl;
    } else {
        std::cout << "Gradients look healthy." << std::endl;
    }
    
    return 0;
}