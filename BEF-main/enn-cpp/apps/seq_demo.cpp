#include "enn/cell.hpp"
#include "enn/collapse.hpp"
#include "enn/optim.hpp"
#include "enn/data.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace enn;

int main() {
    // Hyperparameters for parity task
    const int k = 32;
    const int input_dim = 1;
    const int hidden_dim = 64;
    const F lambda = 0.05;
    const F lr = 1e-3;
    const int epochs = 50;
    const int batch_size = 32;
    const int seq_len = 20;
    
    std::cout << "ENN Sequence Demo: Parity Task" << std::endl;
    std::cout << "Sequence length: " << seq_len << ", Batch size: " << batch_size << std::endl;
    
    // Initialize model
    EntangledCell cell(k, input_dim, hidden_dim, lambda);
    Collapse collapse(k);
    
    // Initialize optimizer
    Adam optimizer(lr);
    
    // Optimizer states
    Mat m_Wx = Mat::Zero(k, input_dim), v_Wx = Mat::Zero(k, input_dim);
    Mat m_Wh = Mat::Zero(k, hidden_dim), v_Wh = Mat::Zero(k, hidden_dim);
    Mat m_L = Mat::Zero(k, k), v_L = Mat::Zero(k, k);
    Vec m_b = Vec::Zero(k), v_b = Vec::Zero(k);
    F m_lambda = 0.0, v_lambda = 0.0;
    Mat m_Wg = Mat::Zero(k, k), v_Wg = Mat::Zero(k, k);
    
    // Generate training data
    DataGenerator generator;
    SeqBatch train_data = generator.generate_parity_task(1000, seq_len);
    SeqBatch test_data = generator.generate_parity_task(200, seq_len);
    
    std::cout << "Generated " << train_data.batch_size() << " training sequences" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        F epoch_loss = 0.0;
        F correct_predictions = 0.0;
        F total_predictions = 0.0;
        
        // Training loop
        for (size_t seq_idx = 0; seq_idx < train_data.batch_size(); seq_idx += batch_size) {
            size_t end_idx = std::min(seq_idx + batch_size, train_data.batch_size());
            
            EntangledCell::Grads cell_grads(k, input_dim, hidden_dim);
            Mat collapse_grads = Mat::Zero(k, k);
            F batch_loss = 0.0;
            
            // Process batch of sequences
            for (size_t b = seq_idx; b < end_idx; ++b) {
                const auto& sequence = train_data.sequences[b];
                const auto& targets = train_data.targets[b];
                
                // BPTT caches for this sequence
                std::vector<CellCache> cell_caches(seq_len);
                std::vector<CollapseCache> collapse_caches(seq_len);
                std::vector<Vec> psi_history(seq_len);
                
                // Forward pass through sequence
                Vec psi = Vec::Zero(k);
                Vec h = Vec::Zero(hidden_dim);
                
                F seq_loss = 0.0;
                
                for (int t = 0; t < seq_len; ++t) {
                    // Forward through cell
                    psi = cell.forward(sequence[t], h, psi, cell_caches[t]);
                    psi_history[t] = psi;
                    
                    // Forward through collapse
                    F pred = collapse.forward(psi, collapse_caches[t]);
                    
                    // Compute loss only on final timestep (or all timesteps)
                    F target = targets[t];
                    F loss = 0.5 * (pred - target) * (pred - target);
                    seq_loss += loss;
                    
                    // Accuracy tracking (final timestep only)
                    if (t == seq_len - 1) {
                        bool pred_bit = pred > 0.5;
                        bool true_bit = target > 0.5;
                        if (pred_bit == true_bit) correct_predictions += 1.0;
                        total_predictions += 1.0;
                    }
                }
                
                batch_loss += seq_loss;
                
                // Backward pass through time (simplified - only final timestep)
                int final_t = seq_len - 1;
                F target = targets[final_t];
                F pred = collapse_caches[final_t].alpha.dot(psi_history[final_t]);
                F dL_dpred = pred - target;
                
                // Collapse backward
                Vec dpsi;
                Mat dWg;
                collapse.backward(dL_dpred, psi_history[final_t], collapse_caches[final_t], dpsi, dWg);
                collapse_grads += dWg;
                
                // Cell backward (only final step for simplicity)
                Vec dpsi_unused, dh_unused;
                cell.backward(dpsi, cell_caches[final_t], cell_grads, dpsi_unused, dh_unused);
            }
            
            // Scale gradients
            F scale = 1.0 / (end_idx - seq_idx);
            cell_grads.dWx *= scale;
            cell_grads.dWh *= scale;
            cell_grads.dL *= scale;
            cell_grads.db *= scale;
            cell_grads.dlambda *= scale;
            collapse_grads *= scale;
            
            epoch_loss += batch_loss * scale;
            
            // Apply gradients
            optimizer.step(cell.Wx, m_Wx, v_Wx, cell_grads.dWx);
            optimizer.step(cell.Wh, m_Wh, v_Wh, cell_grads.dWh);
            optimizer.step(cell.L, m_L, v_L, cell_grads.dL);
            optimizer.step(cell.b, m_b, v_b, cell_grads.db);
            optimizer.step(cell.lambda, m_lambda, v_lambda, cell_grads.dlambda);
            optimizer.step(collapse.Wg, m_Wg, v_Wg, collapse_grads);
        }
        
        F accuracy = correct_predictions / total_predictions;
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(4) << epoch_loss
                      << " | Accuracy: " << std::setprecision(3) << accuracy * 100 << "%"
                      << " | Lambda: " << std::setprecision(4) << cell.lambda
                      << std::endl;
        }
        
        // Early stopping if we achieve good accuracy
        if (accuracy > 0.95) {
            std::cout << "Achieved 95% accuracy, stopping early." << std::endl;
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\nTraining completed in " << duration.count() << " ms" << std::endl;
    
    // Test on held-out data
    std::cout << "\nTesting on " << test_data.batch_size() << " sequences..." << std::endl;
    F test_correct = 0.0;
    F test_total = 0.0;
    
    for (size_t b = 0; b < test_data.batch_size(); ++b) {
        const auto& sequence = test_data.sequences[b];
        const auto& targets = test_data.targets[b];
        
        Vec psi = Vec::Zero(k);
        Vec h = Vec::Zero(hidden_dim);
        
        for (int t = 0; t < seq_len; ++t) {
            CellCache cache;
            psi = cell.forward(sequence[t], h, psi, cache);
            
            if (t == seq_len - 1) {  // Only check final prediction
                CollapseCache collapse_cache;
                F pred = collapse.forward(psi, collapse_cache);
                
                bool pred_bit = pred > 0.5;
                bool true_bit = targets[t] > 0.5;
                if (pred_bit == true_bit) test_correct += 1.0;
                test_total += 1.0;
            }
        }
    }
    
    F test_accuracy = test_correct / test_total;
    std::cout << "Test accuracy: " << std::fixed << std::setprecision(1) 
              << test_accuracy * 100 << "%" << std::endl;
    
    return 0;
}