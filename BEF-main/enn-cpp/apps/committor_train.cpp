#include "enn/cell.hpp"
#include "enn/collapse.hpp"
#include "enn/optim.hpp"
#include "enn/data.hpp"
#include "enn/regularizers.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace enn;

int main() {
    // Hyperparameters
    const int k = 64;
    const int input_dim = 2;
    const int hidden_dim = 128;
    const F lambda = 0.1;
    const F lr = 5e-3;
    const int epochs = 100;
    const int batch_size = 256;
    const int n_samples = 10000;
    
    std::cout << "Initializing ENN with k=" << k << ", input_dim=" << input_dim 
              << ", hidden_dim=" << hidden_dim << std::endl;
    
    // Initialize model
    EntangledCell cell(k, input_dim, hidden_dim, lambda);
    Collapse collapse(k);
    
    // Initialize optimizers
    Adam optimizer(lr);
    
    // Optimizer state for cell parameters
    Mat m_Wx = Mat::Zero(k, input_dim), v_Wx = Mat::Zero(k, input_dim);
    Mat m_Wh = Mat::Zero(k, hidden_dim), v_Wh = Mat::Zero(k, hidden_dim);
    Mat m_L = Mat::Zero(k, k), v_L = Mat::Zero(k, k);
    Vec m_b = Vec::Zero(k), v_b = Vec::Zero(k);
    F m_lambda = 0.0, v_lambda = 0.0;
    
    // Optimizer state for collapse parameters
    Mat m_Wg = Mat::Zero(k, k), v_Wg = Mat::Zero(k, k);
    
    // Generate synthetic data
    std::cout << "Generating " << n_samples << " training samples..." << std::endl;
    DataGenerator generator;
    Batch data = generator.generate_double_well_committor(n_samples);
    
    BatchSampler sampler;
    Vec h = Vec::Zero(hidden_dim);  // Hidden state (kept zero for simplicity)
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto batches = sampler.create_batches(data, batch_size, true);
        F epoch_loss = 0.0;
        Metrics metrics;
        
        for (const auto& batch : batches) {
            EntangledCell::Grads cell_grads(k, input_dim, hidden_dim);
            Mat collapse_grads = Mat::Zero(k, k);
            F batch_loss = 0.0;
            
            for (size_t i = 0; i < batch.inputs.size(); ++i) {
                // Forward pass
                Vec psi_in = Vec::Zero(k);  // Start with zero entangled state
                
                CellCache cell_cache;
                Vec psi = cell.forward(batch.inputs[i], h, psi_in, cell_cache);
                
                CollapseCache collapse_cache;
                F pred = collapse.forward(psi, collapse_cache);
                
                // Compute loss (MSE)
                F target = batch.targets[i];
                F loss = 0.5 * (pred - target) * (pred - target);
                batch_loss += loss;
                
                // Update metrics
                metrics.update(pred, target, loss);
                
                // Backward pass
                F dL_dpred = pred - target;
                
                // Collapse backward
                Vec dpsi;
                Mat dWg;
                collapse.backward(dL_dpred, psi, collapse_cache, dpsi, dWg);
                collapse_grads += dWg;
                
                // Cell backward
                Vec dpsi_in_unused, dh_unused;
                cell.backward(dpsi, cell_cache, cell_grads, dpsi_in_unused, dh_unused);
            }
            
            // Scale gradients by batch size
            F scale = 1.0 / batch.inputs.size();
            cell_grads.dWx *= scale;
            cell_grads.dWh *= scale;
            cell_grads.dL *= scale;
            cell_grads.db *= scale;
            cell_grads.dlambda *= scale;
            collapse_grads *= scale;
            
            epoch_loss += batch_loss * scale;
            
            // Apply gradients with Adam
            optimizer.step(cell.Wx, m_Wx, v_Wx, cell_grads.dWx);
            optimizer.step(cell.Wh, m_Wh, v_Wh, cell_grads.dWh);
            optimizer.step(cell.L, m_L, v_L, cell_grads.dL);
            optimizer.step(cell.b, m_b, v_b, cell_grads.db);
            optimizer.step(cell.lambda, m_lambda, v_lambda, cell_grads.dlambda);
            optimizer.step(collapse.Wg, m_Wg, v_Wg, collapse_grads);
        }
        
        metrics.finalize();
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(6) << epoch_loss
                      << " | MSE: " << std::setprecision(6) << metrics.mse
                      << " | MAE: " << std::setprecision(6) << metrics.mae
                      << " | Lambda: " << std::setprecision(4) << cell.lambda
                      << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\nTraining completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Final lambda: " << cell.lambda << std::endl;
    
    // Test PSD property
    bool is_psd = cell.is_entanglement_psd();
    std::cout << "Entanglement matrix is PSD: " << (is_psd ? "Yes" : "No") << std::endl;
    
    // Test on a few examples
    std::cout << "\nTesting on sample points:" << std::endl;
    Vec test_points[] = {
        (Vec(2) << -1.5, 0.0).finished(),  // Should be ~0 (basin A)
        (Vec(2) << 1.5, 0.0).finished(),   // Should be ~1 (basin B)  
        (Vec(2) << 0.0, 0.0).finished()    // Should be ~0.5 (middle)
    };
    
    for (const auto& point : test_points) {
        Vec psi_in = Vec::Zero(k);
        CellCache cache;
        Vec psi = cell.forward(point, h, psi_in, cache);
        
        CollapseCache collapse_cache;
        F pred = collapse.forward(psi, collapse_cache);
        
        F expected = 0.5 * (1.0 + std::tanh(point(0) / 1.5));
        
        std::cout << "Point (" << point(0) << ", " << point(1) << "): "
                  << "Pred=" << std::fixed << std::setprecision(3) << pred
                  << ", Expected=" << std::setprecision(3) << expected
                  << std::endl;
    }
    
    return 0;
}