#include "enn/trainer.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace enn;

int main() {
    std::cout << "=== ENN Sequence Demo with Full BPTT ===" << std::endl;
    
    // Training configuration
    TrainConfig config;
    config.learning_rate = 5e-3;
    config.weight_decay = 1e-5;
    config.batch_size = 16;
    config.epochs = 100;
    config.reg_beta = 1e-4;
    config.reg_eta = 1e-6;
    config.verbose = true;
    config.print_every = 10;
    config.bptt_length = -1;  // Full BPTT
    config.accumulate_grads = true;
    
    // Model architecture  
    const int k = 32;
    const int input_dim = 1;
    const int hidden_dim = 64;
    const F lambda = 0.02;
    const int seq_len = 15;
    
    std::cout << "Architecture: k=" << k << ", input_dim=" << input_dim 
              << ", hidden_dim=" << hidden_dim << ", seq_len=" << seq_len << std::endl;
    std::cout << "Training config: lr=" << config.learning_rate 
              << ", batch_size=" << config.batch_size << ", epochs=" << config.epochs << std::endl;
    
    // Create trainer with scheduler
    auto trainer = std::make_unique<SequenceTrainer>(k, input_dim, hidden_dim, lambda, config);
    TrainerWithScheduler scheduled_trainer(std::move(trainer), config.learning_rate, 
                                          config.learning_rate * 0.1, config.epochs);
    
    // Generate training and test data
    std::cout << "\nGenerating parity task data..." << std::endl;
    DataGenerator generator(42);
    SeqBatch train_data = generator.generate_parity_task(800, seq_len);
    SeqBatch test_data = generator.generate_parity_task(200, seq_len);
    
    std::cout << "Train sequences: " << train_data.batch_size() << std::endl;
    std::cout << "Test sequences: " << test_data.batch_size() << std::endl;
    
    // Training loop
    auto start_time = std::chrono::high_resolution_clock::now();
    
    F best_test_acc = 0.0;
    int patience_counter = 0;
    const int patience = 20;
    
    for (int epoch = 1; epoch <= config.epochs; ++epoch) {
        // Training
        F train_loss = scheduled_trainer.train_epoch(train_data);
        
        // Evaluation
        Metrics train_metrics, test_metrics;
        scheduled_trainer.evaluate(train_data, train_metrics);
        F test_loss = scheduled_trainer.evaluate(test_data, test_metrics);
        
        if (epoch % config.print_every == 0) {
            F current_lr = scheduled_trainer.get_current_lr();
            const auto& cell = scheduled_trainer.get_trainer().get_cell();
            
            std::cout << "Epoch " << std::setw(3) << epoch
                      << " | Train Loss: " << std::fixed << std::setprecision(4) << train_loss
                      << " | Test Loss: " << std::setprecision(4) << test_loss  
                      << " | Test Acc: " << std::setprecision(1) << test_metrics.accuracy * 100 << "%"
                      << " | LR: " << std::setprecision(2) << std::scientific << current_lr
                      << " | λ: " << std::fixed << std::setprecision(4) << cell.lambda
                      << std::endl;
        }
        
        // Early stopping
        if (test_metrics.accuracy > best_test_acc) {
            best_test_acc = test_metrics.accuracy;
            patience_counter = 0;
        } else {
            patience_counter++;
        }
        
        if (test_metrics.accuracy > 0.90) {
            std::cout << "Achieved 90% accuracy! Stopping early." << std::endl;
            break;
        }
        
        if (patience_counter >= patience) {
            std::cout << "Early stopping due to no improvement." << std::endl;
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\n=== Final Results ===" << std::endl;
    std::cout << "Training completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Best test accuracy: " << std::fixed << std::setprecision(1) 
              << best_test_acc * 100 << "%" << std::endl;
    
    // Final evaluation
    Metrics final_metrics;
    F final_loss = scheduled_trainer.evaluate(test_data, final_metrics);
    
    std::cout << "Final test metrics:" << std::endl;
    std::cout << "  Loss: " << std::setprecision(6) << final_loss << std::endl;
    std::cout << "  Accuracy: " << std::setprecision(1) << final_metrics.accuracy * 100 << "%" << std::endl;
    std::cout << "  MSE: " << std::setprecision(6) << final_metrics.mse << std::endl;
    std::cout << "  MAE: " << std::setprecision(6) << final_metrics.mae << std::endl;
    
    // Test on a few specific sequences
    std::cout << "\n=== Sample Predictions ===" << std::endl;
    for (int i = 0; i < std::min(5, static_cast<int>(test_data.batch_size())); ++i) {
        const auto& sequence = test_data.sequences[i];
        const auto& targets = test_data.targets[i];
        
        auto predictions = scheduled_trainer.get_trainer().forward_sequence(sequence);
        
        std::cout << "Sequence " << i << ":" << std::endl;
        std::cout << "  Input:  ";
        for (int t = 0; t < seq_len; ++t) {
            std::cout << static_cast<int>(sequence[t](0)) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "  Target: ";
        for (int t = 0; t < seq_len; ++t) {
            std::cout << static_cast<int>(targets[t]) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "  Pred:   ";
        for (int t = 0; t < seq_len; ++t) {
            std::cout << (predictions[t] > 0.5 ? 1 : 0) << " ";
        }
        std::cout << std::endl;
        
        // Check final prediction correctness
        bool correct = (predictions.back() > 0.5) == (targets.back() > 0.5);
        std::cout << "  Final: " << (correct ? "✓" : "✗") 
                  << " (pred=" << std::setprecision(3) << predictions.back()
                  << ", target=" << targets.back() << ")" << std::endl << std::endl;
    }
    
    // Analyze entanglement properties
    const auto& cell = scheduled_trainer.get_trainer().get_cell();
    Mat E = cell.get_entanglement_matrix();
    
    std::cout << "=== Model Analysis ===" << std::endl;
    std::cout << "Entanglement matrix is PSD: " << (cell.is_entanglement_psd() ? "Yes" : "No") << std::endl;
    std::cout << "E matrix trace: " << std::setprecision(4) << E.trace() << std::endl;
    std::cout << "E matrix determinant: " << std::setprecision(2) << std::scientific << E.determinant() << std::endl;
    std::cout << "Final lambda: " << std::fixed << std::setprecision(4) << cell.lambda << std::endl;
    
    return 0;
}