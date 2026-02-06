#include "enn/data.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <numeric>

namespace enn {

Batch DataGenerator::generate_double_well_committor(int n_samples, F a, F b,
                                                   F x_range, F y_range) {
    Batch batch;
    batch.inputs.reserve(n_samples);
    batch.targets.reserve(n_samples);
    
    std::uniform_real_distribution<F> x_dist(-x_range/2, x_range/2);
    std::uniform_real_distribution<F> y_dist(-y_range/2, y_range/2);
    
    for (int i = 0; i < n_samples; ++i) {
        F x = x_dist(gen_);
        F y = y_dist(gen_);
        
        // Input is 2D position
        Vec input(2);
        input << x, y;
        batch.inputs.push_back(input);
        
        // Committor function: probability of hitting x>b before x<a
        // Simple approximation: sigmoid-like function based on x position
        F committor = 0.5 * (1.0 + std::tanh((x - (a+b)/2) / ((b-a)/4)));
        batch.targets.push_back(committor);
    }
    
    return batch;
}

SeqBatch DataGenerator::generate_copy_task(int batch_size, int seq_len, int vocab_size) {
    SeqBatch batch;
    batch.sequences.resize(batch_size);
    batch.targets.resize(batch_size);
    
    std::uniform_int_distribution<int> vocab_dist(0, vocab_size - 1);
    
    for (int b = 0; b < batch_size; ++b) {
        batch.sequences[b].reserve(seq_len);
        batch.targets[b].reserve(seq_len);
        
        // Generate random sequence
        std::vector<int> sequence(seq_len/2);
        for (int i = 0; i < seq_len/2; ++i) {
            sequence[i] = vocab_dist(gen_);
        }
        
        // First half: input sequence, target is 0 (no output)
        for (int i = 0; i < seq_len/2; ++i) {
            Vec input(vocab_size);
            input.setZero();
            input(sequence[i]) = 1.0;  // One-hot encoding
            
            batch.sequences[b].push_back(input);
            batch.targets[b].push_back(0.0);
        }
        
        // Second half: recall sequence, target is the original input
        for (int i = 0; i < seq_len/2; ++i) {
            Vec input = Vec::Zero(vocab_size);  // No input
            
            batch.sequences[b].push_back(input);
            batch.targets[b].push_back(static_cast<F>(sequence[i]) / (vocab_size - 1));
        }
    }
    
    return batch;
}

SeqBatch DataGenerator::generate_parity_task(int batch_size, int seq_len) {
    SeqBatch batch;
    batch.sequences.resize(batch_size);
    batch.targets.resize(batch_size);
    
    std::bernoulli_distribution binary_dist(0.5);
    
    for (int b = 0; b < batch_size; ++b) {
        batch.sequences[b].reserve(seq_len);
        batch.targets[b].reserve(seq_len);
        
        int parity = 0;
        
        for (int t = 0; t < seq_len; ++t) {
            int bit = binary_dist(gen_) ? 1 : 0;
            parity ^= bit;  // XOR for parity
            
            Vec input(1);
            input << static_cast<F>(bit);
            
            batch.sequences[b].push_back(input);
            batch.targets[b].push_back(static_cast<F>(parity));
        }
    }
    
    return batch;
}

SeqBatch DataGenerator::generate_adding_task(int batch_size, int seq_len) {
    SeqBatch batch;
    batch.sequences.resize(batch_size);
    batch.targets.resize(batch_size);
    
    std::uniform_real_distribution<F> value_dist(0.0, 1.0);
    std::bernoulli_distribution marker_dist(2.0 / seq_len);  // ~2 markers per sequence
    
    for (int b = 0; b < batch_size; ++b) {
        batch.sequences[b].reserve(seq_len);
        batch.targets[b].reserve(seq_len);
        
        F sum = 0.0;
        int markers_placed = 0;
        
        for (int t = 0; t < seq_len; ++t) {
            F value = value_dist(gen_);
            bool marker = (markers_placed < 2) && marker_dist(gen_);
            
            if (marker) {
                sum += value;
                markers_placed++;
            }
            
            Vec input(2);
            input << value, marker ? 1.0 : 0.0;
            
            batch.sequences[b].push_back(input);
            
            // Target is the sum only at the final timestep
            F target = (t == seq_len - 1) ? sum : 0.0;
            batch.targets[b].push_back(target);
        }
    }
    
    return batch;
}

SeqBatch DataGenerator::generate_ou_process(int batch_size, int seq_len, F theta, F mu, F sigma, F dt) {
    SeqBatch batch;
    batch.sequences.resize(batch_size);
    batch.targets.resize(batch_size);
    
    for (int b = 0; b < batch_size; ++b) {
        batch.sequences[b].reserve(seq_len);
        batch.targets[b].reserve(seq_len);
        
        F x = mu + normal_(gen_) * sigma / std::sqrt(2 * theta);  // Stationary initial condition
        
        for (int t = 0; t < seq_len; ++t) {
            Vec input(1);
            input << x;
            
            batch.sequences[b].push_back(input);
            
            // Predict next value (for autoregressive modeling)
            F next_x = x + theta * (mu - x) * dt + sigma * std::sqrt(dt) * normal_(gen_);
            batch.targets[b].push_back(next_x);
            
            x = next_x;
        }
    }
    
    return batch;
}

// Simple CSV loader implementation
Batch DataLoader::load_csv(const std::string& filename) {
    Batch batch;
    std::ifstream file(filename);
    std::string line;
    
    if (!std::getline(file, line)) {
        throw std::runtime_error("Could not read header from " + filename);
    }
    
    // Count columns from header
    std::istringstream header_stream(line);
    std::string column;
    int n_cols = 0;
    while (std::getline(header_stream, column, ',')) {
        n_cols++;
    }
    
    int input_dim = n_cols - 1;  // Last column is target
    
    while (std::getline(file, line)) {
        std::istringstream line_stream(line);
        std::string cell;
        
        Vec input(input_dim);
        int col = 0;
        
        while (std::getline(line_stream, cell, ',')) {
            F value = std::stod(cell);
            
            if (col < input_dim) {
                input(col) = value;
            } else {
                batch.targets.push_back(value);
            }
            col++;
        }
        
        batch.inputs.push_back(input);
    }
    
    return batch;
}

void DataLoader::save_csv(const Batch& batch, const std::string& filename) {
    if (batch.inputs.empty()) return;
    
    std::ofstream file(filename);
    
    // Write header
    int input_dim = batch.inputs[0].size();
    for (int i = 0; i < input_dim; ++i) {
        file << "x" << i;
        if (i < input_dim - 1) file << ",";
    }
    file << ",target\n";
    
    // Write data
    for (size_t i = 0; i < batch.inputs.size(); ++i) {
        const Vec& input = batch.inputs[i];
        for (int j = 0; j < input.size(); ++j) {
            file << input(j) << ",";
        }
        file << batch.targets[i] << "\n";
    }
}

std::vector<Batch> BatchSampler::create_batches(const Batch& data, int batch_size, bool shuffle) {
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    if (shuffle) {
        std::shuffle(indices.begin(), indices.end(), gen_);
    }
    
    std::vector<Batch> batches;
    for (size_t i = 0; i < indices.size(); i += batch_size) {
        Batch batch;
        size_t end_idx = std::min(i + batch_size, indices.size());
        
        for (size_t j = i; j < end_idx; ++j) {
            size_t idx = indices[j];
            batch.inputs.push_back(data.inputs[idx]);
            batch.targets.push_back(data.targets[idx]);
        }
        
        batches.push_back(batch);
    }
    
    return batches;
}

} // namespace enn