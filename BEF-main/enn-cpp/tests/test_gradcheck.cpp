#include "enn/cell.hpp"
#include "enn/collapse.hpp"
#include <iostream>
#include <cmath>

using namespace enn;

bool gradient_check(const std::function<F()>& f, const std::function<void(F)>& update_param,
                   F analytical_grad, const std::string& param_name, F h = 1e-6) {
    // Central difference: (f(x+h) - f(x-h)) / (2*h)
    update_param(h);
    F f_plus = f();
    
    update_param(-2*h);
    F f_minus = f();
    
    update_param(h);  // Restore original value
    
    F numerical_grad = (f_plus - f_minus) / (2.0 * h);
    F error = std::abs(analytical_grad - numerical_grad);
    F rel_error = error / (std::abs(numerical_grad) + 1e-8);
    
    bool passed = rel_error < 1e-4;
    
    std::cout << param_name << ": analytical=" << analytical_grad 
              << ", numerical=" << numerical_grad 
              << ", rel_error=" << rel_error;
    if (passed) {
        std::cout << " PASS" << std::endl;
    } else {
        std::cout << " FAIL" << std::endl;
    }
    
    return passed;
}

bool test_cell_gradients() {
    const int k = 4;
    const int input_dim = 2;  
    const int hidden_dim = 3;
    
    EntangledCell cell(k, input_dim, hidden_dim, 0.1, 42);
    
    // Test inputs
    Vec x = Vec::Random(input_dim);
    Vec h = Vec::Random(hidden_dim);
    Vec psi_in = Vec::Random(k);
    Vec target = Vec::Random(k);
    
    // Forward and backward pass
    auto compute_loss = [&]() -> F {
        CellCache cache;
        Vec psi = cell.forward(x, h, psi_in, cache);
        return 0.5 * (psi - target).squaredNorm();
    };
    
    CellCache cache;
    Vec psi = cell.forward(x, h, psi_in, cache);
    Vec dpsi = psi - target;  // dL/dpsi
    
    EntangledCell::Grads grads(k, input_dim, hidden_dim);
    Vec dpsi_in, dh;
    cell.backward(dpsi, cache, grads, dpsi_in, dh);
    
    bool all_passed = true;
    
    // Test a few elements of Wx
    for (int i = 0; i < std::min(k, 2); ++i) {
        for (int j = 0; j < std::min(input_dim, 2); ++j) {
            auto update_Wx = [&](F delta) { cell.Wx(i, j) += delta; };
            std::string param_name = "Wx(" + std::to_string(i) + "," + std::to_string(j) + ")";
            all_passed &= gradient_check(compute_loss, update_Wx, grads.dWx(i, j), param_name);
        }
    }
    
    // Test a few elements of b
    for (int i = 0; i < std::min(k, 2); ++i) {
        auto update_b = [&](F delta) { cell.b(i) += delta; };
        std::string param_name = "b(" + std::to_string(i) + ")";
        all_passed &= gradient_check(compute_loss, update_b, grads.db(i), param_name);
    }
    
    // Test lambda
    auto update_lambda = [&](F delta) { cell.lambda += delta; };
    all_passed &= gradient_check(compute_loss, update_lambda, grads.dlambda, "lambda");
    
    return all_passed;
}

bool test_collapse_gradients() {
    const int k = 4;
    Collapse collapse(k, 123);
    
    Vec psi = Vec::Random(k);
    F target = 0.7;  // Target scalar
    
    auto compute_loss = [&]() -> F {
        CollapseCache cache;
        F pred = collapse.forward(psi, cache);
        return 0.5 * (pred - target) * (pred - target);
    };
    
    CollapseCache cache;
    F pred = collapse.forward(psi, cache);
    F dL_dpred = pred - target;
    
    Vec dpsi;
    Mat dWg;
    collapse.backward(dL_dpred, psi, cache, dpsi, dWg);
    
    bool all_passed = true;
    
    // Test a few elements of Wg
    for (int i = 0; i < std::min(k, 2); ++i) {
        for (int j = 0; j < std::min(k, 2); ++j) {
            auto update_Wg = [&](F delta) { collapse.Wg(i, j) += delta; };
            std::string param_name = "Wg(" + std::to_string(i) + "," + std::to_string(j) + ")";
            all_passed &= gradient_check(compute_loss, update_Wg, dWg(i, j), param_name);
        }
    }
    
    return all_passed;
}

int main() {
    std::cout << "Testing cell gradients..." << std::endl;
    if (!test_cell_gradients()) {
        std::cout << "FAIL: Cell gradient check failed" << std::endl;
        return 1;
    }
    std::cout << "PASS: Cell gradient tests\n" << std::endl;
    
    std::cout << "Testing collapse gradients..." << std::endl;
    if (!test_collapse_gradients()) {
        std::cout << "FAIL: Collapse gradient check failed" << std::endl;
        return 1;
    }
    std::cout << "PASS: Collapse gradient tests" << std::endl;
    
    std::cout << "\nAll gradient checks passed!" << std::endl;
    return 0;
}