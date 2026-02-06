#include "enn/collapse.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace enn;

bool test_softmax_properties() {
    Collapse collapse(10);
    
    // Test 1: Softmax sums to 1
    Vec z(5);
    z << 1.0, 2.0, 3.0, 4.0, 5.0;
    Vec alpha = collapse.softmax(z);
    
    F sum = alpha.sum();
    if (std::abs(sum - 1.0) > 1e-10) {
        std::cout << "FAIL: Softmax does not sum to 1, got " << sum << std::endl;
        return false;
    }
    
    // Test 2: All elements are positive
    for (int i = 0; i < alpha.size(); ++i) {
        if (alpha(i) <= 0) {
            std::cout << "FAIL: Softmax element " << i << " is not positive: " << alpha(i) << std::endl;
            return false;
        }
    }
    
    // Test 3: Shift invariance (adding constant to all elements)
    Vec z_shifted = z.array() + 10.0;
    Vec alpha_shifted = collapse.softmax(z_shifted);
    
    for (int i = 0; i < alpha.size(); ++i) {
        if (std::abs(alpha(i) - alpha_shifted(i)) > 1e-12) {
            std::cout << "FAIL: Softmax is not shift invariant at element " << i 
                      << ": " << alpha(i) << " vs " << alpha_shifted(i) << std::endl;
            return false;
        }
    }
    
    // Test 4: Large values don't cause overflow
    Vec z_large(3);
    z_large << 700.0, 800.0, 900.0;
    Vec alpha_large = collapse.softmax(z_large);
    
    for (int i = 0; i < alpha_large.size(); ++i) {
        if (!std::isfinite(alpha_large(i))) {
            std::cout << "FAIL: Softmax overflow at element " << i << std::endl;
            return false;
        }
    }
    
    F sum_large = alpha_large.sum();
    if (std::abs(sum_large - 1.0) > 1e-10) {
        std::cout << "FAIL: Softmax with large values does not sum to 1, got " << sum_large << std::endl;
        return false;
    }
    
    return true;
}

bool test_collapse_forward_backward() {
    const int k = 5;
    Collapse collapse(k, 42);
    
    Vec psi = Vec::Random(k);
    CollapseCache cache;
    
    F output = collapse.forward(psi, cache);
    
    // Test that output is reasonable
    if (!std::isfinite(output)) {
        std::cout << "FAIL: Collapse output is not finite" << std::endl;
        return false;
    }
    
    // Test backward pass
    Vec dpsi;
    Mat dWg;
    collapse.backward(1.0, psi, cache, dpsi, dWg);
    
    // Check dimensions
    if (dpsi.size() != k) {
        std::cout << "FAIL: dpsi has wrong size: " << dpsi.size() << " vs " << k << std::endl;
        return false;
    }
    
    if (dWg.rows() != k || dWg.cols() != k) {
        std::cout << "FAIL: dWg has wrong dimensions: " << dWg.rows() << "x" << dWg.cols() 
                  << " vs " << k << "x" << k << std::endl;
        return false;
    }
    
    // Check that gradients are finite
    for (int i = 0; i < dpsi.size(); ++i) {
        if (!std::isfinite(dpsi(i))) {
            std::cout << "FAIL: dpsi element " << i << " is not finite" << std::endl;
            return false;
        }
    }
    
    for (int i = 0; i < dWg.rows(); ++i) {
        for (int j = 0; j < dWg.cols(); ++j) {
            if (!std::isfinite(dWg(i, j))) {
                std::cout << "FAIL: dWg element (" << i << "," << j << ") is not finite" << std::endl;
                return false;
            }
        }
    }
    
    return true;
}

int main() {
    std::cout << "Testing softmax properties..." << std::endl;
    if (!test_softmax_properties()) {
        return 1;
    }
    std::cout << "PASS: Softmax tests" << std::endl;
    
    std::cout << "Testing collapse forward/backward..." << std::endl;
    if (!test_collapse_forward_backward()) {
        return 1;
    }
    std::cout << "PASS: Collapse tests" << std::endl;
    
    std::cout << "All softmax tests passed!" << std::endl;
    return 0;
}