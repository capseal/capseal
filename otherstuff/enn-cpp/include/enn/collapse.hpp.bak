#pragma once
#include "types.hpp"

namespace enn {

struct CollapseCache { 
    Vec logits; 
    Vec alpha; 
    
    CollapseCache() = default;
    CollapseCache(int k) : logits(Vec::Zero(k)), alpha(Vec::Zero(k)) {}
};

struct Collapse {
    Mat Wg;  // [k x k] attention weights
    int k;   // entanglement dimension
    
    explicit Collapse(int k_, unsigned seed = 123);
    
    // Numerically stable softmax
    Vec softmax(const Vec& z) const;
    
    // Forward pass: alpha = softmax(Wg * psi), output = alpha^T * psi
    F forward(const Vec& psi, CollapseCache& cache) const;
    
    // Backward pass: given dL/dpred, compute dL/dpsi and dL/dWg
    // Uses Jacobian J = diag(alpha) - alpha * alpha^T
    void backward(F dL_dpred, const Vec& psi, const CollapseCache& cache, 
                  Vec& dpsi, Mat& dWg) const;
    
    // Multi-head collapse (optional extension)
    struct MultiHead {
        std::vector<Mat> heads;  // [num_heads][k x k]
        Mat Wout;               // [1 x num_heads] combination weights
        
        MultiHead(int k, int num_heads, unsigned seed = 456);
        F forward(const Vec& psi, std::vector<CollapseCache>& caches) const;
        void backward(F dL_dpred, const Vec& psi, 
                      const std::vector<CollapseCache>& caches,
                      Vec& dpsi, std::vector<Mat>& dheads, Mat& dWout) const;
    };
};

} // namespace enn