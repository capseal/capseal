#pragma once
#include "types.hpp"

namespace enn {

// Exact PSD constraint via Cholesky factorization: E = L * L^T
inline Mat psd_from_factor(const Mat& L) { 
    return L * L.transpose(); 
}

// Symmetry penalty for unconstrained E (fallback method)
inline F sym_penalty(const Mat& E) { 
    return (E - E.transpose()).squaredNorm(); 
}

// Spectral clipping to prevent E from becoming too large
void spectral_clip(Mat& E, F max_eigenvalue);

// KL divergence penalty to encourage decisive collapse
// KL(alpha || one_hot) where one_hot has mass at argmax(alpha)
F collapse_kl_penalty(const Vec& alpha);

// L2 penalty on psi to prevent explosion
inline F psi_l2_penalty(const Vec& psi) {
    return psi.squaredNorm();
}

// Compute PSD regularization loss and gradient
struct PSDRegularizer {
    F beta;  // regularization strength
    
    explicit PSDRegularizer(F beta_ = 1e-3) : beta(beta_) {}
    
    // For E = L*L^T parameterization, returns loss and dL/dL
    F compute_loss_and_grad(const Mat& L, Mat& dL) const;
};

} // namespace enn