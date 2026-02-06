#include "enn/regularizers.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>

namespace enn {

void spectral_clip(Mat& E, F max_eigenvalue) {
    Eigen::SelfAdjointEigenSolver<Mat> solver(E);
    if (solver.info() != Eigen::Success) {
        return; // Failed to compute eigenvalues
    }
    
    Vec eigenvalues = solver.eigenvalues();
    Mat eigenvectors = solver.eigenvectors();
    
    // Clip eigenvalues to [0, max_eigenvalue]
    bool modified = false;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i) < 0) {
            eigenvalues(i) = 0;
            modified = true;
        } else if (eigenvalues(i) > max_eigenvalue) {
            eigenvalues(i) = max_eigenvalue;
            modified = true;
        }
    }
    
    if (modified) {
        E = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
    }
}

F collapse_kl_penalty(const Vec& alpha) {
    // KL(alpha || one_hot) where one_hot puts mass at argmax(alpha)
    int best_idx;
    alpha.maxCoeff(&best_idx);
    
    F kl = 0.0;
    for (int i = 0; i < alpha.size(); ++i) {
        if (alpha(i) > 1e-12) { // Avoid log(0)
            F target = (i == best_idx) ? 1.0 : 0.0;
            if (target > 1e-12) {
                kl += alpha(i) * std::log(alpha(i) / target);
            } else {
                kl += alpha(i) * std::log(alpha(i) / 1e-12); // Regularized
            }
        }
    }
    return kl;
}

F PSDRegularizer::compute_loss_and_grad(const Mat& L, Mat& dL) const {
    // For E = L*L^T parameterization, the PSD constraint is automatically satisfied
    // We can add a small penalty on the Frobenius norm of L to prevent explosion
    F loss = 0.5 * beta * L.squaredNorm();
    dL = beta * L;
    return loss;
}

} // namespace enn