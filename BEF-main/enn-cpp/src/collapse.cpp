#include "enn/collapse.hpp"
#include <random>
#include <algorithm>
#include <cmath>

namespace enn {

Collapse::Collapse(int k_, int output_dim_, unsigned seed) : k(k_), output_dim(output_dim_) {
    std::mt19937 gen(seed);
    std::normal_distribution<F> dist(0.0, 0.05);

    Wq = Mat::NullaryExpr(k, k, [&]() { return dist(gen); });
    Wout = Mat::NullaryExpr(output_dim, k, [&]() { return dist(gen); });
    bout = Vec::Zero(output_dim);
    log_temp = std::log(1.0);
}

Vec Collapse::softmax(const Vec& z) const {
    F max_z = z.maxCoeff();
    Vec exp_z = (z.array() - max_z).exp();
    return exp_z / exp_z.sum();
}

Vec Collapse::softmax_jacobian_matvec(const Vec& alpha, const Vec& vec) const {
    F dot = alpha.dot(vec);
    return alpha.cwiseProduct(vec - Vec::Constant(alpha.size(), dot));
}

F Collapse::forward(const Vec& psi, CollapseCache& cache) const {
    cache.gated = Wq * psi;
    cache.scores = cache.gated.cwiseProduct(psi);
    F temp = std::exp(log_temp);
    cache.temperature = temp;
    Vec scaled = cache.scores / temp;
    cache.alpha = softmax(scaled);
    cache.collapsed = cache.alpha.cwiseProduct(psi);
    // For scalar output (output_dim=1), return first element
    return Wout.row(0).dot(cache.collapsed) + bout(0);
}

Vec Collapse::forward_multi(const Vec& psi, CollapseCache& cache) const {
    cache.gated = Wq * psi;
    cache.scores = cache.gated.cwiseProduct(psi);
    F temp = std::exp(log_temp);
    cache.temperature = temp;
    Vec scaled = cache.scores / temp;
    cache.alpha = softmax(scaled);
    cache.collapsed = cache.alpha.cwiseProduct(psi);
    // Multi-bit output: Wout is [output_dim x k], returns [output_dim]
    return Wout * cache.collapsed + bout;
}

void Collapse::backward(F dL_dpred, const Vec& psi, const CollapseCache& cache,
                        Vec& dpsi, Grads& grads) const {
    // Scalar backward (output_dim=1): use first row of Wout
    Vec dcollapsed = dL_dpred * Wout.row(0).transpose();
    grads.dWout.row(0) += dL_dpred * cache.collapsed.transpose();
    grads.dbias(0) += dL_dpred;

    Vec dpsi_total = dcollapsed.cwiseProduct(cache.alpha);
    Vec dalpha = dcollapsed.cwiseProduct(psi);

    Vec dz = softmax_jacobian_matvec(cache.alpha, dalpha);
    F temp = cache.temperature;
    Vec dscores = dz / temp;
    grads.dlog_temp += -(dz.dot(cache.scores)) / temp;

    Vec psi_weight = dscores.cwiseProduct(psi);
    grads.dWq += psi_weight * psi.transpose();

    Vec term_from_u = dscores.cwiseProduct(cache.gated);
    Vec term_from_mat = Wq.transpose() * psi_weight;
    dpsi = dpsi_total + term_from_u + term_from_mat;
}

void Collapse::backward_multi(const Vec& dL_dpred, const Vec& psi, const CollapseCache& cache,
                              Vec& dpsi, Grads& grads) const {
    // Multi-bit backward: dL_dpred is [output_dim], Wout is [output_dim x k]
    // dcollapsed = Wout^T * dL_dpred, shape [k]
    Vec dcollapsed = Wout.transpose() * dL_dpred;

    // Gradient for Wout: outer product dL_dpred * collapsed^T
    grads.dWout += dL_dpred * cache.collapsed.transpose();
    grads.dbias += dL_dpred;

    Vec dpsi_total = dcollapsed.cwiseProduct(cache.alpha);
    Vec dalpha = dcollapsed.cwiseProduct(psi);

    Vec dz = softmax_jacobian_matvec(cache.alpha, dalpha);
    F temp = cache.temperature;
    Vec dscores = dz / temp;
    grads.dlog_temp += -(dz.dot(cache.scores)) / temp;

    Vec psi_weight = dscores.cwiseProduct(psi);
    grads.dWq += psi_weight * psi.transpose();

    Vec term_from_u = dscores.cwiseProduct(cache.gated);
    Vec term_from_mat = Wq.transpose() * psi_weight;
    dpsi = dpsi_total + term_from_u + term_from_mat;
}

} // namespace enn
