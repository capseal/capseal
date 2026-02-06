#include "enn/cell.hpp"
#include <random>
#include <algorithm>
#include <cmath>
#include <Eigen/Eigenvalues>

namespace enn {

namespace {
constexpr F kLayerNormEps = 1e-5;
}

EntangledCell::EntangledCell(int k_, int input_dim_, int hidden_dim_, F lambda_,
                             bool use_layer_norm_, unsigned seed)
    : k(k_), input_dim(input_dim_), hidden_dim(hidden_dim_),
      ln_gamma(Vec::Ones(k_)), ln_beta(Vec::Zero(k_)),
      use_layer_norm(use_layer_norm_) {
    log_lambda = std::log(std::max(lambda_, static_cast<F>(1e-6)));

    std::mt19937 gen(seed);
    std::normal_distribution<F> dist(0.0, 0.05);

    Wx = Mat::NullaryExpr(k, input_dim, [&]() { return dist(gen); });
    Wh = Mat::NullaryExpr(k, hidden_dim, [&]() { return dist(gen); });
    L = Mat::NullaryExpr(k, k, [&]() { return dist(gen); });
    b = Vec::Zero(k);
}

Vec EntangledCell::forward(const Vec& x, const Vec& h, const Vec& psi_in, CellCache& cache) const {
    cache.x = x;
    cache.h = h;
    cache.psi_in = psi_in;

    cache.E = L * L.transpose();
    const F lambda_val = lambda();

    cache.pre_act = Wx * x + Wh * h + (cache.E * psi_in) - lambda_val * psi_in + b;

    Vec activ = cache.pre_act;
    if (use_layer_norm) {
        cache.ln_mean = activ.mean();
        Vec centered = activ.array() - cache.ln_mean;
        F var = centered.array().square().mean();
        cache.ln_inv_std = 1.0 / std::sqrt(var + kLayerNormEps);
        cache.normed = centered * cache.ln_inv_std;
        activ = cache.normed.cwiseProduct(ln_gamma) + ln_beta;
    } else {
        cache.normed = activ;
        cache.ln_mean = 0.0;
        cache.ln_inv_std = 1.0;
    }

    cache.psi = activ.array().tanh();
    return cache.psi;
}

void EntangledCell::backward(const Vec& dpsi_out, const CellCache& cache,
                             Grads& grads, Vec& dpsi_in, Vec& dh, Vec& dx) const {
    Vec ds = dpsi_out.cwiseProduct((1.0 - cache.psi.array().square()).matrix());
    Vec dpre = ds;

    if (use_layer_norm) {
        grads.dgamma += ds.cwiseProduct(cache.normed);
        grads.dbeta += ds;
        Vec dnorm = ds.cwiseProduct(ln_gamma);
        const F n = static_cast<F>(k);
        const F sum1 = dnorm.sum();
        const F sum2 = dnorm.cwiseProduct(cache.normed).sum();
        Vec term = dnorm.array() * n - sum1 - cache.normed.array() * sum2;
        dpre = (cache.ln_inv_std / n) * term;
    }

    grads.dWx += dpre * cache.x.transpose();
    grads.dWh += dpre * cache.h.transpose();
    grads.db += dpre;

    Mat dE = dpre * cache.psi_in.transpose();
    const F lambda_val = lambda();
    grads.dlog_lambda += -(dpre.dot(cache.psi_in)) * lambda_val;

    grads.dL += (dE + dE.transpose()) * L;

    dpsi_in = cache.E.transpose() * dpre - lambda_val * dpre;
    dh = Wh.transpose() * dpre;
    dx = Wx.transpose() * dpre;
}

Mat EntangledCell::get_entanglement_matrix() const {
    return L * L.transpose();
}

bool EntangledCell::is_entanglement_psd(F tolerance) const {
    Mat E = get_entanglement_matrix();
    Eigen::SelfAdjointEigenSolver<Mat> solver(E);
    if (solver.info() != Eigen::Success) {
        return false;
    }
    Vec eigenvals = solver.eigenvalues();
    return eigenvals.minCoeff() >= -tolerance;
}

F EntangledCell::compute_psd_regularizer_loss() const {
    return 0.5 * 1e-6 * L.squaredNorm();
}

F EntangledCell::compute_param_l2_loss() const {
    return 0.5 * (Wx.squaredNorm() + Wh.squaredNorm() + L.squaredNorm() +
                  b.squaredNorm() + ln_gamma.squaredNorm() + ln_beta.squaredNorm());
}

} // namespace enn
