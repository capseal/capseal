#include "enn/frontend.hpp"
#include <random>
#include <cmath>

namespace enn {

namespace {
inline Vec elu(const Vec& x) {
    Vec out = x;
    for (int i = 0; i < out.size(); ++i) {
        if (out[i] < 0.0) {
            out[i] = std::exp(out[i]) - 1.0;
        }
    }
    return out;
}

inline Vec elu_grad(const Vec& x) {
    Vec grad(x.size());
    for (int i = 0; i < x.size(); ++i) {
        if (x[i] > 0.0) {
            grad[i] = 1.0;
        } else {
            grad[i] = std::exp(x[i]);
        }
    }
    return grad;
}
}

FrontendGrads::FrontendGrads(int filters, int input_dim, int temporal_kernel,
                             int depth_kernel, int embed_dim)
    : dW_temporal(Mat::Zero(filters, input_dim * temporal_kernel)),
      db_temporal(Vec::Zero(filters)),
      dW_spatial(Mat::Zero(filters, filters)),
      db_spatial(Vec::Zero(filters)),
      dW_depthwise(Mat::Zero(filters, depth_kernel)),
      db_depthwise(Vec::Zero(filters)),
      dW_proj(Mat::Zero(embed_dim, filters)),
      db_proj(Vec::Zero(embed_dim)) {}

void FrontendGrads::zero() {
    dW_temporal.setZero();
    db_temporal.setZero();
    dW_spatial.setZero();
    db_spatial.setZero();
    dW_depthwise.setZero();
    db_depthwise.setZero();
    dW_proj.setZero();
    db_proj.setZero();
}

void FrontendGrads::add_scaled(const FrontendGrads& other, F scale) {
    dW_temporal += scale * other.dW_temporal;
    db_temporal += scale * other.db_temporal;
    dW_spatial += scale * other.dW_spatial;
    db_spatial += scale * other.db_spatial;
    dW_depthwise += scale * other.dW_depthwise;
    db_depthwise += scale * other.db_depthwise;
    dW_proj += scale * other.dW_proj;
    db_proj += scale * other.db_proj;
}

void FrontendGrads::scale(F s) {
    dW_temporal *= s;
    db_temporal *= s;
    dW_spatial *= s;
    db_spatial *= s;
    dW_depthwise *= s;
    db_depthwise *= s;
    dW_proj *= s;
    db_proj *= s;
}

SpatialTemporalCNN::SpatialTemporalCNN(int input_dim, int embed_dim, int filters,
                                       int temporal_kernel, int depth_kernel,
                                       unsigned seed)
    : input_dim_(input_dim), embed_dim_(embed_dim), filters_(filters),
      temporal_kernel_(temporal_kernel), depth_kernel_(depth_kernel) {
    temporal_pad_ = temporal_kernel_ / 2;
    depth_pad_ = depth_kernel_ / 2;

    std::mt19937 gen(seed);
    std::normal_distribution<F> dist(0.0, 0.05);

    W_temporal_ = Mat::NullaryExpr(filters_, input_dim_ * temporal_kernel_, [&]() { return dist(gen); });
    b_temporal_ = Vec::Zero(filters_);

    W_spatial_ = Mat::NullaryExpr(filters_, filters_, [&]() { return dist(gen); });
    b_spatial_ = Vec::Zero(filters_);

    W_depthwise_ = Mat::NullaryExpr(filters_, depth_kernel_, [&]() { return dist(gen); });
    b_depthwise_ = Vec::Zero(filters_);

    W_proj_ = Mat::NullaryExpr(embed_dim_, filters_, [&]() { return dist(gen); });
    b_proj_ = Vec::Zero(embed_dim_);
}

Vec SpatialTemporalCNN::gather_temporal_window(const std::vector<Vec>& inputs, int t) const {
    Vec window(input_dim_ * temporal_kernel_);
    for (int k = 0; k < temporal_kernel_; ++k) {
        int idx = t + k - temporal_pad_;
        for (int c = 0; c < input_dim_; ++c) {
            int target_idx = k * input_dim_ + c;
            if (idx >= 0 && idx < static_cast<int>(inputs.size())) {
                window[target_idx] = inputs[idx][c];
            } else {
                window[target_idx] = 0.0;
            }
        }
    }
    return window;
}

void SpatialTemporalCNN::forward_sequence(const std::vector<Vec>& inputs,
                                          std::vector<Vec>& outputs,
                                          FrontendCache& cache) const {
    const int T = static_cast<int>(inputs.size());
    outputs.resize(T);

    cache.temporal_windows.resize(T);
    cache.temporal_linear.resize(T);
    cache.temporal_activated.resize(T);
    cache.spatial_linear.resize(T);
    cache.spatial_activated.resize(T);
    cache.depthwise_linear.resize(T);
    cache.depthwise_activated.resize(T);

    // Stage 1: temporal convolution + ELU
    for (int t = 0; t < T; ++t) {
        cache.temporal_windows[t] = gather_temporal_window(inputs, t);
        cache.temporal_linear[t] = W_temporal_ * cache.temporal_windows[t] + b_temporal_;
        cache.temporal_activated[t] = elu(cache.temporal_linear[t]);
    }

    // Stage 2: spatial mixing + ELU
    for (int t = 0; t < T; ++t) {
        cache.spatial_linear[t] = W_spatial_ * cache.temporal_activated[t] + b_spatial_;
        cache.spatial_activated[t] = elu(cache.spatial_linear[t]);
    }

    // Stage 3: depthwise temporal convolution + ELU
    for (int t = 0; t < T; ++t) {
        Vec depth = Vec::Zero(filters_);
        for (int f = 0; f < filters_; ++f) {
            F sum = 0.0;
            for (int k = 0; k < depth_kernel_; ++k) {
                int idx = t + k - depth_pad_;
                if (idx >= 0 && idx < T) {
                    sum += W_depthwise_(f, k) * cache.spatial_activated[idx][f];
                }
            }
            depth[f] = sum + b_depthwise_[f];
        }
        cache.depthwise_linear[t] = depth;
        cache.depthwise_activated[t] = elu(depth);
    }

    // Stage 4: projection (linear)
    for (int t = 0; t < T; ++t) {
        outputs[t] = W_proj_ * cache.depthwise_activated[t] + b_proj_;
    }
}

void SpatialTemporalCNN::backward_sequence(const std::vector<Vec>& inputs,
                                           const FrontendCache& cache,
                                           const std::vector<Vec>& d_outputs,
                                           FrontendGrads& grads) const {
    (void)inputs;
    const int T = static_cast<int>(d_outputs.size());

    std::vector<Vec> d_depthwise_act(T, Vec::Zero(filters_));
    std::vector<Vec> d_depthwise_lin(T, Vec::Zero(filters_));
    std::vector<Vec> d_spatial_act(T, Vec::Zero(filters_));
    std::vector<Vec> d_spatial_lin(T, Vec::Zero(filters_));
    std::vector<Vec> d_temporal_act(T, Vec::Zero(filters_));
    std::vector<Vec> d_temporal_lin(T, Vec::Zero(filters_));

    // Stage 4: projection gradients
    for (int t = 0; t < T; ++t) {
        const Vec& upstream = d_outputs[t];
        grads.dW_proj += upstream * cache.depthwise_activated[t].transpose();
        grads.db_proj += upstream;
        d_depthwise_act[t] = W_proj_.transpose() * upstream;
    }

    // Stage 3: depthwise temporal conv + ELU
    for (int t = 0; t < T; ++t) {
        Vec grad_act = d_depthwise_act[t];
        Vec act_grad = elu_grad(cache.depthwise_linear[t]);
        d_depthwise_lin[t] = grad_act.cwiseProduct(act_grad);
        grads.db_depthwise += d_depthwise_lin[t];
    }

    // Depthwise weight gradients + propagate to spatial activations
    for (int t = 0; t < T; ++t) {
        for (int f = 0; f < filters_; ++f) {
            F grad_val = d_depthwise_lin[t][f];
            if (grad_val == 0.0) continue;
            for (int k = 0; k < depth_kernel_; ++k) {
                int idx = t + k - depth_pad_;
                if (idx >= 0 && idx < T) {
                    grads.dW_depthwise(f, k) += grad_val * cache.spatial_activated[idx][f];
                    d_spatial_act[idx][f] += grad_val * W_depthwise_(f, k);
                }
            }
        }
    }

    // Stage 2: spatial mixing + ELU
    for (int t = 0; t < T; ++t) {
        Vec act_grad = elu_grad(cache.spatial_linear[t]);
        d_spatial_lin[t] = d_spatial_act[t].cwiseProduct(act_grad);
        grads.dW_spatial += d_spatial_lin[t] * cache.temporal_activated[t].transpose();
        grads.db_spatial += d_spatial_lin[t];
        d_temporal_act[t] = W_spatial_.transpose() * d_spatial_lin[t];
    }

    // Stage 1: temporal conv + ELU
    for (int t = 0; t < T; ++t) {
        Vec grad_act = d_temporal_act[t];
        Vec act_grad = elu_grad(cache.temporal_linear[t]);
        d_temporal_lin[t] = grad_act.cwiseProduct(act_grad);
        grads.db_temporal += d_temporal_lin[t];
        grads.dW_temporal += d_temporal_lin[t] * cache.temporal_windows[t].transpose();
    }
}

} // namespace enn
