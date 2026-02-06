#pragma once
#include "types.hpp"

namespace enn {

struct CollapseCache {
    Vec scores;
    Vec alpha;
    Vec collapsed;
    Vec gated;
    F temperature = 1.0;

    CollapseCache() = default;
    CollapseCache(int k) : scores(Vec::Zero(k)), alpha(Vec::Zero(k)),
                           collapsed(Vec::Zero(k)), gated(Vec::Zero(k)) {}
};

struct Collapse {
    Mat Wq;         // [k x k] attention query weights
    Mat Wout;       // [output_dim x k] projection weights (was Vec for scalar output)
    Vec bout;       // [output_dim] biases (was scalar for output_dim=1)
    F log_temp;     // learned log-temperature
    int k;          // entanglement dimension
    int output_dim; // number of output bits (1 for scalar, >1 for multi-bit)

    explicit Collapse(int k_, int output_dim_ = 1, unsigned seed = 123);

    // Numerically stable softmax helper
    Vec softmax(const Vec& z) const;
    Vec softmax_jacobian_matvec(const Vec& alpha, const Vec& vec) const;

    // Forward pass returning scalar prediction (for output_dim=1, backwards compatible)
    F forward(const Vec& psi, CollapseCache& cache) const;

    // Forward pass returning vector prediction (for multi-bit output)
    Vec forward_multi(const Vec& psi, CollapseCache& cache) const;

    struct Grads {
        Mat dWq;
        Mat dWout;      // [output_dim x k]
        Vec dbias;      // [output_dim]
        F dlog_temp = 0.0;
        int output_dim;

        explicit Grads(int k, int output_dim_ = 1)
            : dWq(Mat::Zero(k, k)),
              dWout(Mat::Zero(output_dim_, k)),
              dbias(Vec::Zero(output_dim_)),
              output_dim(output_dim_) {}

        void zero() {
            dWq.setZero();
            dWout.setZero();
            dbias.setZero();
            dlog_temp = 0.0;
        }

        void add_scaled(const Grads& other, F scale) {
            dWq += scale * other.dWq;
            dWout += scale * other.dWout;
            dbias += scale * other.dbias;
            dlog_temp += scale * other.dlog_temp;
        }

        void scale(F s) {
            dWq *= s;
            dWout *= s;
            dbias *= s;
            dlog_temp *= s;
        }
    };

    // Backward for scalar output (output_dim=1)
    void backward(F dL_dpred, const Vec& psi, const CollapseCache& cache,
                  Vec& dpsi, Grads& grads) const;

    // Backward for multi-bit output
    void backward_multi(const Vec& dL_dpred, const Vec& psi, const CollapseCache& cache,
                        Vec& dpsi, Grads& grads) const;
};

} // namespace enn
