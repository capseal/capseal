#pragma once
#include "types.hpp"
#include <cmath>

namespace enn {

// Adam optimizer with bias correction
// IMPORTANT: Call tick() ONCE per batch, before any step() calls.
// step() applies the update using the current t; tick() advances t.
struct Adam {
    F lr;      // learning rate
    F beta1;   // exponential decay rate for first moment
    F beta2;   // exponential decay rate for second moment
    F eps;     // small constant for numerical stability
    int64_t t; // time step counter (counts batches, not parameter updates)

    explicit Adam(F lr_ = 1e-3, F beta1_ = 0.9, F beta2_ = 0.999, F eps_ = 1e-8)
        : lr(lr_), beta1(beta1_), beta2(beta2_), eps(eps_), t(0) {}

    // Advance time step - call ONCE per batch before applying gradients
    void tick() { ++t; }

    // Update parameter with gradient (matrix version)
    // NOTE: Does NOT increment t - call tick() once per batch instead
    void step(Mat& param, Mat& m, Mat& v, const Mat& grad) {
        // Adam moments
        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad.cwiseProduct(grad);

        // Bias correction using global t (should be >= 1 after tick())
        const F b1t = std::pow(beta1, static_cast<F>(t));
        const F b2t = std::pow(beta2, static_cast<F>(t));
        Mat m_hat = m / (1.0 - b1t);
        Mat v_hat = v / (1.0 - b2t);

        param -= lr * m_hat.cwiseQuotient((v_hat.array().sqrt() + eps).matrix());
    }

    // Update parameter with gradient (vector version)
    void step(Vec& param, Vec& m, Vec& v, const Vec& grad) {
        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad.cwiseProduct(grad);

        const F b1t = std::pow(beta1, static_cast<F>(t));
        const F b2t = std::pow(beta2, static_cast<F>(t));
        Vec m_hat = m / (1.0 - b1t);
        Vec v_hat = v / (1.0 - b2t);

        param -= lr * m_hat.cwiseQuotient((v_hat.array().sqrt() + eps).matrix());
    }

    // Update parameter with gradient (scalar version)
    void step(F& param, F& m, F& v, F grad) {
        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad * grad;

        const F b1t = std::pow(beta1, static_cast<F>(t));
        const F b2t = std::pow(beta2, static_cast<F>(t));
        F m_hat = m / (1.0 - b1t);
        F v_hat = v / (1.0 - b2t);

        param -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }

    void reset() { t = 0; }
};

// AdamW optimizer (Adam with decoupled weight decay)
struct AdamW : Adam {
    F weight_decay;
    
    explicit AdamW(F lr_ = 1e-3, F beta1_ = 0.9, F beta2_ = 0.999, 
                   F eps_ = 1e-8, F weight_decay_ = 1e-4)
        : Adam(lr_, beta1_, beta2_, eps_), weight_decay(weight_decay_) {}
    
    void step(Mat& param, Mat& m, Mat& v, const Mat& grad) {
        // Weight decay applied directly to parameters
        param *= (1.0 - lr * weight_decay);
        Adam::step(param, m, v, grad);
    }
    
    void step(Vec& param, Vec& m, Vec& v, const Vec& grad) {
        param *= (1.0 - lr * weight_decay);
        Adam::step(param, m, v, grad);
    }
    
    void step(F& param, F& m, F& v, F grad) {
        param *= (1.0 - lr * weight_decay);
        Adam::step(param, m, v, grad);
    }
};

// Learning rate schedulers
struct CosineScheduler {
    F base_lr;     // initial learning rate
    F min_lr;      // minimum learning rate
    int T_max;     // maximum number of iterations
    
    CosineScheduler(F base_lr_, F min_lr_, int T_max_) 
        : base_lr(base_lr_), min_lr(min_lr_), T_max(T_max_) {}
    
    F operator()(int t) const {
        if (t >= T_max) return min_lr;
        return min_lr + (base_lr - min_lr) * (1.0 + std::cos(M_PI * t / T_max)) / 2.0;
    }
};

struct LinearScheduler {
    F base_lr;
    F final_lr;
    int T_max;
    
    LinearScheduler(F base_lr_, F final_lr_, int T_max_)
        : base_lr(base_lr_), final_lr(final_lr_), T_max(T_max_) {}
        
    F operator()(int t) const {
        if (t >= T_max) return final_lr;
        F alpha = static_cast<F>(t) / T_max;
        return base_lr + alpha * (final_lr - base_lr);
    }
};

} // namespace enn