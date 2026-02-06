#pragma once
#include "types.hpp"
#include "regularizers.hpp"

namespace enn {

struct CellCache { 
    Vec x;        // input at timestep
    Vec h;        // hidden state
    Vec psi_in;   // incoming entangled state
    Vec pre_act;  // raw pre-activation before norm
    Vec normed;   // normalized pre-activation (if LN enabled)
    Vec psi;      // output psi after tanh
    Mat E;        // entanglement matrix (cached for backprop)
    F ln_inv_std = 1.0;
    F ln_mean = 0.0;
    
    CellCache() = default;
    CellCache(int input_dim, int hidden_dim, int k) 
        : x(Vec::Zero(input_dim)), h(Vec::Zero(hidden_dim)), 
          psi_in(Vec::Zero(k)), pre_act(Vec::Zero(k)), normed(Vec::Zero(k)),
          psi(Vec::Zero(k)), E(Mat::Zero(k, k)) {}
};

struct EntangledCell {
    // Dimensions
    int k;           // entanglement dimension
    int input_dim;   // input dimension
    int hidden_dim;  // hidden state dimension
    
    // Parameters
    Mat Wx;          // [k x input_dim] input weights
    Mat Wh;          // [k x hidden_dim] hidden weights  
    Mat L;           // [k x k] Cholesky factor: E = L * L^T
    Vec b;           // [k] bias
    Vec ln_gamma;    // layer norm gain
    Vec ln_beta;     // layer norm bias
    F log_lambda;    // log shrinkage parameter (ensures positivity)
    bool use_layer_norm = true;
    
    explicit EntangledCell(int k_, int input_dim_, int hidden_dim_, 
                           F lambda_, bool use_layer_norm_ = true,
                           unsigned seed = 42);
    
    // Forward pass: psi_out = tanh(Wx*x + Wh*h + (E - lambda*I)*psi_in + b)
    Vec forward(const Vec& x, const Vec& h, const Vec& psi_in, CellCache& cache) const;
    
    // Gradient accumulation structure
    struct Grads { 
        Mat dWx;      // [k x input_dim]
        Mat dWh;      // [k x hidden_dim]
        Mat dL;       // [k x k] gradient w.r.t. L (not E directly)
        Vec db;       // [k]
        Vec dgamma;   // [k]
        Vec dbeta;    // [k]
        F dlog_lambda;    // scalar
        
        Grads() : dlog_lambda(0.0) {}
        Grads(int k, int input_dim, int hidden_dim) 
            : dWx(Mat::Zero(k, input_dim)), dWh(Mat::Zero(k, hidden_dim)),
              dL(Mat::Zero(k, k)), db(Vec::Zero(k)), 
              dgamma(Vec::Zero(k)), dbeta(Vec::Zero(k)), dlog_lambda(0.0) {}
        
        void zero() {
            dWx.setZero();
            dWh.setZero();
            dL.setZero();
            db.setZero();
            dgamma.setZero();
            dbeta.setZero();
            dlog_lambda = 0.0;
        }

        void add_scaled(const Grads& other, F scale) {
            dWx += scale * other.dWx;
            dWh += scale * other.dWh;
            dL += scale * other.dL;
            db += scale * other.db;
            dgamma += scale * other.dgamma;
            dbeta += scale * other.dbeta;
            dlog_lambda += scale * other.dlog_lambda;
        }

        void scale(F s) {
            dWx *= s;
            dWh *= s;
            dL *= s;
            db *= s;
            dgamma *= s;
            dbeta *= s;
            dlog_lambda *= s;
        }
    };
    
    // Backward pass: accumulate gradients and return upstream derivatives
    void backward(const Vec& dpsi_out, const CellCache& cache, 
                  Grads& grads, Vec& dpsi_in, Vec& dh, Vec& dx) const;
    
    // Get current entanglement matrix E = L * L^T
    Mat get_entanglement_matrix() const;
    
    // Check if entanglement matrix is positive semi-definite (for debugging)
    bool is_entanglement_psd(F tolerance = 1e-8) const;
    
    // Compute regularization losses
    F compute_psd_regularizer_loss() const;
    F compute_param_l2_loss() const;
    F lambda() const { return std::exp(log_lambda); }
    bool layer_norm_enabled() const { return use_layer_norm; }
};

} // namespace enn
