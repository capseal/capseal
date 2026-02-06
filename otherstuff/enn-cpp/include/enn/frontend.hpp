#pragma once
#include "types.hpp"
#include <vector>

namespace enn {

struct FrontendCache {
    std::vector<Vec> temporal_windows;
    std::vector<Vec> temporal_linear;
    std::vector<Vec> temporal_activated;
    std::vector<Vec> spatial_linear;
    std::vector<Vec> spatial_activated;
    std::vector<Vec> depthwise_linear;
    std::vector<Vec> depthwise_activated;
};

struct FrontendGrads {
    Mat dW_temporal;
    Vec db_temporal;
    Mat dW_spatial;
    Vec db_spatial;
    Mat dW_depthwise;
    Vec db_depthwise;
    Mat dW_proj;
    Vec db_proj;

    FrontendGrads() = default;
    FrontendGrads(int filters, int input_dim, int temporal_kernel,
                  int depth_kernel, int embed_dim);

    void zero();
    void add_scaled(const FrontendGrads& other, F scale);
    void scale(F s);
};

class SpatialTemporalCNN {
public:
    SpatialTemporalCNN(int input_dim, int embed_dim, int filters,
                       int temporal_kernel, int depth_kernel,
                       unsigned seed = 1337);

    int input_dim() const { return input_dim_; }
    int embed_dim() const { return embed_dim_; }
    int filters() const { return filters_; }

    // Forward entire sequence producing embedded features per timestep
    void forward_sequence(const std::vector<Vec>& inputs,
                          std::vector<Vec>& outputs,
                          FrontendCache& cache) const;

    // Backward through the sequence given upstream gradients for each timestep
    void backward_sequence(const std::vector<Vec>& inputs,
                           const FrontendCache& cache,
                           const std::vector<Vec>& d_outputs,
                           FrontendGrads& grads) const;

    // Parameter accessors (used by optimizer)
    Mat& W_temporal() { return W_temporal_; }
    Vec& b_temporal() { return b_temporal_; }
    Mat& W_spatial() { return W_spatial_; }
    Vec& b_spatial() { return b_spatial_; }
    Mat& W_depthwise() { return W_depthwise_; }
    Vec& b_depthwise() { return b_depthwise_; }
    Mat& W_proj() { return W_proj_; }
    Vec& b_proj() { return b_proj_; }

    const Mat& W_temporal() const { return W_temporal_; }
    const Vec& b_temporal() const { return b_temporal_; }
    const Mat& W_spatial() const { return W_spatial_; }
    const Vec& b_spatial() const { return b_spatial_; }
    const Mat& W_depthwise() const { return W_depthwise_; }
    const Vec& b_depthwise() const { return b_depthwise_; }
    const Mat& W_proj() const { return W_proj_; }
    const Vec& b_proj() const { return b_proj_; }

private:
    int input_dim_;
    int embed_dim_;
    int filters_;
    int temporal_kernel_;
    int depth_kernel_;
    int temporal_pad_;
    int depth_pad_;

    Mat W_temporal_;
    Vec b_temporal_;
    Mat W_spatial_;
    Vec b_spatial_;
    Mat W_depthwise_;
    Vec b_depthwise_;
    Mat W_proj_;
    Vec b_proj_;

    Vec gather_temporal_window(const std::vector<Vec>& inputs, int t) const;
};

} // namespace enn
