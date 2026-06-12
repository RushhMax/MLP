#include "flatten_layer.h"
#include <stdexcept>

// -------------------------------------------------------
// Forward
// -------------------------------------------------------

std::vector<double> FlattenLayer::forward(const Tensor3D& input) {
    shape_c_   = (int)input.size();
    shape_h_   = (int)input[0].size();
    shape_w_   = (int)input[0][0].size();
    flat_size_ = shape_c_ * shape_h_ * shape_w_;

    std::vector<double> out;
    out.reserve(flat_size_);

    // Orden C-contiguous: [c][h][w] → índice c*H*W + h*W + w
    for (int c = 0; c < shape_c_; c++)
        for (int h = 0; h < shape_h_; h++)
            for (int w = 0; w < shape_w_; w++)
                out.push_back(input[c][h][w]);

    return out;
}

// -------------------------------------------------------
// Backward
// -------------------------------------------------------

Tensor3D FlattenLayer::backward(const std::vector<double>& grad_out) const {
    if ((int)grad_out.size() != flat_size_)
        throw std::invalid_argument(
            "FlattenLayer::backward: tamaño de grad_out no coincide con flat_size_");

    Tensor3D grad_in(shape_c_,
        std::vector<std::vector<double>>(shape_h_,
            std::vector<double>(shape_w_, 0.0)));

    int idx = 0;
    for (int c = 0; c < shape_c_; c++)
        for (int h = 0; h < shape_h_; h++)
            for (int w = 0; w < shape_w_; w++)
                grad_in[c][h][w] = grad_out[idx++];

    return grad_in;
}
