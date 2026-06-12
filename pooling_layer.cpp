#include "pooling_layer.h"
#include <algorithm>
#include <stdexcept>

// -------------------------------------------------------
// Helper
// -------------------------------------------------------

Tensor3D PoolingLayer::zeros3D(int c, int h, int w) {
    return Tensor3D(c, std::vector<std::vector<double>>(
                        h, std::vector<double>(w, 0.0)));
}

// -------------------------------------------------------
// Constructor
// -------------------------------------------------------

PoolingLayer::PoolingLayer(int pool_size, int stride, PoolType type)
    : pool_size_(pool_size),
      stride_(stride < 0 ? pool_size : stride),
      type_(type)
{
    if (pool_size_ <= 0 || stride_ <= 0)
        throw std::invalid_argument("PoolingLayer: pool_size y stride deben ser > 0");
}

// -------------------------------------------------------
// Dimensiones de salida
// -------------------------------------------------------

int PoolingLayer::outHeight(int in_h) const {
    return (in_h - pool_size_) / stride_ + 1;
}

int PoolingLayer::outWidth(int in_w) const {
    return (in_w - pool_size_) / stride_ + 1;
}

// -------------------------------------------------------
// Forward
// -------------------------------------------------------

Tensor3D PoolingLayer::forward(const Tensor3D& input) {
    input_cache_ = input;

    int C  = (int)input.size();
    int H  = (int)input[0].size();
    int W  = (int)input[0][0].size();
    int OH = outHeight(H);
    int OW = outWidth(W);

    Tensor3D out = zeros3D(C, OH, OW);

    // Inicializa máscara (usada en MAX y MIN)
    mask_.assign(C, std::vector<std::vector<int>>(
                     OH, std::vector<int>(OW, 0)));

    for (int c = 0; c < C; c++) {
        for (int i = 0; i < OH; i++) {
            for (int j = 0; j < OW; j++) {
                int r0 = i * stride_;
                int c0 = j * stride_;

                if (type_ == PoolType::AVERAGE) {
                    double sum = 0.0;
                    for (int pr = 0; pr < pool_size_; pr++)
                        for (int pc = 0; pc < pool_size_; pc++)
                            sum += input[c][r0 + pr][c0 + pc];
                    out[c][i][j] = sum / (pool_size_ * pool_size_);

                } else {
                    // MAX o MIN: busca el valor extremo y guarda su índice
                    double best = input[c][r0][c0];
                    int best_idx = 0;

                    for (int pr = 0; pr < pool_size_; pr++) {
                        for (int pc = 0; pc < pool_size_; pc++) {
                            double v = input[c][r0 + pr][c0 + pc];
                            bool better = (type_ == PoolType::MAX)
                                          ? (v > best)
                                          : (v < best);
                            if (better) {
                                best     = v;
                                best_idx = pr * pool_size_ + pc;
                            }
                        }
                    }
                    out[c][i][j]    = best;
                    mask_[c][i][j]  = best_idx;
                }
            }
        }
    }

    return out;
}

// -------------------------------------------------------
// Backward
// -------------------------------------------------------

Tensor3D PoolingLayer::backward(const Tensor3D& grad_out) {
    int C  = (int)input_cache_.size();
    int H  = (int)input_cache_[0].size();
    int W  = (int)input_cache_[0][0].size();
    int OH = (int)grad_out[0].size();
    int OW = (int)grad_out[0][0].size();

    Tensor3D grad_in = zeros3D(C, H, W);

    for (int c = 0; c < C; c++) {
        for (int i = 0; i < OH; i++) {
            for (int j = 0; j < OW; j++) {
                int r0 = i * stride_;
                int c0 = j * stride_;
                double g = grad_out[c][i][j];

                if (type_ == PoolType::AVERAGE) {
                    // Distribuye uniformemente entre los pool_size² elementos
                    double share = g / (pool_size_ * pool_size_);
                    for (int pr = 0; pr < pool_size_; pr++)
                        for (int pc = 0; pc < pool_size_; pc++)
                            grad_in[c][r0 + pr][c0 + pc] += share;

                } else {
                    // Solo el elemento ganador recibe el gradiente
                    int idx = mask_[c][i][j];
                    int pr  = idx / pool_size_;
                    int pc  = idx % pool_size_;
                    grad_in[c][r0 + pr][c0 + pc] += g;
                }
            }
        }
    }

    return grad_in;
}
