#include "conv_layer.h"
#include <cmath>
#include <random>
#include <algorithm>

// -------------------------------------------------------
// Helpers estáticos
// -------------------------------------------------------

Tensor3D ConvLayer::zeros3D(int c, int h, int w) {
    return Tensor3D(c, std::vector<std::vector<double>>(
                        h, std::vector<double>(w, 0.0)));
}

Tensor3D ConvLayer::relu(const Tensor3D& x) {
    Tensor3D out = x;
    for (auto& mat : out)
        for (auto& row : mat)
            for (auto& v : row)
                v = v > 0.0 ? v : 0.0;
    return out;
}

Tensor3D ConvLayer::reluDeriv(const Tensor3D& x) {
    Tensor3D out = x;
    for (auto& mat : out)
        for (auto& row : mat)
            for (auto& v : row)
                v = v > 0.0 ? 1.0 : 0.0;
    return out;
}

// -------------------------------------------------------
// Constructor — inicialización He para ReLU
// -------------------------------------------------------

ConvLayer::ConvLayer(int in_channels, int num_filters, int kernel_size,
                     int stride, Padding padding, unsigned seed)
    : in_channels_(in_channels), num_filters_(num_filters),
      kernel_size_(kernel_size), stride_(stride), padding_(padding)
{
    std::mt19937 rng(seed);
    double std_dev = std::sqrt(2.0 / (in_channels * kernel_size * kernel_size));
    std::normal_distribution<double> dist(0.0, std_dev);

    kernels_.resize(num_filters_,
        std::vector<std::vector<std::vector<double>>>(in_channels_,
            std::vector<std::vector<double>>(kernel_size_,
                std::vector<double>(kernel_size_, 0.0))));

    for (int f = 0; f < num_filters_; f++)
        for (int c = 0; c < in_channels_; c++)
            for (int kr = 0; kr < kernel_size_; kr++)
                for (int kc = 0; kc < kernel_size_; kc++)
                    kernels_[f][c][kr][kc] = dist(rng);

    biases_.assign(num_filters_, 0.0);
}

// -------------------------------------------------------
// Dimensiones de salida
// -------------------------------------------------------

int ConvLayer::padH(int in_h) const {
    if (padding_ == Padding::VALID) return 0;
    int out_h = (int)std::ceil((double)in_h / stride_);
    int total  = std::max(0, (out_h - 1) * stride_ + kernel_size_ - in_h);
    return total;   // se reparte: top = total/2, bottom = total - total/2
}

int ConvLayer::padW(int in_w) const {
    if (padding_ == Padding::VALID) return 0;
    int out_w = (int)std::ceil((double)in_w / stride_);
    int total  = std::max(0, (out_w - 1) * stride_ + kernel_size_ - in_w);
    return total;
}

int ConvLayer::outHeight(int in_h) const {
    if (padding_ == Padding::SAME)
        return (int)std::ceil((double)in_h / stride_);
    return (in_h - kernel_size_) / stride_ + 1;
}

int ConvLayer::outWidth(int in_w) const {
    if (padding_ == Padding::SAME)
        return (int)std::ceil((double)in_w / stride_);
    return (in_w - kernel_size_) / stride_ + 1;
}

// -------------------------------------------------------
// Padding / unpadding
// -------------------------------------------------------

Tensor3D ConvLayer::applyPadding(const Tensor3D& x) const {
    int C  = (int)x.size();
    int H  = (int)x[0].size();
    int W  = (int)x[0][0].size();
    int ph = padH(H);
    int pw = padW(W);

    int pad_top  = ph / 2;
    int pad_left = pw / 2;
    int pH = H + ph;
    int pW = W + pw;

    Tensor3D out = zeros3D(C, pH, pW);
    for (int c = 0; c < C; c++)
        for (int r = 0; r < H; r++)
            for (int col = 0; col < W; col++)
                out[c][r + pad_top][col + pad_left] = x[c][r][col];
    return out;
}

Tensor3D ConvLayer::removePadding(const Tensor3D& x, int ph, int pw) const {
    int C  = (int)x.size();
    int pH = (int)x[0].size();
    int pW = (int)x[0][0].size();
    int H  = pH - ph;
    int W  = pW - pw;

    int pad_top  = ph / 2;
    int pad_left = pw / 2;

    Tensor3D out = zeros3D(C, H, W);
    for (int c = 0; c < C; c++)
        for (int r = 0; r < H; r++)
            for (int col = 0; col < W; col++)
                out[c][r][col] = x[c][r + pad_top][col + pad_left];
    return out;
}

// -------------------------------------------------------
// Forward pass
// -------------------------------------------------------

Tensor3D ConvLayer::forward(const Tensor3D& input) {
    input_cache_  = input;
    padded_cache_ = applyPadding(input);

    int H  = (int)padded_cache_[0].size();
    int W  = (int)padded_cache_[0][0].size();
    int OH = outHeight((int)input[0].size());
    int OW = outWidth ((int)input[0][0].size());

    pre_relu_cache_ = zeros3D(num_filters_, OH, OW);

    for (int f = 0; f < num_filters_; f++) {
        for (int i = 0; i < OH; i++) {
            for (int j = 0; j < OW; j++) {
                double sum = biases_[f];
                for (int c = 0; c < in_channels_; c++) {
                    for (int kr = 0; kr < kernel_size_; kr++) {
                        for (int kc = 0; kc < kernel_size_; kc++) {
                            int r_in = i * stride_ + kr;
                            int c_in = j * stride_ + kc;
                            sum += padded_cache_[c][r_in][c_in]
                                 * kernels_[f][c][kr][kc];
                        }
                    }
                }
                pre_relu_cache_[f][i][j] = sum;
            }
        }
    }

    return relu(pre_relu_cache_);
}

// -------------------------------------------------------
// Backward pass
// -------------------------------------------------------

Tensor3D ConvLayer::backward(const Tensor3D& grad_out, double lr) {
    int OH = (int)grad_out[0].size();
    int OW = (int)grad_out[0][0].size();
    int in_H = (int)input_cache_[0].size();
    int in_W = (int)input_cache_[0][0].size();
    int pH   = (int)padded_cache_[0].size();
    int pW   = (int)padded_cache_[0][0].size();

    // 1. Gradiente a través de ReLU
    Tensor3D dReLU = reluDeriv(pre_relu_cache_);
    Tensor3D grad_z = zeros3D(num_filters_, OH, OW);
    for (int f = 0; f < num_filters_; f++)
        for (int i = 0; i < OH; i++)
            for (int j = 0; j < OW; j++)
                grad_z[f][i][j] = grad_out[f][i][j] * dReLU[f][i][j];

    // 2. Gradiente respecto a kernels y biases
    Tensor4D grad_k(num_filters_,
        std::vector<std::vector<std::vector<double>>>(in_channels_,
            std::vector<std::vector<double>>(kernel_size_,
                std::vector<double>(kernel_size_, 0.0))));

    std::vector<double> grad_b(num_filters_, 0.0);

    for (int f = 0; f < num_filters_; f++) {
        for (int i = 0; i < OH; i++) {
            for (int j = 0; j < OW; j++) {
                double g = grad_z[f][i][j];
                grad_b[f] += g;
                for (int c = 0; c < in_channels_; c++)
                    for (int kr = 0; kr < kernel_size_; kr++)
                        for (int kc = 0; kc < kernel_size_; kc++)
                            grad_k[f][c][kr][kc] +=
                                g * padded_cache_[c][i*stride_+kr][j*stride_+kc];
            }
        }
    }

    // 3. Gradiente respecto al input (sobre el tensor con padding)
    Tensor3D grad_pad = zeros3D(in_channels_, pH, pW);

    for (int f = 0; f < num_filters_; f++)
        for (int i = 0; i < OH; i++)
            for (int j = 0; j < OW; j++) {
                double g = grad_z[f][i][j];
                for (int c = 0; c < in_channels_; c++)
                    for (int kr = 0; kr < kernel_size_; kr++)
                        for (int kc = 0; kc < kernel_size_; kc++)
                            grad_pad[c][i*stride_+kr][j*stride_+kc] +=
                                g * kernels_[f][c][kr][kc];
            }

    // 4. Actualizar pesos
    for (int f = 0; f < num_filters_; f++) {
        biases_[f] -= lr * grad_b[f];
        for (int c = 0; c < in_channels_; c++)
            for (int kr = 0; kr < kernel_size_; kr++)
                for (int kc = 0; kc < kernel_size_; kc++)
                    kernels_[f][c][kr][kc] -= lr * grad_k[f][c][kr][kc];
    }

    // 5. Quitar padding y devolver grad respecto al input original
    return removePadding(grad_pad, padH(in_H), padW(in_W));
}
