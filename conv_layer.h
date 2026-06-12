#pragma once
#include <vector>
#include <string>
#include <stdexcept>

// -------------------------------------------------------
// Tipos de tensores (solo STL)
// -------------------------------------------------------

// Tensor 3D: [canales][filas][columnas]
using Tensor3D = std::vector<std::vector<std::vector<double>>>;

// Tensor 4D: [filtros][canales][filas][columnas]
using Tensor4D = std::vector<std::vector<std::vector<std::vector<double>>>>;

// -------------------------------------------------------
// Enums compartidos
// -------------------------------------------------------

enum class Padding  { VALID, SAME };
enum class PoolType { MAX, MIN, AVERAGE };

// -------------------------------------------------------
// ConvLayer
//
//   Input:  Tensor3D [C_in][H][W]
//   Output: Tensor3D [F][H'][W']
//
//   VALID: H' = floor((H - K) / stride) + 1
//   SAME:  H' = ceil(H / stride)   (pad simétricamente)
//
//   Activación fija: ReLU
// -------------------------------------------------------

class ConvLayer {
public:
    // in_channels : profundidad del input (1 para escala de grises, 3 para RGB)
    // num_filters : número de filtros (profundidad del output)
    // kernel_size : K — el kernel es K×K
    // stride      : paso de la convolución
    // padding     : VALID (sin relleno) o SAME (output mismo H/W que input/stride)
    // seed        : semilla para inicialización aleatoria He
    ConvLayer(int in_channels,
              int num_filters,
              int kernel_size,
              int stride  = 1,
              Padding padding = Padding::VALID,
              unsigned seed   = 42);

    // ---- Dimensiones de salida ----
    int outHeight(int in_h) const;
    int outWidth (int in_w) const;

    // ---- Forward ----
    // Devuelve Tensor3D [num_filters][H'][W'] tras conv + bias + ReLU
    Tensor3D forward(const Tensor3D& input);

    // ---- Backward ----
    // grad_out : gradiente de la pérdida respecto a la salida [F][H'][W']
    // lr       : learning rate (actualiza kernels y biases in-place)
    // Devuelve : gradiente respecto al input original [C_in][H][W]
    Tensor3D backward(const Tensor3D& grad_out, double lr);

    // ---- Acceso a pesos (para inspección / serialización) ----
    const Tensor4D&           kernels() const { return kernels_; }
    const std::vector<double>& biases() const { return biases_; }

private:
    int     in_channels_;
    int     num_filters_;
    int     kernel_size_;
    int     stride_;
    Padding padding_;

    Tensor4D           kernels_;   // [F][C][K][K]
    std::vector<double> biases_;   // [F]

    // Caché para backprop
    Tensor3D input_cache_;       // input original (sin pad)
    Tensor3D padded_cache_;      // input con padding aplicado
    Tensor3D pre_relu_cache_;    // salida antes de ReLU (z = conv + bias)

    // ---- Helpers internos ----
    int padH(int in_h) const;    // padding total en eje H
    int padW(int in_w) const;    // padding total en eje W

    Tensor3D applyPadding (const Tensor3D& x) const;
    Tensor3D removePadding(const Tensor3D& x, int ph, int pw) const;

    static Tensor3D relu     (const Tensor3D& x);
    static Tensor3D reluDeriv(const Tensor3D& x);

    static Tensor3D zeros3D(int c, int h, int w);
};
