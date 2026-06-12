#pragma once
#include "conv_layer.h"  // Tensor3D, PoolType

// -------------------------------------------------------
// PoolingLayer
//
//   Input:  Tensor3D [C][H][W]
//   Output: Tensor3D [C][H'][W']
//
//   H' = floor((H - pool_size) / stride) + 1
//   W' = floor((W - pool_size) / stride) + 1
//
//   Sin parámetros entrenables — backward propaga
//   el gradiente solo a las posiciones que "ganaron"
//   (MAX/MIN) o lo distribuye uniformemente (AVERAGE).
// -------------------------------------------------------

class PoolingLayer {
public:
    // pool_size : ventana cuadrada pool_size × pool_size
    // stride    : paso de la ventana (por defecto = pool_size → sin solapamiento)
    // type      : MAX, MIN o AVERAGE
    PoolingLayer(int pool_size, int stride = -1,
                 PoolType type = PoolType::MAX);

    // Dimensiones de salida dado el tamaño del input
    int outHeight(int in_h) const;
    int outWidth (int in_w) const;

    // Forward: devuelve [C][H'][W']
    Tensor3D forward(const Tensor3D& input);

    // Backward: devuelve grad respecto al input [C][H][W]
    Tensor3D backward(const Tensor3D& grad_out);

private:
    int      pool_size_;
    int      stride_;
    PoolType type_;

    // Caché para backward
    Tensor3D input_cache_;

    // Máscara de posiciones ganadoras para MAX/MIN:
    // mask_[c][h'][w'] = índice lineal (r*W + c) dentro de la ventana
    std::vector<std::vector<std::vector<int>>> mask_;

    static Tensor3D zeros3D(int c, int h, int w);
};
