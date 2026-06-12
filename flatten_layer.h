#pragma once
#include "conv_layer.h"  // Tensor3D
#include <vector>

// -------------------------------------------------------
// FlattenLayer
//   Convierte Tensor3D [C][H][W]  →  vector<double> [C*H*W]
//   Orden: canal externo, fila media, columna interna
//   (C-contiguous / row-major)
//
//   No tiene parámetros entrenables.
//   Backward simplemente remodela el vector plano de vuelta
//   a [C][H][W].
// -------------------------------------------------------

class FlattenLayer {
public:
    FlattenLayer() = default;

    // Forward: aplana el tensor y guarda la forma para backward
    // Devuelve vector<double> de tamaño C*H*W
    std::vector<double> forward(const Tensor3D& input);

    // Backward: remodela grad (vector<double>) → Tensor3D [C][H][W]
    Tensor3D backward(const std::vector<double>& grad_out) const;

    // Tamaño del vector aplanado (válido tras el primer forward)
    int flatSize() const { return flat_size_; }

private:
    int shape_c_ = 0;
    int shape_h_ = 0;
    int shape_w_ = 0;
    int flat_size_ = 0;
};
