#include "flatten_layer.h"
#include <stdexcept>

// ═══════════════════════════════════════════════════════
// FLUJO CNN → CONEXIÓN CONV → MLP
// FlattenLayer es el puente entre las capas convolucionales
// y el MLP. No tiene pesos que aprender — solo reorganiza
// los datos de 3D a 1D para que el MLP pueda procesarlos.
//
// Ejemplo:
//   entrada:  [32][7][7]  → 32 feature maps de 7×7
//   salida:   [1568]      → vector plano de 1568 valores
// ═══════════════════════════════════════════════════════

// -------------------------------------------------------
// Forward — aplanar Tensor3D → vector 1D
// -------------------------------------------------------

std::vector<double> FlattenLayer::forward(const Tensor3D& input) {

    // Guarda las dimensiones originales del tensor
    // El backward las necesita para reconstruir la forma 3D
    shape_c_   = (int)input.size();         // canales (feature maps)
    shape_h_   = (int)input[0].size();      // alto de cada feature map
    shape_w_   = (int)input[0][0].size();   // ancho de cada feature map
    flat_size_ = shape_c_ * shape_h_ * shape_w_; // total de elementos

    std::vector<double> out;
    out.reserve(flat_size_); // reserva memoria para evitar realocaciones

    // Recorre en orden C-contiguous: canal por canal, fila por fila
    // Esto garantiza que el índice en el vector plano sea:
    //   idx = c * H * W + h * W + w
    // El MLP recibe este vector como si fuera la "imagen de entrada"
    for (int c = 0; c < shape_c_; c++)
        for (int h = 0; h < shape_h_; h++)
            for (int w = 0; w < shape_w_; w++)
                out.push_back(input[c][h][w]);

    return out; // vector 1D listo para entrar al MLP
}

// -------------------------------------------------------
// Backward — reconstruir Tensor3D desde gradiente 1D
// -------------------------------------------------------

Tensor3D FlattenLayer::backward(const std::vector<double>& grad_out) const {

    // El MLP devuelve un gradiente como vector 1D
    // Verifica que tenga exactamente el mismo tamaño que el forward aplanó
    // Si no coincide, algo está mal en la arquitectura
    if ((int)grad_out.size() != flat_size_)
        throw std::invalid_argument(
            "FlattenLayer::backward: tamaño de grad_out no coincide con flat_size_");

    // Reconstruye el tensor 3D con las dimensiones guardadas en forward
    // Se inicializa en 0 antes de rellenar
    Tensor3D grad_in(shape_c_,
        std::vector<std::vector<double>>(shape_h_,
            std::vector<double>(shape_w_, 0.0)));

    // Recorre en el mismo orden que el forward
    // Cada valor del vector plano vuelve a su posición [c][h][w] original
    // idx avanza secuencialmente igual que lo hizo en el forward
    int idx = 0;
    for (int c = 0; c < shape_c_; c++)
        for (int h = 0; h < shape_h_; h++)
            for (int w = 0; w < shape_w_; w++)
                grad_in[c][h][w] = grad_out[idx++];

    // Devuelve el gradiente en forma 3D para que la última
    // capa de pooling o conv pueda continuar el backward
    return grad_in;
}