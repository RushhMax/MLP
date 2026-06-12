#include "pooling_layer.h"
#include <algorithm>
#include <stdexcept>

// ═══════════════════════════════════════════════════════
// FLUJO CNN → REDUCCIÓN DE DIMENSIONALIDAD
// PoolingLayer reduce el tamaño de los feature maps
// después de la capa convolucional.
//
// Propósito:
//   1. Reducir el costo computacional de las capas siguientes
//   2. Hacer la red más robusta a pequeñas traslaciones
//      (si el patrón se mueve 1px, el pooling igual lo detecta)
//   3. Extraer la característica más importante de cada región
//
// Ejemplo con pool_size=2, stride=2:
//   entrada: [32][28][28] → salida: [32][14][14]
//   cada ventana 2×2 se colapsa en 1 valor
// ═══════════════════════════════════════════════════════

// Crea tensor 3D lleno de ceros — usado para inicializar
// salidas y gradientes antes de acumular valores
Tensor3D PoolingLayer::zeros3D(int c, int h, int w) {
    return Tensor3D(c, std::vector<std::vector<double>>(
                        h, std::vector<double>(w, 0.0)));
}

// -------------------------------------------------------
// Constructor
// -------------------------------------------------------

// pool_size : tamaño de la ventana (2=2×2, 3=3×3)
// stride    : salto entre ventanas (-1 = igual a pool_size,
//             así las ventanas no se solapan)
// type      : MAX, MIN o AVERAGE — cómo colapsar la ventana
PoolingLayer::PoolingLayer(int pool_size, int stride, PoolType type)
    : pool_size_(pool_size),
      stride_(stride < 0 ? pool_size : stride), // -1 → stride = pool_size
      type_(type)
{
    if (pool_size_ <= 0 || stride_ <= 0)
        throw std::invalid_argument("PoolingLayer: pool_size y stride deben ser > 0");
}

// -------------------------------------------------------
// Dimensiones de salida
// -------------------------------------------------------

// Mismo cálculo que ConvLayer con VALID (sin padding)
// Cada dimensión se reduce según cuántas ventanas caben
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

    // Guarda el input para el backward
    // Average pooling no necesita saber qué pixel ganó,
    // pero MAX/MIN sí necesitan recordar de dónde viene cada valor
    input_cache_ = input;

    int C  = (int)input.size();        // canales (feature maps)
    int H  = (int)input[0].size();     // alto
    int W  = (int)input[0][0].size();  // ancho
    int OH = outHeight(H);             // alto de salida
    int OW = outWidth(W);              // ancho de salida

    Tensor3D out = zeros3D(C, OH, OW);

    // Máscara: guarda QUÉ pixel ganó en cada ventana
    // Solo se usa en MAX y MIN — necesaria en backward para
    // saber a quién devolver el gradiente
    // Se guarda como índice lineal: pr * pool_size + pc
    mask_.assign(C, std::vector<std::vector<int>>(
                     OH, std::vector<int>(OW, 0)));

    for (int c = 0; c < C; c++) {         // por cada feature map
        for (int i = 0; i < OH; i++) {    // por cada fila de salida
            for (int j = 0; j < OW; j++) { // por cada columna de salida

                // Coordenada inicial de la ventana en el input
                int r0 = i * stride_;
                int c0 = j * stride_;

                if (type_ == PoolType::AVERAGE) {
                    // ── AVERAGE POOLING ──────────────────────
                    // Suma todos los valores de la ventana
                    // y divide por el total de elementos
                    // Conserva información global de la región
                    double sum = 0.0;
                    for (int pr = 0; pr < pool_size_; pr++)
                        for (int pc = 0; pc < pool_size_; pc++)
                            sum += input[c][r0 + pr][c0 + pc];
                    out[c][i][j] = sum / (pool_size_ * pool_size_);

                } else {
                    // ── MAX o MIN POOLING ────────────────────
                    // Busca el valor más alto (MAX) o más bajo (MIN)
                    // de toda la ventana
                    // MAX: "¿se detectó este patrón en algún lugar?"
                    // MIN: útil para detectar regiones oscuras/ausencias

                    // Empieza con el primer elemento de la ventana
                    double best     = input[c][r0][c0];
                    int    best_idx = 0; // índice lineal del ganador

                    for (int pr = 0; pr < pool_size_; pr++) {
                        for (int pc = 0; pc < pool_size_; pc++) {
                            double v = input[c][r0 + pr][c0 + pc];

                            // Compara según el tipo de pooling
                            bool better = (type_ == PoolType::MAX)
                                          ? (v > best)   // MAX: busca el mayor
                                          : (v < best);  // MIN: busca el menor
                            if (better) {
                                best     = v;
                                // Guarda posición como índice lineal
                                // para reconstruirla en backward:
                                // pr * pool_size + pc → (pr, pc)
                                best_idx = pr * pool_size_ + pc;
                            }
                        }
                    }
                    // El valor ganador pasa a la salida
                    out[c][i][j]   = best;
                    // Guarda quién ganó — backward lo necesita
                    mask_[c][i][j] = best_idx;
                }
            }
        }
    }

    return out;
}

// -------------------------------------------------------
// Backward
// -------------------------------------------------------

// ═══════════════════════════════════════════════════════
// FLUJO CNN → BACKWARD DEL POOLING
// El pooling no tiene pesos que actualizar — su backward
// solo redistribuye el gradiente hacia el input
//
// La regla es simple:
//   MAX/MIN: el gradiente va SOLO al pixel que ganó
//            los demás reciben 0 (no contribuyeron)
//   AVERAGE: el gradiente se reparte IGUAL entre todos
//            los pixels de la ventana
// ═══════════════════════════════════════════════════════
Tensor3D PoolingLayer::backward(const Tensor3D& grad_out) {
    int C  = (int)input_cache_.size();
    int H  = (int)input_cache_[0].size();
    int W  = (int)input_cache_[0][0].size();
    int OH = (int)grad_out[0].size();
    int OW = (int)grad_out[0][0].size();

    // Gradiente del input inicializado en 0
    // Se irá acumulando pixel a pixel
    Tensor3D grad_in = zeros3D(C, H, W);

    for (int c = 0; c < C; c++) {
        for (int i = 0; i < OH; i++) {
            for (int j = 0; j < OW; j++) {

                // Coordenada inicial de la ventana
                int r0 = i * stride_;
                int c0 = j * stride_;

                // Gradiente que llegó desde la capa siguiente
                // para esta posición del feature map reducido
                double g = grad_out[c][i][j];

                if (type_ == PoolType::AVERAGE) {
                    // ── AVERAGE backward ─────────────────────
                    // En el forward todos contribuyeron por igual
                    // → en el backward todos reciben la misma parte
                    // share = g / (pool_size²)
                    double share = g / (pool_size_ * pool_size_);
                    for (int pr = 0; pr < pool_size_; pr++)
                        for (int pc = 0; pc < pool_size_; pc++)
                            grad_in[c][r0 + pr][c0 + pc] += share;

                } else {
                    // ── MAX/MIN backward ──────────────────────
                    // Solo el pixel ganador recibe el gradiente
                    // Los demás reciben 0 — no participaron en
                    // la salida, no les corresponde gradiente
                    //
                    // Reconstruye (pr, pc) desde el índice lineal
                    // guardado en la máscara durante el forward
                    int idx = mask_[c][i][j];
                    int pr  = idx / pool_size_; // fila dentro de la ventana
                    int pc  = idx % pool_size_; // columna dentro de la ventana
                    grad_in[c][r0 + pr][c0 + pc] += g;
                }
            }
        }
    }

    // Devuelve el gradiente en forma [C][H][W]
    // para que la capa convolucional anterior continue el backward
    return grad_in;
}