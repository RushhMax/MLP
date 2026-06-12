#include "conv_layer.h"
#include <cmath>
#include <random>
#include <algorithm>

// Helpers estáticos
// Crea un tensor vacío lleno de ceros
// Se usa para inicializar resultados antes de acumular sumas
// Equivale al "reservar memoria limpia" antes de operar
Tensor3D ConvLayer::zeros3D(int c, int h, int w) {
    return Tensor3D(c, std::vector<std::vector<double>>(
                        h, std::vector<double>(w, 0.0)));
}

// FLUJO CNN → paso 3 del forward: activación ReLU
// Aplica max(0,x) a cada elemento del feature map
// Los valores negativos (patrones que NO se detectaron) se hacen 0
// Los positivos (patrones detectados) se mantienen
Tensor3D ConvLayer::relu(const Tensor3D& x) {
    Tensor3D out = x;
    for (auto& mat : out)
        for (auto& row : mat)
            for (auto& v : row)
                v = v > 0.0 ? v : 0.0;
    return out;
}

// FLUJO CNN → paso 1 del backward: derivada de ReLU
// Devuelve 1 donde la neurona estaba activa (x>0)
// Devuelve 0 donde estaba bloqueada (x<=0)
// Es la "puerta" que decide qué gradientes pasan hacia atrás
Tensor3D ConvLayer::reluDeriv(const Tensor3D& x) {
    Tensor3D out = x;
    for (auto& mat : out)
        for (auto& row : mat)
            for (auto& v : row)
                v = v > 0.0 ? 1.0 : 0.0;
    return out;
}

// Constructor — inicialización He para ReLU
ConvLayer::ConvLayer(int in_channels, int num_filters, int kernel_size,
                     int stride, Padding padding, unsigned seed)
    : in_channels_(in_channels), num_filters_(num_filters),
      kernel_size_(kernel_size), stride_(stride), padding_(padding)
{
    // Semilla fija → mismos pesos iniciales en cada ejecución
    std::mt19937 rng(seed);

    // Escala He: compensa que ReLU mata ~50% de las neuronas
    // fan_in = C × K × K (cuántas entradas tiene cada neurona del filtro)
    // Sin esta escala los pesos serían muy grandes o muy pequeños
    // y la red no aprendería bien desde el inicio
    double std_dev = std::sqrt(2.0 / (in_channels * kernel_size * kernel_size));
    std::normal_distribution<double> dist(0.0, std_dev); // aleatorios desde 0 a std_dev

    // Reserva la estructura [F][C][K][K] en memoria
    // F filtros, cada uno opera sobre C canales con ventana K×K
    kernels_.resize(num_filters_,
        std::vector<std::vector<std::vector<double>>>(in_channels_,
            std::vector<std::vector<double>>(kernel_size_,
                std::vector<double>(kernel_size_, 0.0))));

    // Rellena con valores aleatorios He
    // Cada filtro arranca diferente → aprenderá patrones distintos
    // Si todos fueran iguales, todos los filtros aprenderían lo mismo
    for (int f = 0; f < num_filters_; f++)
        for (int c = 0; c < in_channels_; c++)
            for (int kr = 0; kr < kernel_size_; kr++)
                for (int kc = 0; kc < kernel_size_; kc++)
                    kernels_[f][c][kr][kc] = dist(rng);

    // Biases en 0 — convención estándar
    // Un bias por filtro, se suma al resultado de cada convolución
    biases_.assign(num_filters_, 0.0);
}

// ═══════════════════════════════════════════════════════
// FLUJO CNN → paso 1 del forward: preparar dimensiones
// Calculan cuánto padding agregar y qué tamaño tendrá la salida
// ═══════════════════════════════════════════════════════

// Cuántos ceros agregar verticalmente
// VALID → 0 (la imagen se encoge)
// SAME  → los suficientes para que la salida tenga ceil(H/stride) filas
int ConvLayer::padH(int in_h) const {
    if (padding_ == Padding::VALID) return 0;
    int out_h = (int)std::ceil((double)in_h / stride_);
    // despeja padding de la fórmula: out = (H + pad - K) / stride + 1
    int total  = std::max(0, (out_h - 1) * stride_ + kernel_size_ - in_h);
    return total;
}

// Ídem horizontal
int ConvLayer::padW(int in_w) const {
    if (padding_ == Padding::VALID) return 0;
    int out_w = (int)std::ceil((double)in_w / stride_);
    int total  = std::max(0, (out_w - 1) * stride_ + kernel_size_ - in_w);
    return total;
}

// Alto de la salida tras la convolución
// SAME:  ceil(H/stride)        → conserva resolución
// VALID: (H-K)/stride + 1      → se encoge por K-1 píxeles
int ConvLayer::outHeight(int in_h) const {
    if (padding_ == Padding::SAME)
        return (int)std::ceil((double)in_h / stride_);
    return (in_h - kernel_size_) / stride_ + 1;
}

// Ídem ancho
int ConvLayer::outWidth(int in_w) const {
    if (padding_ == Padding::SAME)
        return (int)std::ceil((double)in_w / stride_);
    return (in_w - kernel_size_) / stride_ + 1;
}

// ═══════════════════════════════════════════════════════
// FLUJO CNN → paso 1 del forward: agregar padding
// Rodea la imagen con ceros para que el kernel pueda
// centrarse en los píxeles del borde sin salirse
// ═══════════════════════════════════════════════════════
Tensor3D ConvLayer::applyPadding(const Tensor3D& x) const {
    int C  = (int)x.size();
    int H  = (int)x[0].size();
    int W  = (int)x[0][0].size();
    int ph = padH(H);
    int pw = padW(W);

    // El padding se reparte simétricamente: mitad arriba, mitad abajo
    int pad_top  = ph / 2;
    int pad_left = pw / 2;
    int pH = H + ph;
    int pW = W + pw;

    // Tensor de ceros del tamaño final con padding
    Tensor3D out = zeros3D(C, pH, pW);

    // Copia la imagen original en el centro — los bordes quedan en 0
    for (int c = 0; c < C; c++)
        for (int r = 0; r < H; r++)
            for (int col = 0; col < W; col++)
                out[c][r + pad_top][col + pad_left] = x[c][r][col];
    return out;
}

// ═══════════════════════════════════════════════════════
// FLUJO CNN → paso 5 del backward: limpiar padding
// La capa anterior no sabe que usamos padding
// Le devolvemos el gradiente del tamaño original sin los bordes de ceros
// ═══════════════════════════════════════════════════════
Tensor3D ConvLayer::removePadding(const Tensor3D& x, int ph, int pw) const {
    int C  = (int)x.size();
    int pH = (int)x[0].size();
    int pW = (int)x[0][0].size();
    int H  = pH - ph;
    int W  = pW - pw;

    int pad_top  = ph / 2;
    int pad_left = pw / 2;

    Tensor3D out = zeros3D(C, H, W);

    // Extrae solo la región central — descarta los bordes de padding
    for (int c = 0; c < C; c++)
        for (int r = 0; r < H; r++)
            for (int col = 0; col < W; col++)
                out[c][r][col] = x[c][r + pad_top][col + pad_left];
    return out;
}

// ═══════════════════════════════════════════════════════
// FLUJO CNN → FORWARD COMPLETO
// Recibe una imagen o feature map y produce nuevos feature maps
// que representan qué patrones detectó cada filtro y dónde
//
// Pasos:
//   1. Aplicar padding
//   2. Deslizar cada kernel sobre la imagen (convolución)
//   3. Sumar bias
//   4. Aplicar ReLU
// ═══════════════════════════════════════════════════════
Tensor3D ConvLayer::forward(const Tensor3D& input) {
    // Guarda el input original para backward
    // grad_input necesita saber el tamaño original sin padding
    input_cache_  = input;

    // Paso 1: agregar padding
    // Guarda la versión con padding porque grad_kernels
    // necesita los píxeles exactos que multiplicaron cada peso
    padded_cache_ = applyPadding(input);

    int H  = (int)padded_cache_[0].size();
    int W  = (int)padded_cache_[0][0].size();
    int OH = outHeight((int)input[0].size());
    int OW = outWidth ((int)input[0][0].size());

    // Inicializa la salida en 0 antes de acumular la convolución
    // Se guarda en caché porque backward necesita saber qué valores
    // tenía ANTES de ReLU para calcular la derivada
    pre_relu_cache_ = zeros3D(num_filters_, OH, OW);

    // Paso 2 y 3: convolución + bias
    // Para cada filtro f → produce un feature map [OH][OW]
    for (int f = 0; f < num_filters_; f++) {
        for (int i = 0; i < OH; i++) {       // fila de salida
            for (int j = 0; j < OW; j++) {   // columna de salida

                // Empieza acumulando el bias del filtro f
                double sum = biases_[f];

                // Producto punto entre el kernel f y la ventana del input
                // La ventana en el input empieza en (i*stride, j*stride)
                // y tiene tamaño K×K
                for (int c = 0; c < in_channels_; c++) {
                    for (int kr = 0; kr < kernel_size_; kr++) {
                        for (int kc = 0; kc < kernel_size_; kc++) {
                            // Coordenada en el input con padding
                            int r_in = i * stride_ + kr;
                            int c_in = j * stride_ + kc;
                            // Acumula: pixel × peso del kernel
                            sum += padded_cache_[c][r_in][c_in]
                                 * kernels_[f][c][kr][kc];
                        }
                    }
                }
                // Valor crudo antes de ReLU — guardado para backward
                pre_relu_cache_[f][i][j] = sum;
            }
        }
    }

    // Paso 4: ReLU — mata los negativos
    // Los feature maps resultantes indican con valores positivos
    // dónde y con qué intensidad se detectó cada patrón
    return relu(pre_relu_cache_);
}

// ═══════════════════════════════════════════════════════
// FLUJO CNN → BACKWARD COMPLETO
// Recibe el gradiente de la pérdida respecto a la salida
// y calcula:
//   1. Gradiente a través de ReLU
//   2. Gradiente de los kernels  → para actualizar pesos
//   3. Gradiente del input       → para propagar a capa anterior
//   4. Actualiza kernels y biases
//   5. Devuelve gradiente del input sin padding
// ═══════════════════════════════════════════════════════
Tensor3D ConvLayer::backward(const Tensor3D& grad_out, double lr) {
    int OH   = (int)grad_out[0].size();
    int OW   = (int)grad_out[0][0].size();
    int in_H = (int)input_cache_[0].size();
    int in_W = (int)input_cache_[0][0].size();
    int pH   = (int)padded_cache_[0].size();
    int pW   = (int)padded_cache_[0][0].size();

    // ── Paso 1: gradiente a través de ReLU ──────────────
    // Las neuronas bloqueadas por ReLU (valor ≤ 0 en forward)
    // no deben recibir gradiente — su derivada es 0
    // Las activas (valor > 0) dejan pasar el gradiente tal cual
    Tensor3D dReLU = reluDeriv(pre_relu_cache_);

    // grad_z = gradiente que llegó × puerta ReLU
    // Si estaba bloqueada → grad_z = 0 → ese peso no se toca
    Tensor3D grad_z = zeros3D(num_filters_, OH, OW);
    for (int f = 0; f < num_filters_; f++)
        for (int i = 0; i < OH; i++)
            for (int j = 0; j < OW; j++)
                grad_z[f][i][j] = grad_out[f][i][j] * dReLU[f][i][j];

    // ── Paso 2: gradiente de kernels y biases ───────────
    // grad_k[f][c][kr][kc] = cuánto contribuyó ese peso al error
    // = suma sobre todas las posiciones de salida de:
    //   error_en_esa_posición × pixel_que_multiplicó_ese_peso_en_forward
    Tensor4D grad_k(num_filters_,
        std::vector<std::vector<std::vector<double>>>(in_channels_,
            std::vector<std::vector<double>>(kernel_size_,
                std::vector<double>(kernel_size_, 0.0))));
    std::vector<double> grad_b(num_filters_, 0.0);

    for (int f = 0; f < num_filters_; f++) {
        for (int i = 0; i < OH; i++) {
            for (int j = 0; j < OW; j++) {
                double g = grad_z[f][i][j];

                // Bias recibe la suma de todos los errores del filtro f
                // porque el bias afecta TODAS las posiciones de salida
                grad_b[f] += g;

                // Cada peso del kernel recibe:
                // error × el pixel que multiplicó ese peso en el forward
                // (por eso necesitábamos guardar padded_cache_)
                for (int c = 0; c < in_channels_; c++)
                    for (int kr = 0; kr < kernel_size_; kr++)
                        for (int kc = 0; kc < kernel_size_; kc++)
                            grad_k[f][c][kr][kc] +=
                                g * padded_cache_[c][i*stride_+kr][j*stride_+kc];
            }
        }
    }

    // ── Paso 3: gradiente del input ─────────────────────
    // Necesario para que la capa anterior (si existe) pueda aprender
    // Es la convolución "al revés":
    //   forward:  input  × kernel → output
    //   backward: grad_z × kernel → grad_input
    // Cada píxel del input recibe la suma de todos los gradientes
    // de las posiciones de salida que lo usaron, ponderada por el kernel
    Tensor3D grad_pad = zeros3D(in_channels_, pH, pW);

    for (int f = 0; f < num_filters_; f++)
        for (int i = 0; i < OH; i++)
            for (int j = 0; j < OW; j++) {
                double g = grad_z[f][i][j];
                for (int c = 0; c < in_channels_; c++)
                    for (int kr = 0; kr < kernel_size_; kr++)
                        for (int kc = 0; kc < kernel_size_; kc++)
                            // Cada píxel acumula su parte del error
                            // de todas las ventanas que lo incluyeron
                            grad_pad[c][i*stride_+kr][j*stride_+kc] +=
                                g * kernels_[f][c][kr][kc];
            }

    // ── Paso 4: actualizar pesos ─────────────────────────
    // Descenso de gradiente: W = W - lr × gradiente
    // El filtro se mueve en la dirección que reduce el error
    for (int f = 0; f < num_filters_; f++) {
        biases_[f] -= lr * grad_b[f];
        for (int c = 0; c < in_channels_; c++)
            for (int kr = 0; kr < kernel_size_; kr++)
                for (int kc = 0; kc < kernel_size_; kc++)
                    kernels_[f][c][kr][kc] -= lr * grad_k[f][c][kr][kc];
    }

    // ── Paso 5: quitar padding del gradiente ─────────────
    // La capa anterior recibió input sin padding
    // Devolvemos su gradiente también sin padding [C_in][H][W]
    // para que pueda continuar el backward normalmente
    return removePadding(grad_pad, padH(in_H), padW(in_W));
}