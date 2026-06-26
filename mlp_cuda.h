#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <iomanip>
#include <memory>
#include <cassert>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "linalg.h"

using namespace std;

// ═══════════════════════════════════════════════════════
// MACROS DE SEGURIDAD — wrappean las llamadas CUDA/cuBLAS
// Si algo falla (memoria insuficiente, GPU no disponible,
// etc.) imprimen el error exacto y matan el programa
// En vez de crashear silenciosamente, dan el archivo y línea
// ═══════════════════════════════════════════════════════
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error en %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t st = (call);                                         \
        if (st != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error en %s:%d — código %d\n",         \
                    __FILE__, __LINE__, (int)st);                           \
            exit(1);                                                        \
        }                                                                   \
    } while (0)


// ═══════════════════════════════════════════════════════
// UTILIDADES CPU — operaciones simples que no valen la
// pena mandar a GPU por el overhead de transferencia
// ═══════════════════════════════════════════════════════

// Crea vector one-hot de tamaño nClases con un 1 en la posición label
// Ej: oneHot(3, 5) → [0, 0, 0, 1, 0]
// Se usa para comparar la salida de la red con la clase correcta
inline Vector oneHot(int label, int nClases) {
    Vector v = Vector::Zero(nClases);
    v(label) = 1.0;
    return v;
}

// Devuelve el índice del valor más alto en un Vector
// Se usa para convertir probabilidades → clase predicha
// Ej: [0.1, 0.8, 0.1] → 1 (clase 1 ganó)
inline int argmax(const Vector& v) {
    int idx = 0;
    for (int i=1; i<v.size(); i++)
        if (v(i) > v(idx)) idx = i;
    return idx;
}

// Versión de argmax para vector<float> en vez de Vector (Eigen)
// Se usa cuando la salida viene de GPU (siempre en float)
inline int argmaxf(const vector<float>& v) {
    return (int)(max_element(v.begin(), v.end()) - v.begin());
}

// ═══════════════════════════════════════════════════════
// GpuBuf 
struct GpuBuf {
    float* ptr = nullptr; // puntero a memoria en GPU
    int    n   = 0;       // cantidad de floats reservados

    GpuBuf() = default;

    // Reserva n floats en GPU e inicializa todo a 0
    // cudaMalloc → reservar, cudaMemset → limpiar
    explicit GpuBuf(int size) : n(size) {
        CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(float)));
        CUDA_CHECK(cudaMemset(ptr, 0, size * sizeof(float)));
    }

    // Libera la memoria GPU automáticamente al destruirse
    ~GpuBuf() { if (ptr) cudaFree(ptr); }

    // Copias deshabilitadas — dos GpuBuf no pueden apuntar
    // al mismo bloque de memoria (doble free al destruirse)
    GpuBuf(const GpuBuf&)            = delete;
    GpuBuf& operator=(const GpuBuf&) = delete;

    // Movimiento permitido — transfiere la propiedad del puntero
    // El original queda con ptr=null para evitar doble free
    GpuBuf(GpuBuf&& o) noexcept : ptr(o.ptr), n(o.n) { o.ptr = nullptr; o.n = 0; }
    GpuBuf& operator=(GpuBuf&& o) noexcept {
        if (ptr) cudaFree(ptr);
        ptr = o.ptr; n = o.n; o.ptr = nullptr; o.n = 0;
        return *this;
    }

    // Copia datos de CPU → GPU (Host to Device)
    // assert verifica que los tamaños coincidan antes de copiar
    void fromCPU(const vector<float>& h) {
        assert((int)h.size() == n);
        CUDA_CHECK(cudaMemcpy(ptr, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Copia datos de GPU → CPU (Device to Host)
    // resize asegura que el vector CPU tenga el tamaño correcto
    void toCPU(vector<float>& h) const {
        h.resize(n);
        CUDA_CHECK(cudaMemcpy(h.data(), ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Reinicia todos los valores a 0 sin liberar memoria
    // Más rápido que fromCPU con un vector de ceros
    void zero() { CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(float))); }
};

// ═══════════════════════════════════════════════════════
// KERNELS CUDA — funciones que corren en GPU en paralelo
// Cada thread procesa un elemento distinto del array
// Patrón estándar: int i = blockIdx.x * blockDim.x + threadIdx.x
// ═══════════════════════════════════════════════════════

// Prepara la entrada agregando el nodo bias al inicio
// Resultado: y = [1.0, x[0], x[1], ..., x[n-1]]
// El bias siempre vale 1 — su peso aprendible está en W[0][:]
__global__ void kernel_bias(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) y[0] = 1.0f;   // bias fijo en posición 0
    if (i < n)  y[i + 1] = x[i]; // datos reales desplazados 1
}

// ── Activaciones forward ────────────────────────────────

// ReLU: max(0, x) — mata negativos, deja positivos intactos
__global__ void kernel_relu(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

// Derivada de ReLU: 1 si x>0, 0 si x≤0
// Se usa en backward como "puerta" que decide qué gradientes pasan
__global__ void kernel_relu_d(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? 1.0f : 0.0f;
}

// LeakyReLU: x si x>0, alpha*x si x≤0
// Evita el problema de "neuronas muertas" de ReLU — siempre
// deja pasar algo de gradiente aunque sea pequeño (alpha=0.01)
__global__ void kernel_leaky_relu(const float* x, float* y, int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? x[i] : alpha * x[i];
}

// Derivada de LeakyReLU: 1 si x>0, alpha si x≤0
__global__ void kernel_leaky_relu_d(const float* x, float* y, int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? 1.0f : alpha;
}

// Sigmoid: 1 / (1 + e^-x) — aplasta todo entre 0 y 1
// Menos usada en capas ocultas modernas por el problema
// del gradiente que desaparece en redes profundas
__global__ void kernel_sigmoid(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.0f / (1.0f + expf(-x[i]));
}

// Derivada de sigmoid: s(x) * (1 - s(x))
// Se recalcula desde x en vez de guardar s(x) para ahorrar memoria
__global__ void kernel_sigmoid_d(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float s = 1.0f / (1.0f + expf(-x[i])); y[i] = s * (1.0f - s); }
}

// ── Softmax estable ──────────────────────────────────────
// Convierte valores crudos en probabilidades que suman 1
// Usa shared memory para reducción paralela eficiente
// "Estable" = resta el máximo antes de exp() para evitar overflow
// Un solo bloque, todos los threads cooperan en el mismo vector
__global__ void kernel_softmax(float* x, int n) {
    extern __shared__ float sdata[]; // memoria compartida entre threads del bloque
    int tid = threadIdx.x;

    // Paso 1: cada thread encuentra el máximo de su porción
    float maxVal = -1e30f;
    for (int i = tid; i < n; i += blockDim.x) maxVal = fmaxf(maxVal, x[i]);
    sdata[tid] = maxVal; __syncthreads();

    // Reducción paralela: árbol de comparaciones hasta quedar con 1 máximo
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]); __syncthreads();
    }
    maxVal = sdata[0]; __syncthreads(); // todos los threads leen el máximo global

    // Paso 2: exp(x - max) y suma — el -max evita exp() de números grandes
    float sumVal = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) { x[i] = expf(x[i] - maxVal); sumVal += x[i]; }
    sdata[tid] = sumVal; __syncthreads();

    // Reducción paralela para la suma total
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s]; __syncthreads();
    }
    sumVal = sdata[0]; __syncthreads();

    // Paso 3: normalizar — cada valor queda entre 0 y 1, todos suman 1
    for (int i = tid; i < n; i += blockDim.x) x[i] /= sumVal;
}

// Multiplicación elemento a elemento: z[i] = x[i] * y[i]
// Se usa en backward para aplicar la derivada de activación al gradiente
__global__ void kernel_mul_ew(const float* x, const float* y, float* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = x[i] * y[i];
}

// Producto matriz-vector con W en row-major: y[r] = sum_c W[r*cols+c] * x[c]
// Evita ambigüedad cuBLAS row/col-major en el backward.
__global__ void kernel_matvec_row(const float* W, const float* x, float* y, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++)
            sum += W[r * cols + c] * x[c];
        y[r] = sum;
    }
}

// Derivada de Cross-Entropy + Softmax combinadas
// La fórmula simplificada es simplemente: delta = prediccion - target
// Ej: predijo [0.1, 0.8, 0.1], target [0, 1, 0] → delta [0.1, -0.2, 0.1]
__global__ void kernel_ce_deriv(const float* yt, const float* yp, float* delta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) delta[i] = yp[i] - yt[i];
}

// ── Optimizadores ────────────────────────────────────────

// SGD: W -= lr * a[r] * delta[c]
// El gradiente de W[r][c] es el producto externo activacion × delta
// Cada thread actualiza un elemento distinto de la matriz W
__global__ void kernel_sgd_update(float* W, const float* a, const float* delta,
                                   float lr, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x; // fila
    int c = blockIdx.y * blockDim.y + threadIdx.y; // columna
    if (r < rows && c < cols)
        W[r * cols + c] -= lr * a[r] * delta[c];
}

// Adam: optimizador adaptativo — ajusta el lr por parámetro
// Mantiene dos momentos por peso:
//   mW = promedio del gradiente (dirección)
//   vW = promedio del gradiente² (velocidad/escala)
// bc1, bc2 = correcciones de sesgo para las primeras iteraciones
__global__ void kernel_adam_update(
    float* W, float* mW, float* vW,
    const float* a, const float* delta,
    float lr, float beta1, float beta2, float eps,
    float bc1, float bc2,
    int rows, int cols)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols) {
        int idx  = r * cols + c;
        float g  = a[r] * delta[c];                              // gradiente crudo
        mW[idx]  = beta1 * mW[idx] + (1.0f - beta1) * g;        // momento 1 (media móvil)
        vW[idx]  = beta2 * vW[idx] + (1.0f - beta2) * g * g;    // momento 2 (varianza móvil)
        W[idx]  -= lr * (mW[idx] * bc1) / (sqrtf(vW[idx] * bc2) + eps); // paso adaptativo
    }
}

// Extrae una submatriz de W saltando las primeras startRow filas
// Se usa para obtener los pesos SIN la fila del bias
// Necesario en backward porque el bias no tiene neurona en la capa anterior
__global__ void kernel_strip_bias_row(const float* W, float* out,
                                       int startRow, int nRows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < nRows && c < cols)
        out[r * cols + c] = W[(startRow + r) * cols + c];
}

// ═══════════════════════════════════════════════════════
// ENUMERACIONES — opciones configurables de la red
// ═══════════════════════════════════════════════════════
enum class Activation { ReLU, LeakyReLU, Sigmoid }; // función de activación capas ocultas
enum class Init       { He, Xavier };                // estrategia de inicialización de pesos
enum class Loss       { CrossEntropy, MSE };         // función de pérdida
enum class Optimizer  { SGD, Adam };                 // algoritmo de optimización

// ═══════════════════════════════════════════════════════
// MLP_CUDA — Perceptrón Multicapa con entrenamiento en GPU
// ═══════════════════════════════════════════════════════
class MLP_CUDA {
public:
    vector<int>    capas;              // arquitectura: [784, 128, 64, 10]
    vector<double> historialLoss;      // loss promedio por época
    vector<double> historialPrecision; // precisión por época

    // Pesos en GPU — se actualizan durante el entrenamiento
    // pesos[l] tiene forma (capas[l]+1) × capas[l+1]  (+1 por bias)
    vector<GpuBuf> d_pesos;
    vector<GpuBuf> d_mw, d_vw; // momentos Adam (solo usados si optim=Adam)

    // Copia CPU de los pesos — sincronizada tras cada época
    // predecir() la usa para inferencia sin overhead GPU
    vector<vector<float>> h_pesos;

    // ── Constructor ──────────────────────────────────────
    MLP_CUDA(const vector<int>& capas_,
             Activation act  = Activation::LeakyReLU,
             Init       init = Init::He,
             Loss       loss = Loss::CrossEntropy,
             Optimizer  opt  = Optimizer::SGD)
        : capas(capas_), actFn(act), lossFn(loss), optim(opt)
    {
        // Verifica que haya GPU disponible
        int count;
        cudaError_t err = cudaGetDeviceCount(&count);
        std::cout << "cudaGetDeviceCount = " << cudaGetErrorString(err) << std::endl;
        std::cout << "count = " << count << std::endl;

        // Inicializa cuBLAS — necesario para cublasSgemv (multiplicación matriz×vector)
        CUBLAS_CHECK(cublasCreate(&handle_));

        cout << "\n==============================\n";
        cout << "CREANDO MLP (CUDA)\n";
        cout << "Capas:       [";
        for (int i = 0; i < (int)capas.size(); i++)
            cout << capas[i] << (i+1<(int)capas.size() ? ", " : "");
        cout << "]\n";
        cout << "Activación:  " << actName()     << "\n";
        cout << "Init:        " << initName(init) << "\n";
        cout << "Pérdida:     " << lossName()     << "\n";
        cout << "Optimizador: " << optName()      << "\n";

        // Imprime el nombre de la GPU detectada
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        cout << "GPU:         " << prop.name << "\n";
        cout << "==============================\n";

        // Inicialización de pesos con distribución normal escalada
        std::mt19937 rng_local(42); // semilla fija → reproducible
        std::normal_distribution<double> dist(0.0, 1.0);

        int L = (int)capas.size() - 1; // número de conexiones entre capas
        d_pesos.reserve(L);
        d_mw.reserve(L);
        d_vw.reserve(L);
        h_pesos.resize(L);

        for (int i = 0; i < L; i++) {
            int rows = capas[i] + 1;  // +1 por el nodo bias
            int cols = capas[i+1];
            int sz   = rows * cols;

            // Escala He para ReLU, Xavier para Sigmoid
            double esc = (init == Init::He)
                ? std::sqrt(2.0 / capas[i])
                : std::sqrt(1.0 / capas[i]);

            // Genera pesos aleatorios en CPU y los sube a GPU
            vector<float> w(sz);
            for (float& x : w) x = (float)(dist(rng_local) * esc);

            d_pesos.emplace_back(sz);
            d_pesos.back().fromCPU(w); // CPU → GPU
            h_pesos[i] = w;            // guarda copia CPU
            d_mw.emplace_back(sz);     // momentos Adam inicializados en 0
            d_vw.emplace_back(sz);
        }
    }

    // Destructor — libera el handle de cuBLAS
    ~MLP_CUDA() { cublasDestroy(handle_); }

    // ═══════════════════════════════════════════════════
    // INTERFAZ PASO A PASO — para integración con CNN
    // En vez de entrenar con un dataset completo,
    // expone forward y backward individuales para que
    // la CNN pueda encadenar sus capas con el MLP
    // ═══════════════════════════════════════════════════

    // Forward individual — recibe vector<float> del flatten layer
    // Devuelve probabilidades softmax de tamaño capas.back()
    // Los buffers se inicializan lazy la primera vez que se llama
    vector<float> stepForward(const vector<float>& x) {
        if (!step_buffers_ready_) initStepBuffers(); // lazy init
        int L = (int)capas.size() - 1;
        int inDim = capas[0];
        assert((int)x.size() == inDim);

        // Copia el vector del flatten a GPU
        step_d_input_.fromCPU(x);

        // Agrega el bias al inicio: [1.0, x[0], x[1], ...]
        {
            int th = 256, bl = (inDim + th - 1) / th;
            kernel_bias<<<bl, th>>>(step_d_input_.ptr, step_d_acts_[0].ptr, inDim);
        }

        // Propaga por todas las capas del MLP
        gpu_forward(step_d_acts_, step_d_nets_, L);

        // Baja la salida softmax a CPU y la devuelve
        vector<float> out;
        step_d_acts_[L].toCPU(out);
        return out;
    }

    // Backward individual — recibe el one-hot del target
    // Devuelve el gradiente respecto al input del MLP
    // (que es la salida del flatten, que es la salida del pooling, etc.)
    // Así el error se propaga de vuelta hacia las capas conv
    vector<float> stepBackward(const vector<float>& target, float lr) {
        int L = (int)capas.size() - 1;
        assert((int)target.size() == capas.back());

        // Sube el one-hot target a GPU
        step_d_target_.fromCPU(target);

        // Backward completo del MLP — actualiza pesos internamente
        gpu_backward(step_d_acts_, step_d_nets_, step_d_deltas_, step_d_wNoBias_,
                     step_d_target_, step_d_actDeriv_, step_d_tmp_, L, lr);

        // Calcula el gradiente respecto al input del MLP
        // Necesita W[0] sin la fila del bias para propagar hacia la CNN
        {
            dim3 bl2((capas[0]+15)/16, (capas[1]+15)/16), th2(16, 16);
            kernel_strip_bias_row<<<bl2, th2>>>(
                d_pesos[0].ptr, step_d_wNoBiasFirst_.ptr, 1, capas[0], capas[1]);
        }
        {
            // grad_input[i] = sum_j W_noBiasFirst[i,j] * delta[0][j]
            // W_noBiasFirst es row-major (capas[0] × capas[1])
            int th = 256, bl = (capas[0] + th - 1) / th;
            kernel_matvec_row<<<bl, th>>>(
                step_d_wNoBiasFirst_.ptr, step_d_deltas_[0].ptr,
                step_d_grad_input_.ptr, capas[0], capas[1]);
        }

        // Baja el gradiente a CPU y lo devuelve al flatten layer
        vector<float> grad;
        step_d_grad_input_.toCPU(grad);
        return grad;
    }

    // Sincroniza pesos GPU → CPU manualmente
    // Se llama al final de cada época o cuando se necesita predecir
    void syncWeights() { syncWeightsToCPU(); }

    // ═══════════════════════════════════════════════════
    // ENTRENAMIENTO STANDALONE — solo disponible si se
    // compila con -DMLPCUDA_WITH_DATASET
    // Para entrenar el MLP solo, sin la CNN
    // ═══════════════════════════════════════════════════
#ifdef MLPCUDA_WITH_DATASET

    // Entrena el MLP standalone con un dataset completo
    // Solo se usa cuando no hay CNN — si hay CNN usa stepForward/stepBackward
    void entrenar(const DatasetInfo& ds, int epocas, float lr, int batchLog = 1) {
        cout << "\n==============================\n";
        cout << "ENTRENAMIENTO — " << ds.nombre << "\n";
        cout << "Muestras train: " << ds.train.X.size() << "\n";

        int N      = (int)ds.train.X.size(); // total de imágenes
        int L      = (int)capas.size() - 1;  // conexiones entre capas
        int outDim = capas.back();            // neuronas de salida (clases)

        // Reserva buffers GPU una sola vez antes del loop
        // Reusar en cada muestra evita cudaMalloc por iteración
        vector<GpuBuf> d_acts(L + 1), d_nets(L);
        for (int l = 0; l <= L; l++)
            d_acts[l] = GpuBuf(l < L ? capas[l] + 1 : capas[l]);
        for (int l = 0; l < L; l++)
            d_nets[l] = GpuBuf(capas[l+1]);

        // Deltas: cuánto error le corresponde a cada neurona por capa
        vector<GpuBuf> d_deltas(L);
        for (int l = 0; l < L; l++) d_deltas[l] = GpuBuf(capas[l+1]);

        // Buffers temporales para backward
        int maxDim = 0;
        for (int d : capas) maxDim = max(maxDim, d + 1);
        GpuBuf d_target(outDim), d_actDeriv(maxDim), d_tmp(maxDim);

        // Pesos sin bias para propagar error en capas ocultas
        vector<GpuBuf> d_wNoBias;
        for (int l = 0; l < L-1; l++)
            d_wNoBias.emplace_back(capas[l+1] * capas[l+2]);

        GpuBuf d_input(capas[0]);
        std::mt19937 rng_local(42);

        for (int ep = 0; ep < epocas; ep++) {
            // Mezcla el orden de muestras cada época
            // Evita que la red aprenda el orden en vez de los patrones
            vector<int> orden(N);
            iota(orden.begin(), orden.end(), 0);
            shuffle(orden.begin(), orden.end(), rng_local);

            double errorTotal = 0.0;

            for (int i : orden) {
                // Convierte imagen double → float y sube a GPU
                const Vector& x = ds.train.X[i];
                int inDim = x.size();
                vector<float> hx(inDim);
                for (int j = 0; j < inDim; j++) hx[j] = (float)x(j);
                d_input.fromCPU(hx);

                // Prepara acts[0] = [1.0, px0, px1, ..., px783]
                {
                    int th = 256, bl = (inDim + th - 1) / th;
                    kernel_bias<<<bl, th>>>(d_input.ptr, d_acts[0].ptr, inDim);
                }

                // Forward — propaga hasta obtener probabilidades softmax
                gpu_forward(d_acts, d_nets, L);

                // Construye one-hot del target y lo sube a GPU
                {
                    vector<float> ht(outDim, 0.0f);
                    ht[ds.train.y[i]] = 1.0f;
                    d_target.fromCPU(ht);
                }

                // Calcula la pérdida en CPU (solo para logging)
                // No afecta el backward — es solo para medir progreso
                {
                    vector<float> hout;
                    d_acts[L].toCPU(hout);
                    double loss = 0.0;
                    const double eps = 1e-15;
                    for (int k = 0; k < outDim; k++) {
                        // Clamp para evitar log(0) = -infinito
                        double yp = max(min((double)hout[k], 1.0-eps), eps);
                        double yt = (ds.train.y[i] == k) ? 1.0 : 0.0;
                        if (lossFn == Loss::CrossEntropy)
                            loss -= yt * log(yp);        // CE: -sum(y * log(yp))
                        else
                            loss += 0.5 * (yp-yt)*(yp-yt) / outDim; // MSE
                    }
                    errorTotal += loss;
                }

                // Backward — calcula deltas y actualiza pesos
                gpu_backward(d_acts, d_nets, d_deltas, d_wNoBias,
                             d_target, d_actDeriv, d_tmp, L, lr);
            }

            // Sincroniza pesos GPU → CPU para que predecir() esté actualizado
            syncWeightsToCPU();

            double errorProm = errorTotal / N;
            double precision = calcularPrecision(ds.train);
            historialLoss.push_back(errorProm);
            historialPrecision.push_back(precision);

            // Imprime cada batchLog épocas
            if ((ep+1) % batchLog == 0) {
                cout << "Época " << setw(3) << (ep+1)
                     << " | Error: "     << fixed << setprecision(6) << errorProm
                     << " | Precisión: " << fixed << setprecision(2) << precision << "%\n";
            }
        }
    }

    // Forward en CPU usando h_pesos — para predecir sin overhead GPU
    // Se usa después de entrenar, cuando los pesos ya están sincronizados
    Vector predecir(const Vector& x) const {
        int L = (int)capas.size() - 1;

        // Prepara activaciones iniciales con bias: [1.0, x[0], ..., x[n-1]]
        vector<float> a(capas[0] + 1);
        a[0] = 1.0f;
        for (int j = 0; j < x.size(); j++) a[j+1] = (float)x(j);

        for (int l = 0; l < L; l++) {
            int rows = capas[l] + 1;
            int cols = capas[l+1];

            // net = W^T × a — multiplicación manual en CPU
            vector<float> net(cols, 0.0f);
            for (int c = 0; c < cols; c++)
                for (int r = 0; r < rows; r++)
                    net[c] += h_pesos[l][r * cols + c] * a[r];

            if (l < L - 1) {
                // Capa oculta: activación + bias para siguiente capa
                vector<float> newa(cols + 1);
                newa[0] = 1.0f;
                for (int j = 0; j < cols; j++) newa[j+1] = applyCPUAct(net[j]);
                a = newa;
            } else {
                // Capa de salida: softmax estable
                float maxv = *max_element(net.begin(), net.end());
                float sumv = 0.0f;
                for (float& v : net) { v = expf(v - maxv); sumv += v; }
                for (float& v : net) v /= sumv;
                a = net;
            }
        }

        // Convierte vector<float> → Vector (Eigen) para compatibilidad
        Vector out(capas.back());
        for (int i = 0; i < capas.back(); i++) out(i) = a[i];
        return out;
    }

    // Calcula el porcentaje de aciertos sobre un split
    // Compara argmax(prediccion) con la etiqueta real
    double calcularPrecision(const Split& s) const {
        int ok = 0;
        for (int i = 0; i < (int)s.X.size(); i++)
            if (argmax(predecir(s.X[i])) == s.y[i]) ok++;
        return 100.0 * ok / (int)s.X.size();
    }

#endif // MLPCUDA_WITH_DATASET

private:
    Activation     actFn;   // activación de capas ocultas
    Loss           lossFn;  // función de pérdida
    Optimizer      optim;   // SGD o Adam
    cublasHandle_t handle_; // handle cuBLAS — reusado en cada llamada
    int            adam_t = 0; // contador de pasos Adam para corrección de sesgo

    // ═══════════════════════════════════════════════════
    // BUFFERS STEP — memoria GPU para stepForward/Backward
    // Se inicializan una sola vez (lazy) y se reusan
    // en cada llamada para evitar cudaMalloc por muestra
    // ═══════════════════════════════════════════════════
    vector<GpuBuf> step_d_acts_;        // activaciones por capa
    vector<GpuBuf> step_d_nets_;        // valores antes de activación
    vector<GpuBuf> step_d_deltas_;      // deltas del backward
    vector<GpuBuf> step_d_wNoBias_;     // pesos sin fila bias (backward ocultas)
    GpuBuf         step_d_target_;      // one-hot del target
    GpuBuf         step_d_actDeriv_;    // derivada de activación (temporal)
    GpuBuf         step_d_tmp_;         // multiplicación temporal del backward
    GpuBuf         step_d_input_;       // input del MLP (salida del flatten)
    GpuBuf         step_d_grad_input_;  // gradiente respecto al input → va a CNN
    GpuBuf         step_d_wNoBiasFirst_; // W[0] sin bias para calcular grad_input
    bool           step_buffers_ready_ = false; // flag de inicialización lazy

    // Inicializa todos los buffers GPU la primera vez que se llama stepForward
    // Lazy para no reservar memoria si solo se usa entrenar() standalone
    void initStepBuffers() {
        int L = (int)capas.size() - 1;
        int outDim = capas.back();
        int maxDim = 0;
        for (int d : capas) maxDim = max(maxDim, d + 1);

        // Limpia buffers anteriores si existían
        step_d_acts_.clear();
        step_d_nets_.clear();
        step_d_deltas_.clear();
        step_d_wNoBias_.clear();

        // Reserva buffers del mismo tamaño que en entrenar()
        for (int l = 0; l <= L; l++)
            step_d_acts_.emplace_back(l < L ? capas[l] + 1 : capas[l]);
        for (int l = 0; l < L; l++)
            step_d_nets_.emplace_back(capas[l+1]);
        for (int l = 0; l < L; l++)
            step_d_deltas_.emplace_back(capas[l+1]);
        for (int l = 0; l < L-1; l++)
            step_d_wNoBias_.emplace_back(capas[l+1] * capas[l+2]);

        step_d_target_       = GpuBuf(outDim);
        step_d_actDeriv_     = GpuBuf(maxDim);
        step_d_tmp_          = GpuBuf(maxDim);
        step_d_input_        = GpuBuf(capas[0]);
        step_d_grad_input_   = GpuBuf(capas[0]); // grad que va de vuelta a CNN
        step_d_wNoBiasFirst_ = GpuBuf(capas[0] * capas[1]); // W[0] sin bias
        step_buffers_ready_  = true;
    }

    // ── Forward GPU ──────────────────────────────────────
    // Por cada capa: net = W·a (cuBLAS) → activación → acts siguiente
    // Última capa: softmax en vez de ReLU/LeakyReLU/Sigmoid
    void gpu_forward(vector<GpuBuf>& d_acts, vector<GpuBuf>& d_nets, int L) {
        const float alpha = 1.0f, beta = 0.0f;
        for (int l = 0; l < L; l++) {
            int rows = capas[l] + 1;
            int cols = capas[l+1];

            // net[l] = W[l]^T × acts[l]
            // cuBLAS trabaja en col-major, W está en row-major
            // CUBLAS_OP_N sobre la vista col-major equivale a transponer
            CUBLAS_CHECK(cublasSgemv(
                handle_, CUBLAS_OP_N,
                cols, rows, &alpha, // dimesiones
                d_pesos[l].ptr, cols, //matriz de pesos
                d_acts[l].ptr, 1, // matriz de activaciones de entrada
                &beta, 
                d_nets[l].ptr, 1)); // resultado net[l]

            if (l < L - 1) {
                // Capa oculta: activación + agregar bias para la siguiente
                applyActKernel(d_nets[l].ptr, d_acts[l+1].ptr + 1, cols);
                float one = 1.0f;
                // Escribe el 1.0 del bias en la posición 0 de acts[l+1]
                CUDA_CHECK(cudaMemcpy(d_acts[l+1].ptr, &one, sizeof(float), cudaMemcpyHostToDevice));
            } else {
                // Capa de salida: softmax estable en vez de activación simple
                int smTh = nextPow2(min(cols, 1024));
                kernel_softmax<<<1, smTh, smTh * sizeof(float)>>>(d_nets[l].ptr, cols);
                // Copia net[L] → acts[L] (softmax modifica in-place net[L])
                CUDA_CHECK(cudaMemcpy(d_acts[L].ptr, d_nets[l].ptr,
                                      cols * sizeof(float), cudaMemcpyDeviceToDevice));
            }
        }
    }

    // Aplica la función de activación configurada sobre src → dst
    void applyActKernel(const float* src, float* dst, int n) {
        int th = 256, bl = (n + th - 1) / th;
        switch (actFn) {
            case Activation::ReLU:      kernel_relu<<<bl,th>>>(src, dst, n); break;
            case Activation::LeakyReLU: kernel_leaky_relu<<<bl,th>>>(src, dst, n, 0.01f); break;
            case Activation::Sigmoid:   kernel_sigmoid<<<bl,th>>>(src, dst, n); break;
        }
    }

    // Aplica la derivada de la activación configurada sobre src → dst
    // Se usa en backward para calcular los deltas de capas ocultas
    void applyActDerivKernel(const float* src, float* dst, int n) {
        int th = 256, bl = (n + th - 1) / th;
        switch (actFn) {
            case Activation::ReLU:      kernel_relu_d<<<bl,th>>>(src, dst, n); break;
            case Activation::LeakyReLU: kernel_leaky_relu_d<<<bl,th>>>(src, dst, n, 0.01f); break;
            case Activation::Sigmoid:   kernel_sigmoid_d<<<bl,th>>>(src, dst, n); break;
        }
    }

    // ── Backward GPU ─────────────────────────────────────
    // 1. delta salida = softmax_out - target  (CE+softmax combinado)
    // 2. propaga deltas hacia atrás por capas ocultas
    // 3. actualiza todos los pesos con SGD o Adam
    void gpu_backward(
        vector<GpuBuf>& d_acts, vector<GpuBuf>& d_nets,
        vector<GpuBuf>& d_deltas, vector<GpuBuf>& d_wNoBias,
        GpuBuf& d_target, GpuBuf& d_actDeriv, GpuBuf& d_tmp,
        int L, float lr)
    {
        // Delta de la capa de salida: prediccion - target
        {
            int cols = capas[L];
            int th = 256, bl = (cols + th - 1) / th;
            kernel_ce_deriv<<<bl, th>>>(d_target.ptr, d_acts[L].ptr, d_deltas[L-1].ptr, cols);
        }

        // Propagar error hacia capas ocultas (de atrás hacia adelante)
        const float alpha = 1.0f, beta = 0.0f;
        for (int l = L-2; l >= 0; l--) {
            int rows      = capas[l+1];
            int cols_next = capas[l+2];
            int cols_cur  = capas[l+1];

            // Extrae pesos sin la fila del bias para no propagar error al nodo bias
            {
                dim3 bl2((rows + 15)/16, (cols_next + 15)/16), th2(16, 16);
                kernel_strip_bias_row<<<bl2, th2>>>(
                    d_pesos[l+1].ptr, d_wNoBias[l].ptr, 1, rows, cols_next);
            }

            // tmp = W_noBias × delta[l+1]   (W_noBias es row-major rows×cols_next)
            {
                int th = 256, bl = (rows + th - 1) / th;
                kernel_matvec_row<<<bl, th>>>(
                    d_wNoBias[l].ptr, d_deltas[l+1].ptr, d_tmp.ptr, rows, cols_next);
            }

            // actDeriv = ReLU'(nets[l]) — puerta que filtra neuronas bloqueadas
            applyActDerivKernel(d_nets[l].ptr, d_actDeriv.ptr, cols_cur);

            // delta[l] = tmp × actDeriv — error ponderado por activación
            {
                int th = 256, bl = (cols_cur + th - 1) / th;
                kernel_mul_ew<<<bl, th>>>(d_tmp.ptr, d_actDeriv.ptr, d_deltas[l].ptr, cols_cur);
            }
        }

        // Actualizar pesos con el optimizador configurado
        if (optim == Optimizer::SGD) {
            for (int l = 0; l < L; l++) {
                int rows = capas[l] + 1, cols = capas[l+1];
                dim3 bl2((rows+15)/16, (cols+15)/16), th2(16,16);
                // W[l] -= lr × acts[l] × delta[l]  (producto externo)
                kernel_sgd_update<<<bl2, th2>>>(
                    d_pesos[l].ptr, d_acts[l].ptr, d_deltas[l].ptr, lr, rows, cols);
            }
        } else {
            // Adam: corrección de sesgo para las primeras iteraciones
            // bc1, bc2 → 1/(1-beta^t) → se acercan a 1 con el tiempo
            const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
            adam_t++;
            float bc1 = 1.0f / (1.0f - powf(beta1, (float)adam_t));
            float bc2 = 1.0f / (1.0f - powf(beta2, (float)adam_t));
            for (int l = 0; l < L; l++) {
                int rows = capas[l] + 1, cols = capas[l+1];
                dim3 bl2((rows+15)/16, (cols+15)/16), th2(16,16);
                kernel_adam_update<<<bl2, th2>>>(
                    d_pesos[l].ptr, d_mw[l].ptr, d_vw[l].ptr,
                    d_acts[l].ptr, d_deltas[l].ptr,
                    lr, beta1, beta2, eps, bc1, bc2, rows, cols);
            }
        }
    }

    // Copia todos los pesos de GPU → CPU
    // Se llama al final de cada época para que predecir() esté actualizado
    void syncWeightsToCPU() {
        for (int l = 0; l < (int)d_pesos.size(); l++)
            d_pesos[l].toCPU(h_pesos[l]);
    }

    // Forward en CPU — para predecir sin overhead de GPU
    float applyCPUAct(float x) const {
        switch (actFn) {
            case Activation::ReLU:      return x > 0.0f ? x : 0.0f;
            case Activation::LeakyReLU: return x > 0.0f ? x : 0.01f * x;
            case Activation::Sigmoid:   return 1.0f / (1.0f + expf(-x));
        }
        return x > 0.0f ? x : 0.0f;
    }

    // Redondea n al siguiente potencia de 2
    // Se usa para el tamaño del bloque del kernel softmax
    // (la reducción paralela requiere potencia de 2)
    static int nextPow2(int n) { int p=1; while(p<n) p<<=1; return p; }

    // Helpers para imprimir configuración en el constructor
    string actName()  const {
        if (actFn == Activation::ReLU)      return "ReLU";
        if (actFn == Activation::LeakyReLU) return "LeakyReLU";
        return "Sigmoid";
    }
    string initName(Init i) const { return (i==Init::He)?"He":"Xavier"; }
    string lossName() const { return (lossFn==Loss::CrossEntropy)?"CrossEntropy":"MSE"; }
    string optName()  const { return (optim==Optimizer::SGD)?"SGD":"Adam"; }
};