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

// ============================================================
// Macros de chequeo de errores CUDA
// ============================================================
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


// Utilidades CPU
inline Vector oneHot(int label, int nClases) {
    Vector v = Vector::Zero(nClases);
    v(label) = 1.0;
    return v;
}

inline int argmax(const Vector& v) {
    int idx = 0;
    for (int i=1; i<v.size(); i++)
        if (v(i) > v(idx)) idx = i;
    return idx;
}

// argmax para vector<float>
inline int argmaxf(const vector<float>& v) {
    return (int)(max_element(v.begin(), v.end()) - v.begin());
}

struct GpuBuf {
    float* ptr = nullptr;
    int    n   = 0;

    GpuBuf() = default;
    // Constructor que reserva memoria en GPU para n floats e inicializa a 0
    explicit GpuBuf(int size) : n(size) {
        CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(float)));
        CUDA_CHECK(cudaMemset(ptr, 0, size * sizeof(float)));
    }
    // Destructor
    ~GpuBuf() { if (ptr) cudaFree(ptr); }

    // No se permiten copias
    GpuBuf(const GpuBuf&)            = delete;
    GpuBuf& operator=(const GpuBuf&) = delete;

    // se permiten movimientos
    GpuBuf(GpuBuf&& o) noexcept : ptr(o.ptr), n(o.n) { o.ptr = nullptr; o.n = 0; }
    GpuBuf& operator=(GpuBuf&& o) noexcept {
        if (ptr) cudaFree(ptr);
        ptr = o.ptr; n = o.n; o.ptr = nullptr; o.n = 0;
        return *this;
    }

    // Copiar datos entre CPU y GPU
    void fromCPU(const vector<float>& h) {
        assert((int)h.size() == n);
        CUDA_CHECK(cudaMemcpy(ptr, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }
    void toCPU(vector<float>& h) const {
        h.resize(n);
        CUDA_CHECK(cudaMemcpy(h.data(), ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
    }
    // Inicializar a 0
    void zero() { CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(float))); }
};

// Kernels CUDA
// Agregar nodo bias al inicio: y = [1, x[0..n-1]]
__global__ void kernel_bias(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) y[0] = 1.0f;
    if (i < n)  y[i + 1] = x[i];
}

// Funciones de activación y sus derivadas
__global__ void kernel_relu(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}
__global__ void kernel_relu_d(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? 1.0f : 0.0f;
}
__global__ void kernel_leaky_relu(const float* x, float* y, int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? x[i] : alpha * x[i];
}
__global__ void kernel_leaky_relu_d(const float* x, float* y, int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? 1.0f : alpha;
}
__global__ void kernel_sigmoid(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.0f / (1.0f + expf(-x[i]));
}
__global__ void kernel_sigmoid_d(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float s = 1.0f / (1.0f + expf(-x[i])); y[i] = s * (1.0f - s); }
}

// Softmax estable (un bloque, shared memory)
__global__ void kernel_softmax(float* x, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float maxVal = -1e30f;
    for (int i = tid; i < n; i += blockDim.x) maxVal = fmaxf(maxVal, x[i]);
    sdata[tid] = maxVal; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]); __syncthreads();
    }
    maxVal = sdata[0]; __syncthreads();

    float sumVal = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) { x[i] = expf(x[i] - maxVal); sumVal += x[i]; }
    sdata[tid] = sumVal; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s]; __syncthreads();
    }
    sumVal = sdata[0]; __syncthreads();
    for (int i = tid; i < n; i += blockDim.x) x[i] /= sumVal;
}

// Element-wise: z = x * y
__global__ void kernel_mul_ew(const float* x, const float* y, float* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = x[i] * y[i];
}

// Derivada CE+softmax: delta = yp - yt
__global__ void kernel_ce_deriv(const float* yt, const float* yp, float* delta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) delta[i] = yp[i] - yt[i];
}

// FUNCIONES DE OPTMIZACION
// SGD: W -= lr * outer(a, delta)
__global__ void kernel_sgd_update(float* W, const float* a, const float* delta,
                                   float lr, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols)
        W[r * cols + c] -= lr * a[r] * delta[c];
}

// Adam: actualiza W, mW, vW con gradiente outer(a, delta)
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
        float g  = a[r] * delta[c];
        mW[idx]  = beta1 * mW[idx] + (1.0f - beta1) * g;
        vW[idx]  = beta2 * vW[idx] + (1.0f - beta2) * g * g;
        W[idx]  -= lr * (mW[idx] * bc1) / (sqrtf(vW[idx] * bc2) + eps);
    }
}

// Extrae filas [startRow, startRow+nRows) de W (cols columnas) → out
__global__ void kernel_strip_bias_row(const float* W, float* out,
                                       int startRow, int nRows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < nRows && c < cols)
        out[r * cols + c] = W[(startRow + r) * cols + c];
}

// ============================================================
// Enumeraciones
// ============================================================

enum class Activation { ReLU, LeakyReLU, Sigmoid };
enum class Init       { He, Xavier };
enum class Loss       { CrossEntropy, MSE };
enum class Optimizer  { SGD, Adam };

// ============================================================
// MLP_CUDA
// ============================================================

class MLP_CUDA {
public:
    vector<int>    capas;
    vector<double> historialLoss;
    vector<double> historialPrecision;

    // Pesos en GPU: pesos[l] tiene forma (capas[l]+1) x capas[l+1]
    vector<GpuBuf> d_pesos;
    vector<GpuBuf> d_mw, d_vw;   // momentos Adam

    // Copia CPU para predicción (sincronizada tras cada época)
    vector<vector<float>> h_pesos;

    // CONSTRUCTOR y DESTRUCTOR
    MLP_CUDA(const vector<int>& capas_,
             Activation act  = Activation::LeakyReLU,
             Init       init = Init::He,
             Loss       loss = Loss::CrossEntropy,
             Optimizer  opt  = Optimizer::SGD)
        : capas(capas_), actFn(act), lossFn(loss), optim(opt)
    {
        // Obtener info de GPU
        int count;
        cudaError_t err = cudaGetDeviceCount(&count);

        std::cout << "cudaGetDeviceCount = "
                << cudaGetErrorString(err)
                << std::endl;

        // count = 0 → no hay GPU disponible o no se pudo acceder a ella
        std::cout << "count = " << count << std::endl;
        // CUBLAS_CHECK (cublasCreate(&handle_));  // Crear handle de cuBLAS para uso posterior
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

        // cudaDeviceProp → información detallada de la GPU
        cudaDeviceProp prop;
        // Obtener propiedades de la GPU (índice 0)
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        cout << "GPU:         " << prop.name << "\n";
        cout << "==============================\n";

        // generador de números aleatorios para inicialización
        std::mt19937 rng_local(42);
        std::normal_distribution<double> dist(0.0, 1.0);

        // L - 1 = número de capas ocultas + salida /  conexion entre capas
        int L = (int)capas.size() - 1;
        // pesos w
        d_pesos.reserve(L);
        // momentos Adam
        d_mw.reserve(L);  // primer momento - promedio de la gradiente - direccion
        d_vw.reserve(L);  // segundo momento - promedio de la gradiente al cuadrado - velocidad
        // copia CPU de pesos para predicción
        h_pesos.resize(L);

        for (int i = 0; i < L; i++) {
            int rows = capas[i] + 1; // neuronas de entrada + bias
            int cols = capas[i+1]; // neuronas de salida
            int sz   = rows * cols; // tamaño total de pesos en esta capa
            double esc = (init == Init::He)
                ? std::sqrt(2.0 / capas[i]) // He para ReLU
                : std::sqrt(1.0 / capas[i]); // Xavier para Sigmoid

            // creamos un vector de pesos en CPU con distribución normal y escala adecuada
            vector<float> w(sz);
            for (float& x : w)
                x = (float)(dist(rng_local) * esc); // aleatorio [0,1] por escala

            // reservamos memoria en GPU para los pesos, copiamos los pesos iniciales
            d_pesos.emplace_back(sz);
            d_pesos.back().fromCPU(w);
            // Guardamos la copia CPU para predicción (sincronizada tras cada época)
            h_pesos[i] = w;
            // Si el optimizador es Adam, reservamos buffers para momentos mW y vW (inicializados a 0)
            d_mw.emplace_back(sz);
            d_vw.emplace_back(sz);
        }
    }

    ~MLP_CUDA() { cublasDestroy(handle_); }

    // ----------------------------------------------------------
    // Interfaz paso a paso para integración con CNN
    // ----------------------------------------------------------

    // Forward: x es vector<float> de tamaño capas[0]
    // Retorna probabilidades softmax de tamaño capas.back()
    vector<float> stepForward(const vector<float>& x) {
        if (!step_buffers_ready_) initStepBuffers();
        int L = (int)capas.size() - 1;
        int inDim = capas[0];
        assert((int)x.size() == inDim);

        step_d_input_.fromCPU(x);
        {
            int th = 256, bl = (inDim + th - 1) / th;
            kernel_bias<<<bl, th>>>(step_d_input_.ptr, step_d_acts_[0].ptr, inDim);
        }
        gpu_forward(step_d_acts_, step_d_nets_, L);

        vector<float> out;
        step_d_acts_[L].toCPU(out);
        return out;
    }

    // Backward: llamar después de stepForward
    // target: one-hot vector<float> de tamaño capas.back()
    // Retorna gradiente respecto al input (tamaño capas[0])
    vector<float> stepBackward(const vector<float>& target, float lr) {
        int L = (int)capas.size() - 1;
        assert((int)target.size() == capas.back());

        step_d_target_.fromCPU(target);

        gpu_backward(step_d_acts_, step_d_nets_, step_d_deltas_, step_d_wNoBias_,
                     step_d_target_, step_d_actDeriv_, step_d_tmp_, L, lr);

        // Gradiente respecto al input: W[0][1:, :] × delta[0]
        // W[0][1:, :] es capas[0] × capas[1] (excluyendo fila de bias)
        {
            dim3 bl2((capas[0]+15)/16, (capas[1]+15)/16), th2(16, 16);
            kernel_strip_bias_row<<<bl2, th2>>>(
                d_pesos[0].ptr, step_d_wNoBiasFirst_.ptr, 1, capas[0], capas[1]);
        }
        {
            const float alpha = 1.0f, beta = 0.0f;
            // W_noBias (capas[0] x capas[1]) row-major → col-major (capas[1] x capas[0])
            // Queremos W_noBias * delta[0] → CUBLAS_OP_T
            CUBLAS_CHECK(cublasSgemv(
                handle_, CUBLAS_OP_T,
                capas[1], capas[0],
                &alpha,
                step_d_wNoBiasFirst_.ptr, capas[1],
                step_d_deltas_[0].ptr, 1,
                &beta,
                step_d_grad_input_.ptr, 1));
        }

        vector<float> grad;
        step_d_grad_input_.toCPU(grad);
        return grad;
    }

    // Sincroniza pesos GPU → CPU (para predicción o al terminar época)
    void syncWeights() { syncWeightsToCPU(); }

    // ----------------------------------------------------------
    // Entrenamiento
    // ----------------------------------------------------------

#ifdef MLPCUDA_WITH_DATASET
    void entrenar(const DatasetInfo& ds, int epocas, float lr, int batchLog = 1) {
        cout << "\n==============================\n";
        cout << "ENTRENAMIENTO — " << ds.nombre << "\n";
        cout << "Muestras train: " << ds.train.X.size() << "\n";

        int N = (int)ds.train.X.size(); // imagenes de entrenamiento
        int L = (int)capas.size() - 1; // conexiones
        int outDim = capas.back(); //neuronas de salida - clases

        // Buffers de activaciones en GPU
        vector<GpuBuf> d_acts(L + 1), d_nets(L);
        // Guardar activaciones
        for (int l = 0; l <= L; l++)
            d_acts[l] = GpuBuf(l < L ? capas[l] + 1 : capas[l]);
        // Guardar net inputs (sin activación) para cada capa (excepto input)
        for (int l = 0; l < L; l++)
            d_nets[l] = GpuBuf(capas[l+1]);

        // Guarda cuanto error se comete en cada capa para cada muestra (para backprop)
        vector<GpuBuf> d_deltas(L);
        for (int l = 0; l < L; l++) d_deltas[l] = GpuBuf(capas[l+1]);

        int maxDim = 0;
        for (int d : capas) maxDim = max(maxDim, d + 1);
        // Buffers temporales para backprop:  target (oen-hot), derivada de activación, multiplicación temporal
        GpuBuf d_target(outDim), d_actDeriv(maxDim), d_tmp(maxDim);

        // wNoBias[l]: capas[l+1] x capas[l+2]  (para backprop capas ocultas)
        vector<GpuBuf> d_wNoBias;
        for (int l = 0; l < L-1; l++)
            d_wNoBias.emplace_back(capas[l+1] * capas[l+2]);

        // Buffer temporal para el input
        GpuBuf d_input(capas[0]);

        // generador de números aleatorios para orden de muestras en cada época
        std::mt19937 rng_local(42);

        // Bucle de entrenamiento por épocas
        for (int ep = 0; ep < epocas; ep++) {
            vector<int> orden(N);
            // llenar orden con índices 0..N-1 y mezclarlo aleatoriamente para cada época
            iota(orden.begin(), orden.end(), 0);
            shuffle(orden.begin(), orden.end(), rng_local);

            double errorTotal = 0.0;

            for (int i : orden) {
                // Copiar input → GPU
                const Vector& x = ds.train.X[i];
                int inDim = x.size();
                vector<float> hx(inDim);
                for (int j = 0; j < inDim; j++) hx[j] = (float)x(j);
                d_input.fromCPU(hx);

                // acts[0] = [1, x]
                {
                    int th = 256, bl = (inDim + th - 1) / th;
                    kernel_bias<<<bl, th>>>(d_input.ptr, d_acts[0].ptr, inDim);
                }

                // Forward
                gpu_forward(d_acts, d_nets, L);

                // Target
                {
                    vector<float> ht(outDim, 0.0f);
                    ht[ds.train.y[i]] = 1.0f;
                    d_target.fromCPU(ht);
                }

                // Pérdida (en CPU sobre la salida copiada)
                {
                    vector<float> hout;
                    d_acts[L].toCPU(hout);
                    double loss = 0.0;
                    const double eps = 1e-15;
                    for (int k = 0; k < outDim; k++) {
                        double yp = max(min((double)hout[k], 1.0-eps), eps);
                        double yt = (ds.train.y[i] == k) ? 1.0 : 0.0;
                        if (lossFn == Loss::CrossEntropy)
                            loss -= yt * log(yp);
                        else
                            loss += 0.5 * (yp - yt) * (yp - yt) / outDim;
                    }
                    errorTotal += loss;
                }

                // Backward + update
                gpu_backward(d_acts, d_nets, d_deltas, d_wNoBias,
                             d_target, d_actDeriv, d_tmp, L, lr);
            }

            syncWeightsToCPU();

            double errorProm = errorTotal / N;
            double precision = calcularPrecision(ds.train);
            historialLoss.push_back(errorProm);
            historialPrecision.push_back(precision);

            if ((ep+1) % batchLog == 0) {
                cout << "Época " << setw(3) << (ep+1)
                     << " | Error: "     << fixed << setprecision(6) << errorProm
                     << " | Precisión: " << fixed << setprecision(2) << precision << "%\n";
            }
        }
    }

    // ----------------------------------------------------------
    // Predicción
    // ----------------------------------------------------------
    Vector predecir(const Vector& x) const {
        // L = número de capas ocultas + salida /  conexion entre capas
        int L = (int)capas.size() - 1;
        // a = activaciones de la capa actual (inicialmente input con bias)
        vector<float> a(capas[0] + 1);
        a[0] = 1.0f;
        for (int j = 0; j < x.size(); j++)
            a[j+1] = (float)x(j);

        // Propagar hacia adelante por cada capa usando los pesos CPU
        for (int l = 0; l < L; l++) {
            // net = W^T * a
            int rows = capas[l] + 1;
            int cols = capas[l+1];
            vector<float> net(cols, 0.0f);
            for (int c = 0; c < cols; c++)
                for (int r = 0; r < rows; r++)
                    net[c] += h_pesos[l][r * cols + c] * a[r];

            // si no es capa de salida, aplicar activación y agregar bias para la siguiente capa
            if (l < L - 1) {
                vector<float> newa(cols + 1);
                newa[0] = 1.0f;
                for (int j = 0; j < cols; j++) {
                    newa[j+1] = applyCPUAct(net[j]);
                }
                a = newa;
            } else {
                // capa de salida: aplicar softmax estable (0,1; 0,2 ; etc)
                float maxv = *max_element(net.begin(), net.end());
                float sumv = 0.0f;
                for (float& v : net) { v = expf(v - maxv); sumv += v; }
                for (float& v : net) v /= sumv;
                a = net;
            }
        }

        // Convertir salida a Vector
        Vector out(capas.back());
        for (int i = 0; i < capas.back(); i++) out(i) = a[i];
        return out;
    }

    double calcularPrecision(const Split& s) const {
        int ok = 0;
        for (int i = 0; i < (int)s.X.size(); i++)
            if (argmax(predecir(s.X[i])) == s.y[i]) ok++;
        return 100.0 * ok / (int)s.X.size();
    }
#endif // MLPCUDA_WITH_DATASET

private:
    Activation     actFn;
    Loss           lossFn;
    Optimizer      optim;
    cublasHandle_t handle_;
    int            adam_t = 0;

    // Buffers para stepForward/stepBackward (inicializados lazy)
    vector<GpuBuf> step_d_acts_;
    vector<GpuBuf> step_d_nets_;
    vector<GpuBuf> step_d_deltas_;
    vector<GpuBuf> step_d_wNoBias_;
    GpuBuf         step_d_target_;
    GpuBuf         step_d_actDeriv_;
    GpuBuf         step_d_tmp_;
    GpuBuf         step_d_input_;
    GpuBuf         step_d_grad_input_;
    GpuBuf         step_d_wNoBiasFirst_;
    bool           step_buffers_ready_ = false;

    void initStepBuffers() {
        int L = (int)capas.size() - 1;
        int outDim = capas.back();
        int maxDim = 0;
        for (int d : capas) maxDim = max(maxDim, d + 1);

        step_d_acts_.clear();
        step_d_nets_.clear();
        step_d_deltas_.clear();
        step_d_wNoBias_.clear();

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
        step_d_grad_input_   = GpuBuf(capas[0]);
        step_d_wNoBiasFirst_ = GpuBuf(capas[0] * capas[1]);
        step_buffers_ready_  = true;
    }

    // ----------------------------------------------------------
    // Forward en GPU
    // ----------------------------------------------------------

    void gpu_forward(vector<GpuBuf>& d_acts, vector<GpuBuf>& d_nets, int L) {
        const float alpha = 1.0f, beta = 0.0f;
        for (int l = 0; l < L; l++) {
            int rows = capas[l] + 1;
            int cols = capas[l+1];

            // net = W^T * a
            // W row-major (rows x cols) ≡ cuBLAS col-major (cols x rows)
            // Queremos W^T * a → CUBLAS_OP_N sobre la vista col-major
            CUBLAS_CHECK(cublasSgemv(
                handle_, CUBLAS_OP_N,
                cols, rows,
                &alpha,
                d_pesos[l].ptr, cols,
                d_acts[l].ptr, 1,
                &beta,
                d_nets[l].ptr, 1));

            if (l < L - 1) {
                applyActKernel(d_nets[l].ptr, d_acts[l+1].ptr + 1, cols);
                float one = 1.0f;
                CUDA_CHECK(cudaMemcpy(d_acts[l+1].ptr, &one, sizeof(float), cudaMemcpyHostToDevice));
            } else {
                int smTh = nextPow2(min(cols, 1024));
                kernel_softmax<<<1, smTh, smTh * sizeof(float)>>>(d_nets[l].ptr, cols);
                CUDA_CHECK(cudaMemcpy(d_acts[L].ptr, d_nets[l].ptr,
                                      cols * sizeof(float), cudaMemcpyDeviceToDevice));
            }
        }
    }

    void applyActKernel(const float* src, float* dst, int n) {
        int th = 256, bl = (n + th - 1) / th;
        switch (actFn) {
            case Activation::ReLU:      kernel_relu<<<bl,th>>>(src, dst, n); break;
            case Activation::LeakyReLU: kernel_leaky_relu<<<bl,th>>>(src, dst, n, 0.01f); break;
            case Activation::Sigmoid:   kernel_sigmoid<<<bl,th>>>(src, dst, n); break;
        }
    }

    void applyActDerivKernel(const float* src, float* dst, int n) {
        int th = 256, bl = (n + th - 1) / th;
        switch (actFn) {
            case Activation::ReLU:      kernel_relu_d<<<bl,th>>>(src, dst, n); break;
            case Activation::LeakyReLU: kernel_leaky_relu_d<<<bl,th>>>(src, dst, n, 0.01f); break;
            case Activation::Sigmoid:   kernel_sigmoid_d<<<bl,th>>>(src, dst, n); break;
        }
    }

    // ----------------------------------------------------------
    // Backward en GPU
    // ----------------------------------------------------------

    void gpu_backward(
        vector<GpuBuf>& d_acts,
        vector<GpuBuf>& d_nets,
        vector<GpuBuf>& d_deltas,
        vector<GpuBuf>& d_wNoBias,
        GpuBuf& d_target,
        GpuBuf& d_actDeriv,
        GpuBuf& d_tmp,
        int L, float lr)
    {
        // delta[L-1] = softmax_out - target
        {
            int cols = capas[L];
            int th = 256, bl = (cols + th - 1) / th;
            kernel_ce_deriv<<<bl, th>>>(d_target.ptr, d_acts[L].ptr, d_deltas[L-1].ptr, cols);
        }

        // Propagar hacia atrás por capas ocultas
        const float alpha = 1.0f, beta = 0.0f;
        for (int l = L-2; l >= 0; l--) {
            int rows      = capas[l+1];
            int cols_next = capas[l+2];
            int cols_cur  = capas[l+1];

            // Extraer wNoBias = pesos[l+1] sin la fila 0 (bias)
            {
                dim3 bl2((rows + 15)/16, (cols_next + 15)/16), th2(16, 16);
                kernel_strip_bias_row<<<bl2, th2>>>(
                    d_pesos[l+1].ptr, d_wNoBias[l].ptr, 1, rows, cols_next);
            }

            // tmp = wNoBias * delta[l+1]
            // wNoBias (rows x cols_next) row-major → cuBLAS col-major (cols_next x rows)
            // Queremos wNoBias * delta → CUBLAS_OP_T
            CUBLAS_CHECK(cublasSgemv(
                handle_, CUBLAS_OP_T,
                cols_next, rows,
                &alpha,
                d_wNoBias[l].ptr, cols_next,
                d_deltas[l+1].ptr, 1,
                &beta,
                d_tmp.ptr, 1));

            applyActDerivKernel(d_nets[l].ptr, d_actDeriv.ptr, cols_cur);

            {
                int th = 256, bl = (cols_cur + th - 1) / th;
                kernel_mul_ew<<<bl, th>>>(d_tmp.ptr, d_actDeriv.ptr, d_deltas[l].ptr, cols_cur);
            }
        }

        // Actualizar pesos
        if (optim == Optimizer::SGD) {
            for (int l = 0; l < L; l++) {
                int rows = capas[l] + 1, cols = capas[l+1];
                dim3 bl2((rows+15)/16, (cols+15)/16), th2(16,16);
                kernel_sgd_update<<<bl2, th2>>>(
                    d_pesos[l].ptr, d_acts[l].ptr, d_deltas[l].ptr, lr, rows, cols);
            }
        } else {
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

    // ----------------------------------------------------------
    // Sync pesos GPU → CPU
    // ----------------------------------------------------------

    void syncWeightsToCPU() {
        for (int l = 0; l < (int)d_pesos.size(); l++)
            d_pesos[l].toCPU(h_pesos[l]);
    }

    float applyCPUAct(float x) const {
        switch (actFn) {
            case Activation::ReLU:      return x > 0.0f ? x : 0.0f;
            case Activation::LeakyReLU: return x > 0.0f ? x : 0.01f * x;
            case Activation::Sigmoid:   return 1.0f / (1.0f + expf(-x));
        }
        return x > 0.0f ? x : 0.0f;
    }

    static int nextPow2(int n) { int p=1; while(p<n) p<<=1; return p; }

    string actName()  const {
        if (actFn == Activation::ReLU)      return "ReLU";
        if (actFn == Activation::LeakyReLU) return "LeakyReLU";
        return "Sigmoid";
    }
    string initName(Init i) const { return (i==Init::He)?"He":"Xavier"; }
    string lossName() const { return (lossFn==Loss::CrossEntropy)?"CrossEntropy":"MSE"; }
    string optName()  const { return (optim==Optimizer::SGD)?"SGD":"Adam"; }
};
