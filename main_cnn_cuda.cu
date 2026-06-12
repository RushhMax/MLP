// Pipeline CNN (CPU) + MLP_CUDA (GPU) end-to-end
//
// Arquitectura:
//   Input [1][28][28]
//   → ConvLayer (1→8, 3×3, SAME) → ReLU → [8][28][28]
//   → PoolingLayer (2×2, MAX)            → [8][14][14]
//   → ConvLayer (8→16, 3×3, SAME) → ReLU → [16][14][14]
//   → PoolingLayer (2×2, MAX)             → [16][7][7]
//   → FlattenLayer                        → [784]
//   → MLP_CUDA [784 → 256 → 128 → 10]    (GPU, SGD)
//
// Backprop completo: gradiente fluye desde MLP_CUDA → Flatten → Pool → Conv.

#include "conv_layer.cpp"
#include "pooling_layer.cpp"
#include "flatten_layer.cpp"

#include "mlp_cuda.h"

#include <fstream>
#include <cstdint>

// ============================================================
// Lector MNIST (formato IDX binario)
// ============================================================
static uint32_t readBE32(ifstream& f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (uint32_t(b[0])<<24)|(uint32_t(b[1])<<16)|(uint32_t(b[2])<<8)|uint32_t(b[3]);
}

static vector<Tensor3D> loadImages(const string& path, int max_n = -1) {
    ifstream f(path, ios::binary);
    if (!f) throw runtime_error("No se pudo abrir: " + path);
    if (readBE32(f) != 0x00000803)
        throw runtime_error("Magic number incorrecto en imágenes");

    int n    = (int)readBE32(f);
    int rows = (int)readBE32(f);
    int cols = (int)readBE32(f);
    if (max_n > 0 && max_n < n) n = max_n;

    vector<Tensor3D> imgs;
    imgs.reserve(n);
    for (int i = 0; i < n; i++) {
        Tensor3D img(1, vector<vector<double>>(rows, vector<double>(cols)));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) {
                uint8_t px; f.read(reinterpret_cast<char*>(&px), 1);
                img[0][r][c] = px / 255.0;
            }
        imgs.push_back(move(img));
    }
    return imgs;
}

static vector<int> loadLabels(const string& path, int max_n = -1) {
    ifstream f(path, ios::binary);
    if (!f) throw runtime_error("No se pudo abrir: " + path);
    if (readBE32(f) != 0x00000801)
        throw runtime_error("Magic number incorrecto en etiquetas");

    int n = (int)readBE32(f);
    if (max_n > 0 && max_n < n) n = max_n;

    vector<int> lbls(n);
    for (int i = 0; i < n; i++) {
        uint8_t lbl; f.read(reinterpret_cast<char*>(&lbl), 1);
        lbls[i] = (int)lbl;
    }
    return lbls;
}

// ============================================================
// MAIN
// ============================================================
int main() {
    const int   TRAIN_N  = 3000;
    const int   TEST_N   = 500;
    const int   EPOCHS   = 15;
    const float LR_MLP   = 0.001f;
    const double LR_CNN  = 0.001;

    cout << "Cargando MNIST...\n";
    auto train_imgs = loadImages("data/mnist/train-images-idx3-ubyte", TRAIN_N);
    auto train_lbls = loadLabels("data/mnist/train-labels-idx1-ubyte", TRAIN_N);
    auto test_imgs  = loadImages("data/mnist/t10k-images-idx3-ubyte",  TEST_N);
    auto test_lbls  = loadLabels("data/mnist/t10k-labels-idx1-ubyte",  TEST_N);
    cout << "Train: " << train_imgs.size() << "  Test: " << test_imgs.size() << "\n";

    // ---- Capas convolucionales (CPU) ----
    ConvLayer    conv1(1,  8, 3, 1, Padding::SAME);   // [1][28][28] → [8][28][28]
    PoolingLayer pool1(2, 2, PoolType::MAX);           // [8][28][28] → [8][14][14]
    ConvLayer    conv2(8, 16, 3, 1, Padding::SAME);   // [8][14][14] → [16][14][14]
    PoolingLayer pool2(2, 2, PoolType::MAX);           // [16][14][14] → [16][7][7]
    FlattenLayer flatten;                              // [16][7][7]  → [784]

    // flat_size = 16 * 7 * 7 = 784
    const int flat_size = 16 * 7 * 7;

    // ---- MLP_CUDA (GPU) ----
    MLP_CUDA mlp({flat_size, 256, 128, 10},
                 Activation::ReLU,
                 Init::He,
                 Loss::CrossEntropy,
                 Optimizer::SGD);

    mt19937 rng(42);

    cout << "\n==============================\n";
    cout << "ENTRENAMIENTO CNN + MLP_CUDA\n";
    cout << "Épocas: " << EPOCHS << "  lr_mlp=" << LR_MLP << "  lr_cnn=" << LR_CNN << "\n";
    cout << "==============================\n";

    for (int ep = 0; ep < EPOCHS; ep++) {
        int N = (int)train_imgs.size();
        vector<int> order(N);
        iota(order.begin(), order.end(), 0);
        shuffle(order.begin(), order.end(), rng);

        double total_loss = 0.0;
        int    correct    = 0;

        for (int i : order) {
            // ---- Forward CNN (CPU, double) ----
            Tensor3D feat = conv1.forward(train_imgs[i]);
            feat = pool1.forward(feat);
            feat = conv2.forward(feat);
            feat = pool2.forward(feat);
            vector<double> flat_d = flatten.forward(feat);

            // ---- Conversión double → float ----
            vector<float> flat_f(flat_d.begin(), flat_d.end());

            // ---- Forward MLP (GPU, float) ----
            vector<float> probs = mlp.stepForward(flat_f);

            // ---- Pérdida cross-entropy ----
            int lbl = train_lbls[i];
            total_loss += -log(max(probs[lbl], 1e-7f));
            if (argmaxf(probs) == lbl) correct++;

            // ---- Backward MLP (GPU) → gradiente respecto al input ----
            vector<float> target_f(10, 0.0f);
            target_f[lbl] = 1.0f;
            vector<float> grad_f = mlp.stepBackward(target_f, LR_MLP);

            // ---- Conversión float → double ----
            vector<double> grad_d(grad_f.begin(), grad_f.end());

            // ---- Backward CNN (CPU) — orden inverso ----
            Tensor3D grad_feat = flatten.backward(grad_d);
            grad_feat = pool2.backward(grad_feat);
            grad_feat = conv2.backward(grad_feat, LR_CNN);
            grad_feat = pool1.backward(grad_feat);
            conv1.backward(grad_feat, LR_CNN);
        }

        // ---- Evaluación en test ----
        mlp.syncWeights();
        int test_correct = 0;
        for (int i = 0; i < (int)test_imgs.size(); i++) {
            Tensor3D feat = conv1.forward(test_imgs[i]);
            feat = pool1.forward(feat);
            feat = conv2.forward(feat);
            feat = pool2.forward(feat);
            vector<double> flat_d = flatten.forward(feat);
            vector<float>  flat_f(flat_d.begin(), flat_d.end());
            vector<float>  probs = mlp.stepForward(flat_f);
            if (argmaxf(probs) == test_lbls[i]) test_correct++;
        }

        cout << "Época " << setw(3) << (ep+1)
             << " | Loss: "  << fixed << setprecision(4) << total_loss / N
             << " | Train: " << fixed << setprecision(1) << 100.0*correct/N         << "%"
             << " | Test: "  << fixed << setprecision(1) << 100.0*test_correct/(int)test_imgs.size() << "%\n";
    }

    return 0;
}
