/*
 * MLP 
 *
 * Datasets incluidos:
 *   - MNIST (IDX)
 *   - Simpsons MNIST (PNG en carpeta data/)
 *
 * Dependencias:
 *   - sudo apt install libeigen3-dev
 *   - wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
 *
 * Compilación:
 *   g++ -O2 -std=c++17 -I/usr/include/eigen3 mlp.cpp -o mlp
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
#include <filesystem>
#include <memory>
#include <Eigen/Dense>
#include "dataset.cpp"

using namespace std;

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// ----------------------------------------------------------
// Utilidades comunes
// ----------------------------------------------------------

Vector oneHot(int label, int nClases) {
    Vector v = Vector::Zero(nClases);
    v(label) = 1.0;
    return v;
}

int argmax(const Vector& v) {
    int idx = 0;
    for (int i = 1; i < (int)v.size(); i++)
        if (v(i) > v(idx)) idx = i;
    return idx;
}
// ==========================================================
// ACTIVACIONES
// ==========================================================

//sigmoidea - logistica (salida binaria)

// relu max(x,0)
Vector relu(const Vector& x) { return x.cwiseMax(0.0); }
Vector relu_d(const Vector& x) { return (x.array() > 0.0).cast<double>(); }

// leaky relu max(0.01x, x)
Vector leaky_relu(const Vector& x, double alpha = 0.01) {
    return x.cwiseMax(alpha * x);
}
Vector leaky_relu_d(const Vector& x, double alpha = 0.01) {
    return (x.array() > 0.0).cast<double>() * (1.0 - alpha) + alpha;
}

// sigmoid 1/(1+e^-x)  — útil para salidas binarias
Vector sigmoid(const Vector& x) {
    return 1.0 / (1.0 + (-x.array()).exp());
}
Vector sigmoid_d(const Vector& x) {
    Vector s = sigmoid(x);
    return s.array() * (1.0 - s.array());
}

// para clasificación multiclase (salida)
Vector softmax(const Vector& x) {
    Vector xs = x.array() - x.maxCoeff();
    Vector e  = xs.array().exp();
    return e / e.sum();
}

// ==========================================================
// PÉRDIDAS
// ==========================================================

// Cross-Entropy (multiclase)
double ceLoss(const Vector& yt, const Vector& yp) {
    const double eps = 1e-15;
    Vector yc = yp.array().cwiseMax(eps).cwiseMin(1.0-eps);
    return -(yt.array() * yc.array().log()).sum();
}
Vector ceDeriv(const Vector& yt, const Vector& yp) {
    const double eps = 1e-15;
    Vector yc = yp.array().cwiseMax(eps).cwiseMin(1.0-eps);
    return yc - yt;
}

// MSE — útil para regresión o como alternativa a CE
double mseLoss(const Vector& yt, const Vector& yp) {
    return 0.5 * (yp - yt).squaredNorm() / yt.size();
}
Vector mseDeriv(const Vector& yt, const Vector& yp) {
    return (yp - yt) / yt.size();
}

// ==========================================================
// CONFIGURACIÓN DE LA RED
// ==========================================================

enum class Activation { ReLU, LeakyReLU, Sigmoid };
enum class Init       { He, Xavier };
enum class Loss       { CrossEntropy, MSE };
enum class Optimizer  { SGD, Adam };

// ==========================================================
// RED NEURONAL
// ==========================================================
// La función (agrégala fuera de la clase):

class MLP {
    public:
        vector<int>    capas;
        vector<Matrix> pesos;
        vector<double> historialLoss;
        vector<double> historialPrecision;

        MLP(const vector<int>& capas_,
            Activation act  = Activation::LeakyReLU,
            Init       init = Init::He,
            Loss       loss = Loss::CrossEntropy,
            Optimizer  opt  = Optimizer::SGD)
            : capas(capas_), actFn(act), lossFn(loss), optim(opt)
        {
            cout << "\n==============================\n";
            cout << "CREANDO MLP\n";
            cout << "Capas:      [";
            for (int i = 0; i < (int)capas.size(); i++)
                cout << capas[i] << (i+1<(int)capas.size() ? ", " : "");
            cout << "]\n";
            cout << "Activación: " << actName()  << "\n";
            cout << "Init:       " << initName(init) << "\n";
            cout << "Pérdida:    " << lossName()  << "\n";
            cout << "Optimizador:" << optName()   << "\n";

            std::normal_distribution<double> dist(0.0, 1.0);
            for (int i = 0; i < (int)capas.size()-1; i++) {
                double esc = (init == Init::He)
                    ? std::sqrt(2.0 / capas[i])          // He   — bueno para ReLU
                    : std::sqrt(1.0 / capas[i]);          // Xavier — bueno para sigmoid/tanh

                Matrix w(capas[i]+1, capas[i+1]);
                for (int r = 0; r < w.rows(); r++)
                    for (int c = 0; c < w.cols(); c++)
                        w(r,c) = dist(rng) * esc;
                pesos.push_back(w);

                // momentos para Adam (cero si se usa SGD, no se usan)
                m_w.push_back(Matrix::Zero(w.rows(), w.cols()));
                v_w.push_back(Matrix::Zero(w.rows(), w.cols()));
            }
        }

        // --------------------------------------------------
        // Forward
        // --------------------------------------------------

        pair<vector<Vector>, vector<Vector>>
        forward(const Vector& entrada) const {
            vector<Vector> acts, nets;
            Vector a = _bias(entrada);
            acts.push_back(a);

            int L = (int)pesos.size();
            for (int l = 0; l < L; l++) {
                Vector net = pesos[l].transpose() * a;
                nets.push_back(net);
                a = (l < L-1) ? _bias(applyAct(net)) : softmax(net);
                acts.push_back(a);
            }
            return {acts, nets};
        }

        // --------------------------------------------------
        // Backward
        // --------------------------------------------------

        double backward(const vector<Vector>& acts, const vector<Vector>& nets,
                        const Vector& objetivo, double lr)
        {
            int L = (int)pesos.size();
            vector<Vector> deltas(L);

            // Capa de salida (softmax + CE se cancelan; con MSE es aproximado)
            deltas[L-1] = applyLossDeriv(objetivo, acts.back());

            // Capas ocultas
            for (int l = L-2; l >= 0; l--) {
                Matrix wNoBias = pesos[l+1].bottomRows(pesos[l+1].rows()-1);
                deltas[l] = (wNoBias * deltas[l+1]).array() * applyActDeriv(nets[l]).array();
            }

            // Actualizar pesos según optimizador
            if (optim == Optimizer::SGD) {
                for (int l = 0; l < L; l++)
                    pesos[l] -= lr * (acts[l] * deltas[l].transpose());
            } else {
                // Adam
                const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
                adam_t++;
                for (int l = 0; l < L; l++) {
                    Matrix grad = acts[l] * deltas[l].transpose();
                    m_w[l] = beta1 * m_w[l] + (1.0-beta1) * grad;
                    v_w[l] = beta2 * v_w[l] + (1.0-beta2) * grad.array().square().matrix();
                    Matrix m_hat = m_w[l] / (1.0 - std::pow(beta1, adam_t));
                    Matrix v_hat = v_w[l] / (1.0 - std::pow(beta2, adam_t));
                    pesos[l].array() -= lr * m_hat.array() / (v_hat.array().sqrt() + eps);
                }
            }

            return applyLoss(objetivo, acts.back());
        }

        // --------------------------------------------------
        // Predicción
        // --------------------------------------------------

        Vector predecir(const Vector& x) const {
            return forward(x).first.back();
        }

        // --------------------------------------------------
        // Entrenamiento
        // --------------------------------------------------

        void entrenar(const DatasetInfo& ds, int epocas, double lr, int batchLog = 1) {
            cout << "\n==============================\n";
            cout << "ENTRENAMIENTO — " << ds.nombre << "\n";
            cout << "Muestras train: " << ds.train.X.size() << "\n";

            int N = (int)ds.train.X.size();

            for (int ep = 0; ep < epocas; ep++) {
                vector<int> orden(N);
                iota(orden.begin(), orden.end(), 0);
                shuffle(orden.begin(), orden.end(), rng);

                double errorTotal = 0.0;
                for (int i : orden) {
                    auto [acts, nets] = forward(ds.train.X[i]);
                    errorTotal += backward(acts, nets, oneHot(ds.train.y[i], (int)capas.back()), lr);
                }

                double errorProm = errorTotal / N;
                double precision = calcularPrecision(ds.train);

                historialLoss.push_back(errorProm);
                historialPrecision.push_back(precision);

                if ((ep+1) % batchLog == 0) {
                    cout << "Época " << std::setw(3) << (ep+1)
                         << " | Error: "     << std::fixed << std::setprecision(6) << errorProm
                         << " | Precisión: " << std::fixed << std::setprecision(2) << precision << "%\n";
                }
            }
        }

        // --------------------------------------------------
        // Métricas
        // --------------------------------------------------

        double calcularPrecision(const Split& s) const {
            int ok = 0;
            for (int i = 0; i < (int)s.X.size(); i++)
                if (argmax(predecir(s.X[i])) == s.y[i]) ok++;
            return 100.0 * ok / (int)s.X.size();
        }

    private:
        Activation     actFn;
        Loss           lossFn;
        Optimizer      optim;
        vector<Matrix> m_w, v_w; // momentos Adam
        int            adam_t = 0;

        // --- aplicar activación / pérdida seleccionada ---

        Vector applyAct(const Vector& x) const {
            switch (actFn) {
                case Activation::ReLU:      return relu(x);
                case Activation::LeakyReLU: return leaky_relu(x);
                case Activation::Sigmoid:   return sigmoid(x);
            }
            return relu(x);
        }
        Vector applyActDeriv(const Vector& x) const {
            switch (actFn) {
                case Activation::ReLU:      return relu_d(x);
                case Activation::LeakyReLU: return leaky_relu_d(x);
                case Activation::Sigmoid:   return sigmoid_d(x);
            }
            return relu_d(x);
        }
        double applyLoss(const Vector& yt, const Vector& yp) const {
            return (lossFn == Loss::CrossEntropy) ? ceLoss(yt, yp) : mseLoss(yt, yp);
        }
        Vector applyLossDeriv(const Vector& yt, const Vector& yp) const {
            return (lossFn == Loss::CrossEntropy) ? ceDeriv(yt, yp) : mseDeriv(yt, yp);
        }

        // --- nombres para el log ---
        string actName()  const {
            if (actFn  == Activation::ReLU)        return "ReLU";
            if (actFn  == Activation::LeakyReLU)   return "LeakyReLU";
            return "Sigmoid";
        }
        string initName(Init i) const { return (i == Init::He) ? "He" : "Xavier"; }
        string lossName() const { return (lossFn == Loss::CrossEntropy) ? "CrossEntropy" : "MSE"; }
        string optName()  const { return (optim  == Optimizer::SGD)     ? "SGD"          : "Adam"; }

        static Vector _bias(const Vector& x) {
            Vector r(x.size()+1);
            r(0) = 1.0;
            r.tail(x.size()) = x;
            return r;
        }
};

#include "visualizar.cpp"

// ==========================================================
// REPORTES
// ==========================================================

void reporteCompleto( const MLP& red, const DatasetInfo& ds) {
    // Precisión train / test
    double pTrain = red.calcularPrecision(ds.train);
    double pTest  = red.calcularPrecision(ds.test);

    cout << "\n==============================\n";
    cout << "RESULTADOS FINALES — " << ds.nombre << "\n";
    cout << "  Precisión entrenamiento: "
              << std::fixed << std::setprecision(2) << pTrain << "%\n";
    cout << "  Precisión prueba:        "
              << std::fixed << std::setprecision(2) << pTest  << "%\n";

    // Matriz de confusión
    int NC = ds.nClases;
    vector<vector<int>> m(NC, vector<int>(NC, 0));
    for (int i = 0; i < (int)ds.test.X.size(); i++) {
        int pred = argmax(red.predecir(ds.test.X[i]));
        m[ds.test.y[i]][pred]++;
    }

    // Nombres cortos (máx 8 chars)
    auto corto = [](const std::string& s) {
        return s.size() > 8 ? s.substr(0, 8) : s;
    };

    cout << "\nMATRIZ DE CONFUSIÓN (test)\n";
    cout << std::setw(10) << "";
    for (auto& e : ds.etiquetas)
        cout << std::setw(9) << corto(e);
    cout << "\n";
    for (int i = 0; i < NC; i++) {
        cout << std::setw(10) << corto(ds.etiquetas[i]);
        for (int j = 0; j < NC; j++)
            cout << std::setw(9) << m[i][j];
        cout << "\n";
    }

    // Precisión por clase
    cout << "\nPRECISIÓN POR CLASE\n";
    for (int i = 0; i < NC; i++) {
        int total = 0;
        for (int j = 0; j < NC; j++) total += m[i][j];
        double pct = total > 0 ? 100.0 * m[i][i] / total : 0.0;
        cout << "  " << std::setw(30) << std::left << ds.etiquetas[i]
                  << " " << std::fixed << std::setprecision(1) << pct << "%"
                  << " (" << m[i][i] << "/" << total << ")\n";
    }

    // Primeros errores
    cout << "\nPRIMEROS ERRORES\n";
    int shown = 0;
    for (int i = 0; i < (int)ds.test.X.size() && shown < 10; i++) {
        int pred = argmax(red.predecir(ds.test.X[i]));
        if (pred != ds.test.y[i]) {
            cout << "  " << std::setw(20) << ds.test.nombres[i]
                      << " | Real: " << std::setw(25) << ds.etiquetas[ds.test.y[i]]
                      << " | Pred: " << ds.etiquetas[pred] << "\n";
            shown++;
        }
    }
    if (shown == 0) cout << "  ¡Sin errores!\n";

    // Historial
    cout << "\nHISTORIAL DE ENTRENAMIENTO\n";
    cout << std::setw(6) << "Época"
              << std::setw(12) << "Loss"
              << std::setw(12) << "Precisión\n";
    for (int i = 0; i < (int)red.historialLoss.size(); i++) {
        cout << std::setw(6) << (i+1)
                  << std::setw(12) << std::fixed << std::setprecision(6) << red.historialLoss[i]
                  << std::setw(11) << std::fixed << std::setprecision(2) << red.historialPrecision[i] << "%\n";
    }
}

// ==========================================================
// REGISTRO DE CONFIGURACIONES DE DATASETS
// ==========================================================

std::unique_ptr<DatasetLoader> crearLoader(const std::string& nombre) {

    if (nombre == "mnist") {
        return std::make_unique<MNISTLoader>(
            "data/mnist/train-images-idx3-ubyte",
            "data/mnist/train-labels-idx1-ubyte",
            "data/mnist/t10k-images-idx3-ubyte",
            "data/mnist/t10k-labels-idx1-ubyte",
            /*maxTrain=*/ 3000,
            /*maxTest=*/  1000
        );
    }

    if (nombre == "simpsons") {
        return std::make_unique<ImageFolderLoader>(
            "data/train",
            "data/test",
            vector<std::string>{
                "bart_simpson",
                "charles_montgomery_burns",
                "homer_simpson",
                "krusty_the_clown",
                "lisa_simpson",
                "marge_simpson",
                "milhouse_van_houten",
                "moe_szyslak",
                "ned_flanders",
                "principal_skinner"
            },
            /*imgSize=*/     28,
            /*maxPorClase=*/ 0,
            /*color=*/       true
        );
    }

    throw std::invalid_argument("Dataset desconocido: " + nombre +
        "\nDatasets disponibles: mnist, simpsons");
}

// Arquitectura recomendada por dataset
vector<int> arquitectura(const DatasetInfo& ds) {
    if (ds.inputDim > 1000)
        return {ds.inputDim, 128, 64, ds.nClases};
    else
        return {ds.inputDim, 64, 32, ds.nClases};
}


// ==========================================================
// MAIN
// ==========================================================

int main(int argc, char* argv[]) {
    string dataset = "mnist"; 
    if (argc >= 2) {
        dataset = argv[1];
    }

    // 1. Cargar dataset
    cout << "Cargando dataset: " << dataset << " ...\n";

    DatasetInfo ds;
    auto loader = crearLoader(dataset);
    ds = loader->cargar();
    
    cout << "\nDataset listo:\n";
    cout << "  Nombre:    " << ds.nombre      << "\n";
    cout << "  Input dim: " << ds.inputDim     << "\n";
    cout << "  Clases:    " << ds.nClases      << "\n";
    cout << "  Train:     " << ds.train.X.size() << " muestras\n";
    cout << "  Test:      " << ds.test.X.size()  << " muestras\n";

    // 2. Crear red
    //   Parámetros opcionales (por defecto = comportamiento original):
    //     Activation::ReLU / LeakyReLU / Sigmoid
    //     Init::He / Xavier
    //     Loss::CrossEntropy / MSE
    //     Optimizer::SGD / Adam
    MLP red(arquitectura(ds),
            Activation::ReLU,
            Init::He,
            Loss::CrossEntropy,
            Optimizer::SGD);

    // 3. Entrenar
    red.entrenar(ds, 70, 0.001);

    // 4. Reporte en consola
    reporteCompleto(red, ds);

    // 5. Reporte visual HTML (equivalente a matplotlib)
    std::string htmlFile = "reporte_" + dataset + ".html";
    Visualizador viz(red, ds);
    viz.guardar(htmlFile);

    // Intentar abrir el navegador automáticamente (Linux)
    std::system(("xdg-open " + htmlFile + " 2>/dev/null &").c_str());

    return 0;
}
