#define MLPCUDA_WITH_DATASET
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "dataset.cpp"
#include "mlp_cuda.h"

using MLP = MLP_CUDA;

#include "visualizar.cpp"

// ============================================================
// Reporte
// ============================================================
void reporteCompleto(const MLP& red, const DatasetInfo& ds) {
    double pTrain = red.calcularPrecision(ds.train);
    double pTest  = red.calcularPrecision(ds.test);

    cout << "\n==============================\n";
    cout << "RESULTADOS FINALES — " << ds.nombre << "\n";
    cout << "  Precisión entrenamiento: " << fixed << setprecision(2) << pTrain << "%\n";
    cout << "  Precisión prueba:        " << fixed << setprecision(2) << pTest  << "%\n";

    int NC = ds.nClases;
    vector<vector<int>> m(NC, vector<int>(NC, 0));
    for (int i = 0; i < (int)ds.test.X.size(); i++)
        m[ds.test.y[i]][argmax(red.predecir(ds.test.X[i]))]++;

    auto corto = [](const string& s){ return s.size()>8 ? s.substr(0,8) : s; };

    // Calcular totales por clase (test)
    vector<int> totales(NC, 0);
    for (int i = 0; i < NC; i++)
        for (int j = 0; j < NC; j++) totales[i] += m[i][j];

    // Matriz de confusión: solo si hay ≤30 clases
    if (NC <= 30) {
        cout << "\nMATRIZ DE CONFUSIÓN (test)\n" << setw(10) << "";
        for (auto& e : ds.etiquetas) cout << setw(9) << corto(e);
        cout << "\n";
        for (int i = 0; i < NC; i++) {
            cout << setw(10) << corto(ds.etiquetas[i]);
            for (int j = 0; j < NC; j++) cout << setw(9) << m[i][j];
            cout << "\n";
        }
    } else {
        cout << "\n(Matriz de confusión omitida: " << NC << " clases)\n";
    }

    // Precisión por clase — solo clases con muestras en test
    cout << "\nPRECISIÓN POR CLASE\n";
    int clasesConDatos = 0;
    for (int i = 0; i < NC; i++) {
        if (totales[i] == 0) continue;
        clasesConDatos++;
        cout << "  " << setw(30) << left << ds.etiquetas[i]
             << " " << fixed << setprecision(1)
             << 100.0*m[i][i]/totales[i] << "% (" << m[i][i] << "/" << totales[i] << ")\n";
    }
    if (clasesConDatos < NC)
        cout << "  (" << (NC - clasesConDatos) << " clases sin muestras en test omitidas)\n";

    cout << "\nHISTORIAL\n" << setw(6) << "Época" << setw(12) << "Loss" << setw(12) << "Precisión\n";
    for (int i = 0; i < (int)red.historialLoss.size(); i++)
        cout << setw(6) << (i+1)
             << setw(12) << fixed << setprecision(6) << red.historialLoss[i]
             << setw(11) << fixed << setprecision(2) << red.historialPrecision[i] << "%\n";
}

// ============================================================
// Datasets
// ============================================================
unique_ptr<DatasetLoader> crearLoader(const string& nombre) {
    if (nombre == "mnist")
        return make_unique<MNISTLoader>(
            "data/mnist/train-images-idx3-ubyte","data/mnist/train-labels-idx1-ubyte",
            "data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte",
            3000, 1000);
    if (nombre == "simpsons")
        return make_unique<ImageFolderLoader>(
            "data/train","data/test",
            vector<string>{"bart_simpson","charles_montgomery_burns","homer_simpson",
                           "krusty_the_clown","lisa_simpson","marge_simpson",
                           "milhouse_van_houten","moe_szyslak","ned_flanders","principal_skinner"},
            28, 0, true);
    if (nombre == "hasy")
        return make_unique<HASYLoader>(
            "data/hasy", 1, 20000, 5000);
    throw invalid_argument("Dataset desconocido: " + nombre +
        "\nDatasets disponibles: mnist, simpsons, hasy");
}

vector<int> arquitectura(const DatasetInfo& ds) {
    if (ds.nClases > 100)
        return {ds.inputDim, 256, 128, ds.nClases};
    else if (ds.inputDim > 1000)
        return {ds.inputDim, 128, 64, ds.nClases};
    else
        return {ds.inputDim,  64, 32, ds.nClases};
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char* argv[]) {
    string dataset = (argc >= 2) ? argv[1] : "mnist";

    cout << "Cargando dataset: " << dataset << " ...\n";
    DatasetInfo ds = crearLoader(dataset)->cargar();

    cout << "\nDataset listo:\n"
         << "  Nombre:    " << ds.nombre        << "\n"
         << "  Input dim: " << ds.inputDim       << "\n"
         << "  Clases:    " << ds.nClases        << "\n"
         << "  Train:     " << ds.train.X.size() << " muestras\n"
         << "  Test:      " << ds.test.X.size()  << " muestras\n";

    MLP_CUDA red(arquitectura(ds),
                 Activation::ReLU,
                 Init::He,
                 Loss::CrossEntropy,
                 Optimizer::SGD);

    red.entrenar(ds, 70, 0.001f);
    reporteCompleto(red, ds);

    string htmlFile = "reporte_" + dataset + ".html";
    Visualizador viz(red, ds);
    viz.guardar(htmlFile);
    system(("xdg-open " + htmlFile + " 2>/dev/null &").c_str());

    return 0;
}
