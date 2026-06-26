// Pipeline CNN (CPU) + MLP_CUDA (GPU) end-to-end
//
// Uso:  ./cnn_cuda [mnist|hasy]   (por defecto: mnist)
//
// Arquitectura (MNIST 28×28):
//   Input [1][28][28]
//   → ConvLayer (1→8, 3×3, SAME) → ReLU → [8][28][28]
//   → PoolingLayer (2×2, MAX)            → [8][14][14]
//   → ConvLayer (8→16, 3×3, SAME) → ReLU → [16][14][14]
//   → PoolingLayer (2×2, MAX)             → [16][7][7]
//   → FlattenLayer                        → [784]
//   → MLP_CUDA [784 → 256 → 128 → 10]    (GPU, SGD)
//
// Arquitectura (HASYv2 32×32):
//   Input [1][32][32]
//   → ConvLayer (1→8, 3×3, SAME) → ReLU → [8][32][32]
//   → PoolingLayer (2×2, MAX)            → [8][16][16]
//   → ConvLayer (8→16, 3×3, SAME) → ReLU → [16][16][16]
//   → PoolingLayer (2×2, MAX)             → [16][8][8]
//   → FlattenLayer                        → [1024]
//   → MLP_CUDA [1024 → 512 → 256 → 369]  (GPU, SGD)
//
// Backprop completo: gradiente fluye desde MLP_CUDA → Flatten → Pool → Conv.

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "dataset.cpp"

#include "conv_layer.cpp"
#include "pooling_layer.cpp"
#include "flatten_layer.cpp"

#include "mlp_cuda.h"

#include <sstream>

// ============================================================
// Convierte Vector plano [H*W] → Tensor3D [1][H][W]
// ============================================================
static Tensor3D vecToTensor(const Vector& v, int H, int W) {
    Tensor3D t(1, vector<vector<double>>(H, vector<double>(W)));
    for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            t[0][h][w] = v(h * W + w);
    return t;
}

// ============================================================
// Canvas HTML con píxeles en escala de grises [0,1]
// ============================================================
static string imgCanvas(const Vector& x, int H, int W, int id) {
    ostringstream s;
    s << "<canvas id='c" << id << "' width='" << W << "' height='" << H
      << "' style='width:112px;height:112px;image-rendering:pixelated'></canvas>\n";
    s << "<script>(function(){"
      << "var cv=document.getElementById('c" << id << "');"
      << "var ctx=cv.getContext('2d');"
      << "var im=ctx.createImageData(" << W << "," << H << ");"
      << "var d=im.data;var px=[";
    for (int i = 0; i < H * W; i++) {
        int v = max(0, min(255, (int)(x(i) * 255.0 + 0.5)));
        s << v << "," << v << "," << v;
        if (i < H * W - 1) s << ",";
    }
    s << "];for(var i=0;i<" << H * W << ";i++){"
      << "d[i*4]=px[i*3];d[i*4+1]=px[i*3+1];d[i*4+2]=px[i*3+2];d[i*4+3]=255;}"
      << "ctx.putImageData(im,0,0);})();</script>\n";
    return s.str();
}

// ============================================================
// Reporte en consola: resumen, confusión, historial
// ============================================================
static void reporteCompleto(
    const DatasetInfo&    ds,
    const vector<double>& lossHist,
    const vector<double>& precHist,
    const vector<int>&    testPred)
{
    int NC    = ds.nClases;
    int testN = (int)ds.test.X.size();

    double pTrain = precHist.empty() ? 0.0 : precHist.back();
    int testOk = 0;
    for (int i = 0; i < testN; i++)
        if (testPred[i] == ds.test.y[i]) testOk++;
    double pTest = 100.0 * testOk / testN;

    cout << "\n==============================\n";
    cout << "RESULTADOS FINALES — " << ds.nombre << "\n";
    cout << "  Precisión entrenamiento: " << fixed << setprecision(2) << pTrain << "%\n";
    cout << "  Precisión prueba:        " << fixed << setprecision(2) << pTest  << "%\n";

    // Confusion matrix
    vector<vector<int>> m(NC, vector<int>(NC, 0));
    for (int i = 0; i < testN; i++)
        m[ds.test.y[i]][testPred[i]]++;

    auto corto = [](const string& s) { return s.size() > 8 ? s.substr(0, 8) : s; };

    if (NC <= 30) {
        cout << "\nMATRIZ DE CONFUSIÓN (test)\n";
        cout << setw(10) << "";
        for (auto& e : ds.etiquetas) cout << setw(9) << corto(e);
        cout << "\n";
        for (int i = 0; i < NC; i++) {
            cout << setw(10) << corto(ds.etiquetas[i]);
            for (int j = 0; j < NC; j++) cout << setw(9) << m[i][j];
            cout << "\n";
        }
    }

    cout << "\nPRECISIÓN POR CLASE\n";
    for (int i = 0; i < NC; i++) {
        int total = 0;
        for (int j = 0; j < NC; j++) total += m[i][j];
        double pct = total > 0 ? 100.0 * m[i][i] / total : 0.0;
        cout << "  " << setw(30) << left << ds.etiquetas[i]
             << " " << fixed << setprecision(1) << pct << "%"
             << " (" << m[i][i] << "/" << total << ")\n";
    }

    cout << "\nHISTORIAL\n";
    cout << setw(6)  << "Época"
         << setw(12) << "Loss"
         << setw(12) << "Precisión\n";
    for (int i = 0; i < (int)lossHist.size(); i++)
        cout << setw(6)  << (i + 1)
             << setw(12) << fixed << setprecision(6) << lossHist[i]
             << setw(11) << fixed << setprecision(2) << precHist[i] << "%\n";
}

// ============================================================
// Reporte HTML
// ============================================================
static void generarHTML(
    const string&         path,
    const DatasetInfo&    ds,
    const vector<double>& lossHist,
    const vector<double>& precHist,
    const vector<int>&    testPred,
    int imgH, int imgW)
{
    ofstream f(path);
    if (!f) { cerr << "No se pudo crear: " << path << "\n"; return; }

    int NC    = ds.nClases;
    int testN = (int)ds.test.X.size();

    // Confusion matrix (pre-computada una sola vez)
    vector<vector<int>> m(NC, vector<int>(NC, 0));
    for (int i = 0; i < testN; i++)
        m[ds.test.y[i]][testPred[i]]++;

    // ── Head ──────────────────────────────────────────────
    f << "<!DOCTYPE html>\n<html lang=\"es\">\n<head>\n"
      << "<meta charset=\"UTF-8\">\n"
      << "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
      << "<title>CNN+MLP \xe2\x80\x94 " << ds.nombre << "</title>\n"
      << "<script src=\"https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js\"></script>\n"
      << R"(<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,sans-serif;background:#f0f2f5;color:#222;padding:24px}
  h1{text-align:center;margin-bottom:24px;font-size:1.8rem}
  h2{margin:24px 0 12px;font-size:1.2rem;color:#333}
  h3{font-size:1rem;margin-bottom:8px;color:#555}
  section{background:#fff;border-radius:12px;padding:20px;margin-bottom:24px;
          box-shadow:0 2px 8px #0001}
  .charts{display:flex;gap:24px;flex-wrap:wrap}
  .chart-box{flex:1;min-width:280px}
  .grid{display:flex;flex-wrap:wrap;gap:12px}
  .card{border:2px solid #ccc;border-radius:8px;padding:8px;text-align:center;
        background:#fafafa;transition:transform .15s}
  .card:hover{transform:scale(1.04)}
  .label{font-size:.75rem;margin-top:4px;line-height:1.4}
  table.cm{border-collapse:collapse;font-size:.8rem}
  table.cm th,table.cm td{border:1px solid #ddd;padding:5px 8px;text-align:center;
                           white-space:nowrap}
  table.cm th{background:#f0f2f5;font-weight:600}
  canvas{display:block}
</style>
</head>
<body>
)";
    f << "<h1>\xf0\x9f\xa7\xa0 CNN+MLP \xe2\x80\x94 " << ds.nombre << "</h1>\n";

    // ── Curvas de aprendizaje ──────────────────────────────
    f << "<section><h2>\xf0\x9f\x93\x88 Curvas de Aprendizaje</h2>\n"
         "<div class='charts'>\n"
         "<div class='chart-box'><h3>P\xc3\xa9rdida</h3><canvas id='chartLoss'></canvas></div>\n"
         "<div class='chart-box'><h3>Precisi\xc3\xb3n (%)</h3><canvas id='chartPrec'></canvas></div>\n"
         "</div>\n<script>\nvar epocas=[";
    for (int i = 0; i < (int)lossHist.size(); i++)
        f << (i + 1) << (i + 1 < (int)lossHist.size() ? "," : "");
    f << "];\nvar loss=[";
    for (int i = 0; i < (int)lossHist.size(); i++)
        f << fixed << setprecision(6) << lossHist[i]
          << (i + 1 < (int)lossHist.size() ? "," : "");
    f << "];\nvar prec=[";
    for (int i = 0; i < (int)precHist.size(); i++)
        f << fixed << setprecision(2) << precHist[i]
          << (i + 1 < (int)precHist.size() ? "," : "");
    f << R"(];
function makeChart(id,labels,data,label,color){new Chart(document.getElementById(id),{type:'line',data:{labels:labels,datasets:[{label:label,data:data,borderColor:color,backgroundColor:color+'22',fill:true,tension:0.3,pointRadius:4}]},options:{responsive:true,plugins:{legend:{display:false}},scales:{x:{title:{display:true,text:'Epoca'}},y:{title:{display:true,text:label}}}}})}
makeChart('chartLoss',epocas,loss,'Loss','#e74c3c');
makeChart('chartPrec',epocas,prec,'Precision %','#2ecc71');
</script></section>
)";

    // ── Predicciones (primeras 10) ─────────────────────────
    f << "<section><h2>\xf0\x9f\x94\x8d Predicciones</h2><div class='grid'>\n";
    int N = min(10, testN);
    for (int i = 0; i < N; i++) {
        bool ok    = (testPred[i] == ds.test.y[i]);
        string brd = ok ? "#2ecc71" : "#e74c3c";
        string ico = ok ? "\xe2\x9c\x85" : "\xe2\x9d\x8c";
        f << "<div class='card' style='border-color:" << brd << "'>\n";
        f << imgCanvas(ds.test.X[i], imgH, imgW, i);
        f << "<div class='label'>" << ico << " Pred: <b>" << ds.etiquetas[testPred[i]]
          << "</b><br>Real: " << ds.etiquetas[ds.test.y[i]] << "</div></div>\n";
    }
    f << "</div></section>\n";

    // ── Errores (primeros 10) ──────────────────────────────
    {
        vector<int> errIdx;
        for (int i = 0; i < testN; i++)
            if (testPred[i] != ds.test.y[i]) errIdx.push_back(i);
        f << "<section><h2>\xe2\x9d\x8c Errores de Clasificaci\xc3\xb3n</h2>\n"
          << "<p>" << errIdx.size() << " errores de " << testN << " muestras</p>\n";
        if (!errIdx.empty()) {
            f << "<div class='grid'>\n";
            int ne = min(10, (int)errIdx.size());
            for (int k = 0; k < ne; k++) {
                int i = errIdx[k];
                f << "<div class='card' style='border-color:#e74c3c'>\n";
                f << imgCanvas(ds.test.X[i], imgH, imgW, 1000 + k);
                f << "<div class='label'>ID: " << i
                  << "<br>Pred: <b style='color:#e74c3c'>" << ds.etiquetas[testPred[i]]
                  << "</b><br>Real: <b style='color:#2ecc71'>" << ds.etiquetas[ds.test.y[i]]
                  << "</b></div></div>\n";
            }
            f << "</div>\n";
        }
        f << "</section>\n";
    }

    // ── Matriz de confusión ────────────────────────────────
    {
        auto corto = [](const string& s) { return s.size() > 8 ? s.substr(0, 8) : s; };
        int maxVal = 1;
        for (int i = 0; i < NC; i++)
            for (int j = 0; j < NC; j++)
                maxVal = max(maxVal, m[i][j]);

        f << "<section><h2>\xf0\x9f\x9f\xa6 Matriz de Confusi\xc3\xb3n</h2>"
             "<div style='overflow-x:auto'>\n"
             "<table class='cm'><tr><th>Real \\ Pred</th>";
        for (int j = 0; j < NC; j++) f << "<th>" << corto(ds.etiquetas[j]) << "</th>";
        f << "</tr>\n";

        for (int i = 0; i < NC; i++) {
            f << "<tr><th>" << corto(ds.etiquetas[i]) << "</th>";
            for (int j = 0; j < NC; j++) {
                double intensity = (double)m[i][j] / maxVal;
                string bg = (i == j)
                    ? "background:rgba(46,204,113,"  + to_string(0.2 + intensity * 0.7) + ")"
                    : (m[i][j] > 0
                        ? "background:rgba(231,76,60," + to_string(intensity * 0.7) + ")"
                        : "background:#f8f9fa");
                f << "<td style='" << bg << "'>";
                if (m[i][j] > 0) f << m[i][j];
                f << "</td>";
            }
            f << "</tr>\n";
        }
        f << "</table></div></section>\n";
    }

    f << "</body></html>\n";
    cout << "\nReporte HTML guardado en: " << path << "\n";
    cout << "\xc3\x81brelo con: xdg-open " << path << "\n";
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char* argv[]) {
    string dsname = (argc >= 2) ? argv[1] : "mnist";

    // ---- Hiperparámetros por dataset ----
    int    EPOCHS  = 50;
    float  LR_MLP  = 0.001f;
    double LR_CNN  = 0.001;

    // ---- Cargar dataset ----
    cout << "Cargando dataset: " << dsname << " ...\n";
    unique_ptr<DatasetLoader> loader;
    int imgH, imgW;

    if (dsname == "mnist") {
        loader = make_unique<MNISTLoader>(
            "data/mnist/train-images-idx3-ubyte",
            "data/mnist/train-labels-idx1-ubyte",
            "data/mnist/t10k-images-idx3-ubyte",
            "data/mnist/t10k-labels-idx1-ubyte",
            3000, 500);
        imgH = imgW = 28;
    } else if (dsname == "hasy") {
        loader = make_unique<HASYLoader>("data/hasy", 1, 5000, 1000);
        imgH = imgW = 32;
    } else {
        cerr << "Dataset desconocido: " << dsname
             << "\nDatasets disponibles: mnist, hasy\n";
        return 1;
    }

    DatasetInfo ds = loader->cargar();
    cout << "\nDataset listo:\n"
         << "  Nombre:    " << ds.nombre         << "\n"
         << "  Input dim: " << ds.inputDim        << "\n"
         << "  Clases:    " << ds.nClases         << "\n"
         << "  Train:     " << ds.train.X.size()  << " muestras\n"
         << "  Test:      " << ds.test.X.size()   << " muestras\n";

    // ---- Capas convolucionales (CPU) ----
    // SAME padding preserva H/W, dos pools de 2×2 reducen a H/4 × W/4.
    ConvLayer    conv1(1,  32, 3, 1, Padding::SAME);
    PoolingLayer pool1(2, 2, PoolType::MAX);
    ConvLayer    conv2(32, 64, 3, 1, Padding::SAME);
    PoolingLayer pool2(2, 2, PoolType::MAX);
    FlattenLayer flatten;

    // flat_size = 64 × (H/4) × (W/4)
    const int flat_size = 64 * (imgH / 4) * (imgW / 4);

    // ---- Arquitectura MLP según número de clases ----
    vector<int> mlp_arch;
    if (ds.nClases > 100)
        mlp_arch = {flat_size, 512, 256, ds.nClases};
    else
        mlp_arch = {flat_size, 256, 128, ds.nClases};

    cout << "\nArquitectura CNN: [1][" << imgH << "][" << imgW << "]"
         << " \xe2\x86\x92 Conv(1\xe2\x86\x9232) \xe2\x86\x92 Pool \xe2\x86\x92 Conv(32\xe2\x86\x9264) \xe2\x86\x92 Pool"
         << " \xe2\x86\x92 Flatten[" << flat_size << "]\n"
         << "Arquitectura MLP:";
    for (int x : mlp_arch) cout << " " << x;
    cout << "\n";

    // ---- MLP_CUDA (GPU) ----
    MLP_CUDA mlp(mlp_arch,
                 Activation::ReLU,
                 Init::He,
                 Loss::CrossEntropy,
                 Optimizer::Adam);

    mt19937 local_rng(42);

    cout << "\n==============================\n";
    cout << "ENTRENAMIENTO CNN + MLP_CUDA \xe2\x80\x94 " << ds.nombre << "\n";
    cout << "Épocas: " << EPOCHS
         << "  lr_mlp=" << LR_MLP
         << "  lr_cnn=" << LR_CNN << "\n";
    cout << "==============================\n";

    vector<double> historialLoss, historialPrecision;

    for (int ep = 0; ep < EPOCHS; ep++) {
        int N = (int)ds.train.X.size();
        vector<int> order(N);
        iota(order.begin(), order.end(), 0);
        shuffle(order.begin(), order.end(), local_rng);

        double total_loss = 0.0;
        int    correct    = 0;

        for (int i : order) {
            // ---- Convertir muestra plana → Tensor3D ----
            Tensor3D img = vecToTensor(ds.train.X[i], imgH, imgW);

            // ---- Forward CNN (CPU, double) ----
            Tensor3D feat = conv1.forward(img);
            feat = pool1.forward(feat);
            feat = conv2.forward(feat);
            feat = pool2.forward(feat);
            vector<double> flat_d = flatten.forward(feat);

            // ---- Conversión double → float ----
            vector<float> flat_f(flat_d.begin(), flat_d.end());

            // ---- Forward MLP (GPU, float) ----
            vector<float> probs = mlp.stepForward(flat_f);

            // ---- Pérdida cross-entropy ----
            int lbl = ds.train.y[i];
            total_loss += -log(max(probs[lbl], 1e-7f));
            if (argmaxf(probs) == lbl) correct++;

            // ---- Backward MLP (GPU) → gradiente respecto al input ----
            vector<float> target_f(ds.nClases, 0.0f);
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
        int test_N = (int)ds.test.X.size();
        for (int i = 0; i < test_N; i++) {
            Tensor3D img  = vecToTensor(ds.test.X[i], imgH, imgW);
            Tensor3D feat = conv1.forward(img);
            feat = pool1.forward(feat);
            feat = conv2.forward(feat);
            feat = pool2.forward(feat);
            vector<double> flat_d = flatten.forward(feat);
            vector<float>  flat_f(flat_d.begin(), flat_d.end());
            vector<float>  probs = mlp.stepForward(flat_f);
            if (argmaxf(probs) == ds.test.y[i]) test_correct++;
        }

        historialLoss.push_back(total_loss / N);
        historialPrecision.push_back(100.0 * correct / N);

        cout << "\xc3\x89poca " << setw(3) << (ep + 1)
             << " | Loss: "  << fixed << setprecision(4) << total_loss / N
             << " | Train: " << fixed << setprecision(1) << 100.0 * correct / N         << "%"
             << " | Test: "  << fixed << setprecision(1) << 100.0 * test_correct / test_N << "%\n";
    }

    // ---- Predicciones finales en test (una sola pasada) ----
    mlp.syncWeights();
    int testN = (int)ds.test.X.size();
    vector<int> testPred(testN);
    cout << "\nComputando predicciones finales...\n";
    for (int i = 0; i < testN; i++) {
        Tensor3D img  = vecToTensor(ds.test.X[i], imgH, imgW);
        Tensor3D feat = conv1.forward(img);
        feat = pool1.forward(feat);
        feat = conv2.forward(feat);
        feat = pool2.forward(feat);
        vector<double> flat_d = flatten.forward(feat);
        vector<float>  flat_f(flat_d.begin(), flat_d.end());
        testPred[i] = argmaxf(mlp.stepForward(flat_f));
    }

    // ---- Reporte consola ----
    reporteCompleto(ds, historialLoss, historialPrecision, testPred);

    // ---- Reporte HTML ----
    string htmlFile = "reporte_" + dsname + ".html";
    generarHTML(htmlFile, ds, historialLoss, historialPrecision, testPred, imgH, imgW);
    system(("xdg-open " + htmlFile + " 2>/dev/null &").c_str());

    return 0;
}
