class Visualizador {
public:
    const MLP&         red;
    const DatasetInfo& ds;
    int                imgSize; // píxeles por lado de cada imagen

    Visualizador(const MLP& red_, const DatasetInfo& ds_, int imgSize_ = 28)
        : red(red_), ds(ds_), imgSize(imgSize_) {}

    // Punto de entrada: genera el HTML completo
    void guardar(const std::string& path = "reporte.html") const {
        std::ofstream f(path);
        if (!f) throw std::runtime_error("No se pudo crear: " + path);

        f << htmlHead(ds.nombre);
        f << seccionAprendizaje();
        f << seccionPredicciones(10);
        f << seccionErrores(10);
        f << seccionMatrizConfusion();
        f << htmlFoot();

        f.close();
        cout << "\nReporte HTML guardado en: " << path << "\n";
        cout << "Ábrelo con: xdg-open " << path << "\n";
    }

private:
    // --------------------------------------------------
    // Imagen como canvas HTML (píxeles → cuadrados CSS)
    // --------------------------------------------------

    // Convierte un vector de píxeles [0,1] a un <canvas> HTML 28×28
    // escalado a displaySize px. Funciona para grises y RGB (3 canales).
    std::string imagenCanvas(
        const Vector& x,
        int           displaySize = 112,   // px en pantalla
        int           id          = 0
    ) const {
        int canales = (x.size() == imgSize * imgSize * 3) ? 3 : 1;
        std::ostringstream s;
        s << "<canvas id='c" << id << "' width='" << imgSize
          << "' height='" << imgSize
          << "' style='width:" << displaySize << "px;height:" << displaySize
          << "px;image-rendering:pixelated'></canvas>\n";
        s << "<script>\n(function(){\n";
        s << "var cv=document.getElementById('c" << id << "');\n";
        s << "var ctx=cv.getContext('2d');\n";
        s << "var id=ctx.createImageData(" << imgSize << "," << imgSize << ");\n";
        s << "var d=id.data;\n";

        // Serializar píxeles inline en el script
        s << "var px=[";
        for (int i = 0; i < imgSize * imgSize; i++) {
            int r, g, b;
            if (canales == 1) {
                int v = (int)(x(i) * 255.0 + 0.5);
                r = g = b = std::clamp(v, 0, 255);
            } else {
                r = std::clamp((int)(x(i)                          * 255.0 + 0.5), 0, 255);
                g = std::clamp((int)(x(imgSize*imgSize + i)        * 255.0 + 0.5), 0, 255);
                b = std::clamp((int)(x(imgSize*imgSize*2 + i)      * 255.0 + 0.5), 0, 255);
            }
            s << r << "," << g << "," << b;
            if (i < imgSize * imgSize - 1) s << ",";
        }
        s << "];\n";
        s << "for(var i=0;i<" << imgSize*imgSize << ";i++){"
          << "d[i*4]=px[i*3];d[i*4+1]=px[i*3+1];d[i*4+2]=px[i*3+2];d[i*4+3]=255;}\n";
        s << "ctx.putImageData(id,0,0);\n";
        s << "})();\n</script>\n";
        return s.str();
    }

    // --------------------------------------------------
    // Sección 1: Curvas de aprendizaje
    // --------------------------------------------------

    std::string seccionAprendizaje() const {
        std::ostringstream s;
        s << "<section><h2>📈 Curvas de Aprendizaje</h2>\n";
        s << "<div class='charts'>\n";

        // Loss
        s << "<div class='chart-box'><h3>Pérdida</h3>\n";
        s << "<canvas id='chartLoss'></canvas></div>\n";

        // Precisión
        s << "<div class='chart-box'><h3>Precisión (%)</h3>\n";
        s << "<canvas id='chartPrec'></canvas></div>\n";

        s << "</div>\n";

        // Chart.js data
        s << "<script>\n";
        s << "var epocas=[";
        for (int i = 0; i < (int)red.historialLoss.size(); i++)
            s << (i+1) << (i+1<(int)red.historialLoss.size()?",":"");
        s << "];\n";

        s << "var loss=[";
        for (int i = 0; i < (int)red.historialLoss.size(); i++)
            s << std::fixed << std::setprecision(6) << red.historialLoss[i]
              << (i+1<(int)red.historialLoss.size()?",":"");
        s << "];\n";

        s << "var prec=[";
        for (int i = 0; i < (int)red.historialPrecision.size(); i++)
            s << std::fixed << std::setprecision(2) << red.historialPrecision[i]
              << (i+1<(int)red.historialPrecision.size()?",":"");
        s << "];\n";

        s << R"(
function makeChart(id, labels, data, label, color) {
    new Chart(document.getElementById(id), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                borderColor: color,
                backgroundColor: color + '22',
                fill: true,
                tension: 0.3,
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { title: { display: true, text: 'Época' } },
                y: { title: { display: true, text: label } }
            }
        }
    });
}
makeChart('chartLoss', epocas, loss, 'Loss', '#e74c3c');
makeChart('chartPrec', epocas, prec, 'Precisión %', '#2ecc71');
)";
        s << "</script></section>\n";
        return s.str();
    }

    // --------------------------------------------------
    // Sección 2: Predicciones (primeras N muestras del test)
    // --------------------------------------------------

    std::string seccionPredicciones(int cantidad) const {
        std::ostringstream s;
        s << "<section><h2>🔍 Predicciones</h2>\n";
        s << "<div class='grid'>\n";

        int N = std::min(cantidad, (int)ds.test.X.size());
        for (int i = 0; i < N; i++) {
            Vector pred = red.predecir(ds.test.X[i]);
            int predIdx = argmax(pred);
            bool ok     = (predIdx == ds.test.y[i]);

            std::string border = ok ? "#2ecc71" : "#e74c3c";
            std::string icon   = ok ? "✅" : "❌";

            s << "<div class='card' style='border-color:" << border << "'>\n";
            s << imagenCanvas(ds.test.X[i], 112, i);
            s << "<div class='label'>" << icon << " "
              << "Pred: <b>" << ds.etiquetas[predIdx] << "</b><br>"
              << "Real: "    << ds.etiquetas[ds.test.y[i]]
              << "</div></div>\n";
        }

        s << "</div></section>\n";
        return s.str();
    }

    // --------------------------------------------------
    // Sección 3: Errores
    // --------------------------------------------------

    std::string seccionErrores(int cantidad) const {
        std::ostringstream s;
        s << "<section><h2>❌ Errores de Clasificación</h2>\n";

        vector<int> errIdx;
        for (int i = 0; i < (int)ds.test.X.size(); i++) {
            if (argmax(red.predecir(ds.test.X[i])) != ds.test.y[i])
                errIdx.push_back(i);
        }

        s << "<p>" << errIdx.size() << " errores de " << ds.test.X.size() << " muestras</p>\n";

        if (errIdx.empty()) {
            s << "<p>¡Sin errores!</p></section>\n";
            return s.str();
        }

        s << "<div class='grid'>\n";
        int N = std::min(cantidad, (int)errIdx.size());
        for (int k = 0; k < N; k++) {
            int i       = errIdx[k];
            int predIdx = argmax(red.predecir(ds.test.X[i]));
            int canvasId = 1000 + k;

            s << "<div class='card' style='border-color:#e74c3c'>\n";
            s << imagenCanvas(ds.test.X[i], 112, canvasId);
            s << "<div class='label'>"
              << "ID: " << i << "<br>"
              << "Pred: <b style='color:#e74c3c'>" << ds.etiquetas[predIdx] << "</b><br>"
              << "Real: <b style='color:#2ecc71'>" << ds.etiquetas[ds.test.y[i]] << "</b>"
              << "</div></div>\n";
        }
        s << "</div></section>\n";
        return s.str();
    }

    // --------------------------------------------------
    // Sección 4: Matriz de confusión
    // --------------------------------------------------

    std::string seccionMatrizConfusion() const {
        int NC = ds.nClases;
        vector<vector<int>> m(NC, vector<int>(NC, 0));
        int total = 0;

        for (int i = 0; i < (int)ds.test.X.size(); i++) {
            int pred = argmax(red.predecir(ds.test.X[i]));
            m[ds.test.y[i]][pred]++;
            total++;
        }

        // Nombre corto para encabezados
        auto corto = [](const std::string& s) -> std::string {
            return s.size() > 8 ? s.substr(0, 8) : s;
        };

        std::ostringstream s;
        s << "<section><h2>🟦 Matriz de Confusión</h2>\n";
        s << "<div style='overflow-x:auto'>\n";
        s << "<table class='cm'>\n";

        // Encabezado
        s << "<tr><th>Real \\ Pred</th>";
        for (int j = 0; j < NC; j++)
            s << "<th>" << corto(ds.etiquetas[j]) << "</th>";
        s << "</tr>\n";

        // Filas
        int maxVal = 1;
        for (int i = 0; i < NC; i++)
            for (int j = 0; j < NC; j++)
                maxVal = std::max(maxVal, m[i][j]);

        for (int i = 0; i < NC; i++) {
            s << "<tr><th>" << corto(ds.etiquetas[i]) << "</th>";
            for (int j = 0; j < NC; j++) {
                double intensity = (double)m[i][j] / maxVal;
                int    blue      = (int)(intensity * 200);
                std::string bg   = (i == j)
                    ? "background:rgba(46,204,113," + std::to_string(0.2 + intensity*0.7) + ")"
                    : (m[i][j] > 0
                        ? "background:rgba(231,76,60,"  + std::to_string(intensity*0.7) + ")"
                        : "background:#f8f9fa");
                (void)blue;

                s << "<td style='" << bg << "'>";
                if (m[i][j] > 0) s << m[i][j];
                s << "</td>";
            }
            s << "</tr>\n";
        }

        s << "</table></div></section>\n";
        return s.str();
    }

    // --------------------------------------------------
    // HTML boilerplate
    // --------------------------------------------------

    std::string htmlHead(const std::string& titulo) const {
        return R"(<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MLP — )" + titulo + R"(</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
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
<h1>🧠 MLP — )" + titulo + R"(</h1>
)";
    }

    std::string htmlFoot() const {
        return "</body></html>\n";
    }
};