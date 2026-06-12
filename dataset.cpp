#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <iomanip>
#include <filesystem>
#include <memory>
#include "linalg.h"

// numeros aleatorio semillas 42
static std::mt19937 rng(42);

struct Split;
void shuffleSplit(Split& s);

namespace fs = std::filesystem;

using namespace std;


// ==========================================================
// DATASET — estructura y cargador base
// ==========================================================

struct Split {
    vector<Vector> X;
    vector<int> y;
    vector<string> nombres;  // nombre de archivo o índice (debug)
};

struct DatasetInfo {
    string nombre;
    int inputDim;
    int nClases;
    vector<string> etiquetas; // nombre de cada clase
    Split train;
    Split test;
};

class DatasetLoader {
public:
    virtual ~DatasetLoader() = default;
    virtual DatasetInfo cargar() = 0;
};

// ==========================================================
// LOADER: MNIST IDX
// ==========================================================

class MNISTLoader : public DatasetLoader {
    public:
        string trainImages;
        string trainLabels;
        string testImages;
        string testLabels;
        int maxTrain;
        int maxTest;

        MNISTLoader(
            const string& trainImg = "train-images-idx3-ubyte",
            const string& trainLbl = "train-labels-idx1-ubyte",
            const string& testImg  = "t10k-images-idx3-ubyte",
            const string& testLbl  = "t10k-labels-idx1-ubyte",
            int maxTrain_ = 3000,
            int maxTest_ = 1000
        ) : trainImages(trainImg), trainLabels(trainLbl),
            testImages(testImg),   testLabels(testLbl),
            maxTrain(maxTrain_),   maxTest(maxTest_) {}

        DatasetInfo cargar() override {
            DatasetInfo info;
            info.nombre   = "MNIST";
            info.inputDim = 784;
            info.nClases  = 10;
            for (int i = 0; i < 10; i++)
                info.etiquetas.push_back(std::to_string(i));

            info.train = leerSplit(trainImages, trainLabels, maxTrain);
            info.test  = leerSplit(testImages,  testLabels,  maxTest);
            return info;
        }

    private:
        static int leerInt32(ifstream& f) {
            unsigned char b[4];
            f.read(reinterpret_cast<char*>(b), 4);
            return (b[0]<<24)|(b[1]<<16)|(b[2]<<8)|b[3];
        }

        Split leerSplit(const string& imgPath,const string& lblPath, int maxN) {
            ifstream fimg(imgPath, ios::binary);
            ifstream flbl(lblPath, ios::binary);
            if (!fimg) throw runtime_error("No se pudo abrir: " + imgPath);
            if (!flbl) throw runtime_error("No se pudo abrir: " + lblPath);

            leerInt32(fimg); // magic
            int n      = leerInt32(fimg);
            int rows   = leerInt32(fimg);
            int cols   = leerInt32(fimg);
            int pixels = rows * cols;

            leerInt32(flbl); // magic
            leerInt32(flbl); // n labels

            if (maxN > 0) n = std::min(n, maxN);

            Split s;
            s.X.reserve(n); s.y.reserve(n); s.nombres.reserve(n);

            for (int i = 0; i < n; i++) {
                Vector img(pixels);
                for (int p = 0; p < pixels; p++) {
                    unsigned char byte;
                    fimg.read(reinterpret_cast<char*>(&byte), 1);
                    img(p) = byte / 255.0;
                }
                unsigned char lbl;
                flbl.read(reinterpret_cast<char*>(&lbl), 1);

                s.X.push_back(img);
                s.y.push_back(lbl);
                s.nombres.push_back("idx_" + std::to_string(i));
            }
            return s;
        }
};

// ==========================================================
// LOADER: IMÁGENES EN CARPETAS (Simpsons, etc.)
// ==========================================================

class ImageFolderLoader : public DatasetLoader {
    public:
        string trainPath;
        string testPath;
        vector<string> clases;
        int imgSize;
        int maxPorClase; // 0 = sin límite, 100 = máximo 100 imágenes por clase, etc.
        bool color; // false = grises (1 canal), true = RGB (3 canales)

        ImageFolderLoader( const string& trainPath_, const string& testPath_, const vector<string>& clases_,
            int  imgSize_ = 28,
            int  maxPorClase_ = 0,
            bool color_ = false
        ) : trainPath(trainPath_), testPath(testPath_),
            clases(clases_),       imgSize(imgSize_),
            maxPorClase(maxPorClase_), color(color_) {}

        DatasetInfo cargar() override {
            int canales = color ? 3 : 1;

            DatasetInfo info;
            info.nombre   = "ImageFolder";
            info.inputDim = imgSize * imgSize * canales;
            info.nClases  = (int)clases.size();
            info.etiquetas = clases;

            cout << "\nCargando entrenamiento desde: " << trainPath << "\n";
            info.train = leerSplit(trainPath);
            shuffleSplit(info.train);

            cout << "Cargando prueba desde: " << testPath << "\n";
            info.test = leerSplit(testPath);

            return info;
        }

    private:
        Vector cargarImagen(const std::string& path) {
            int w, h, ch;
            int req = color ? STBI_rgb : STBI_grey;
            unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, req);
            if (!data) throw std::runtime_error("No se pudo cargar: " + path);

            int canales = color ? 3 : 1;
            Vector img(imgSize * imgSize * canales);

            for (int c = 0; c < canales; c++) {
                for (int iy = 0; iy < imgSize; iy++) {
                    for (int ix = 0; ix < imgSize; ix++) {
                        float srcX = (ix + 0.5f) * w / (float)imgSize - 0.5f;
                        float srcY = (iy + 0.5f) * h / (float)imgSize - 0.5f;

                        int x0 = std::max(0, (int)srcX);
                        int y0 = std::max(0, (int)srcY);
                        int x1 = std::min(w-1, x0+1);
                        int y1 = std::min(h-1, y0+1);
                        float dx = srcX - x0, dy = srcY - y0;

                        float p00 = data[(y0*w + x0)*canales + c];
                        float p10 = data[(y0*w + x1)*canales + c];
                        float p01 = data[(y1*w + x0)*canales + c];
                        float p11 = data[(y1*w + x1)*canales + c];

                        float val = (1-dy)*((1-dx)*p00 + dx*p10)+dy *((1-dx)*p01 + dx*p11);

                        img(c * imgSize * imgSize + iy * imgSize + ix) = val / 255.0;
                    }
                }
            }

            stbi_image_free(data);
            return img;
        }

        Split leerSplit(const std::string& basePath) {
            Split s;
            for (int ci = 0; ci < (int)clases.size(); ci++) {
                fs::path dir = fs::path(basePath) / clases[ci];
                if (!fs::exists(dir)) {
                    std::cerr << "  ADVERTENCIA: no existe " << dir << "\n";
                    continue;
                }
                int count = 0;
                for (auto& entry : fs::directory_iterator(dir)) {
                    if (!entry.is_regular_file()) continue;
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;
                    if (maxPorClase > 0 && count >= maxPorClase) break;
                    try {
                        s.X.push_back(cargarImagen(entry.path().string()));
                        s.y.push_back(ci);
                        s.nombres.push_back(entry.path().filename().string());
                        count++;
                    } catch (...) {}
                }
                cout << "  " << std::setw(30) << std::left << clases[ci] << " → " << count << " imágenes\n";
            }
            return s;
        }
};


// ==========================================================
// LOADER: HASYv2 (símbolos matemáticos 32×32 gris, fold-based)
// ==========================================================

class HASYLoader : public DatasetLoader {
public:
    string dataPath;
    int fold;
    int maxTrain;
    int maxTest;

    HASYLoader(
        const string& dataPath_ = "data/hasy",
        int fold_     = 1,
        int maxTrain_ = 0,
        int maxTest_  = 0
    ) : dataPath(dataPath_), fold(fold_),
        maxTrain(maxTrain_), maxTest(maxTest_) {}

    DatasetInfo cargar() override {
        DatasetInfo info;
        info.nombre   = "HASYv2";
        info.inputDim = 32 * 32;

        // 1. Leer symbols.csv: symbol_id,latex,training_samples,test_samples
        map<int,string> idToLatex;
        map<int,int>    idToClass;
        {
            string path = dataPath + "/symbols.csv";
            ifstream f(path);
            if (!f) throw runtime_error("No se pudo abrir: " + path);
            string line;
            getline(f, line); // header
            int ci = 0;
            while (getline(f, line)) {
                if (line.empty()) continue;
                auto c1 = line.find(',');
                auto c2 = line.find(',', c1 + 1);
                int sid       = stoi(line.substr(0, c1));
                string latex  = line.substr(c1 + 1, c2 - c1 - 1);
                idToLatex[sid] = latex;
                idToClass[sid] = ci++;
            }
        }
        info.nClases = (int)idToLatex.size();
        info.etiquetas.resize(info.nClases);
        for (auto& [sid, ci] : idToClass)
            info.etiquetas[ci] = idToLatex[sid];

        // 2. Cargar el fold solicitado
        string foldDir = dataPath + "/classification-task/fold-" + to_string(fold);
        cout << "\nCargando HASYv2 fold-" << fold << " desde: " << foldDir << "\n";
        cout << "  Clases: " << info.nClases << "  |  InputDim: " << info.inputDim << "\n";

        info.train = leerSplit(foldDir + "/train.csv", foldDir, idToClass, maxTrain, "train");
        shuffleSplit(info.train);
        info.test  = leerSplit(foldDir + "/test.csv",  foldDir, idToClass, maxTest,  "test");
        return info;
    }

private:
    Vector cargarImagen(const string& path) {
        int w, h, ch;
        unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, STBI_grey);
        if (!data) throw runtime_error("No se pudo cargar: " + path);
        Vector img(32 * 32);
        for (int i = 0; i < 32 * 32; i++)
            img(i) = data[i] / 255.0;
        stbi_image_free(data);
        return img;
    }

    Split leerSplit(const string& csvPath, const string& foldDir,
                    const map<int,int>& idToClass, int maxN,
                    const string& splitName) {
        ifstream f(csvPath);
        if (!f) throw runtime_error("No se pudo abrir: " + csvPath);

        // Paso 1: indexar todas las filas por clase (sin cargar imágenes aún)
        map<int, vector<string>> byClass; // classIdx → rutas relativas
        string line;
        getline(f, line); // header: path,symbol_id,latex,user_id
        while (getline(f, line)) {
            if (line.empty()) continue;
            auto c1 = line.find(',');
            auto c2 = line.find(',', c1 + 1);
            string relPath = line.substr(0, c1);
            int sid = stoi(line.substr(c1 + 1, c2 - c1 - 1));
            auto it = idToClass.find(sid);
            if (it == idToClass.end()) continue;
            byClass[it->second].push_back(move(relPath));
        }

        // Paso 2: seleccionar muestras distribuidas por clase
        int nActive = (int)byClass.size();
        int perClass = (maxN <= 0) ? INT_MAX : max(1, maxN / nActive);

        vector<pair<int,string>> selected; // (classIdx, relPath)
        for (auto& [ci, paths] : byClass) {
            vector<int> idx(paths.size());
            iota(idx.begin(), idx.end(), 0);
            shuffle(idx.begin(), idx.end(), rng);
            int take = min((int)paths.size(), perClass);
            for (int k = 0; k < take; k++)
                selected.emplace_back(ci, paths[idx[k]]);
        }
        shuffle(selected.begin(), selected.end(), rng);

        // Paso 3: cargar imágenes de la selección
        Split s;
        s.X.reserve(selected.size());
        s.y.reserve(selected.size());
        s.nombres.reserve(selected.size());
        int count = 0, errores = 0;

        for (auto& [ci, relPath] : selected) {
            fs::path imgPath = (fs::path(foldDir) / relPath).lexically_normal();
            try {
                s.X.push_back(cargarImagen(imgPath.string()));
                s.y.push_back(ci);
                s.nombres.push_back(imgPath.filename().string());
                count++;
                if (count % 5000 == 0)
                    cout << "  [" << splitName << "] " << count << " imágenes cargadas...\n";
            } catch (...) { errores++; }
        }
        cout << "  [" << splitName << "] total: " << count << " imágenes"
             << " (" << nActive << " clases, ~" << perClass << "/clase)\n";
        if (errores) cout << "  (" << errores << " errores)\n";
        return s;
    }
};


void shuffleSplit(Split& s) {
    int N = (int)s.X.size();
    vector<int> idx(N);
    iota(idx.begin(), idx.end(), 0);
    shuffle(idx.begin(), idx.end(), rng);
    Split t;
    t.X.reserve(N); t.y.reserve(N); t.nombres.reserve(N);
    for (int i : idx) {
        t.X.push_back(s.X[i]);
        t.y.push_back(s.y[i]);
        t.nombres.push_back(s.nombres[i]);
    }
    s = move(t);
}
