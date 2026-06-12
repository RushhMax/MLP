#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

// ============================================================
// Vec — vector de doubles con operaciones element-wise
// ============================================================

class Vec {
    std::vector<double> d_;
public:
    Vec() {}
    explicit Vec(int n) : d_(n, 0.0) {}
    Vec(int n, double val) : d_(n, val) {}

    static Vec Zero(int n) { return Vec(n, 0.0); }

    int    size()      const { return (int)d_.size(); }
    double& operator()(int i)       { return d_[i]; }
    double  operator()(int i) const { return d_[i]; }

    Vec operator-() const {
        Vec r(size());
        for (int i = 0; i < size(); i++) r.d_[i] = -d_[i];
        return r;
    }

    // --- Vec op Vec ---
    Vec operator+(const Vec& b) const {
        Vec r(size());
        for (int i = 0; i < size(); i++) r.d_[i] = d_[i] + b.d_[i];
        return r;
    }
    Vec operator-(const Vec& b) const {
        Vec r(size());
        for (int i = 0; i < size(); i++) r.d_[i] = d_[i] - b.d_[i];
        return r;
    }
    Vec operator*(const Vec& b) const {   // element-wise
        Vec r(size());
        for (int i = 0; i < size(); i++) r.d_[i] = d_[i] * b.d_[i];
        return r;
    }
    Vec operator/(const Vec& b) const {   // element-wise
        Vec r(size());
        for (int i = 0; i < size(); i++) r.d_[i] = d_[i] / b.d_[i];
        return r;
    }
    Vec& operator+=(const Vec& b) {
        for (int i = 0; i < size(); i++) d_[i] += b.d_[i];
        return *this;
    }
    Vec& operator-=(const Vec& b) {
        for (int i = 0; i < size(); i++) d_[i] -= b.d_[i];
        return *this;
    }

    // --- Vec op scalar ---
    Vec operator+(double s) const {
        Vec r(size());
        for (int i = 0; i < size(); i++) r.d_[i] = d_[i] + s;
        return r;
    }
    Vec operator-(double s) const {
        Vec r(size());
        for (int i = 0; i < size(); i++) r.d_[i] = d_[i] - s;
        return r;
    }
    Vec operator*(double s) const {
        Vec r(size());
        for (int i = 0; i < size(); i++) r.d_[i] = d_[i] * s;
        return r;
    }
    Vec operator/(double s) const {
        Vec r(size());
        for (int i = 0; i < size(); i++) r.d_[i] = d_[i] / s;
        return r;
    }

    // --- scalar op Vec ---
    friend Vec operator+(double s, const Vec& v) { return v + s; }
    friend Vec operator*(double s, const Vec& v) { return v * s; }
    friend Vec operator-(double s, const Vec& v) {
        Vec r(v.size());
        for (int i = 0; i < v.size(); i++) r.d_[i] = s - v.d_[i];
        return r;
    }
    friend Vec operator/(double s, const Vec& v) {
        Vec r(v.size());
        for (int i = 0; i < v.size(); i++) r.d_[i] = s / v.d_[i];
        return r;
    }

    // --- Element-wise math ---
    Vec exp() const {
        Vec r(size());
        for (int i = 0; i < size(); i++) r.d_[i] = std::exp(d_[i]);
        return r;
    }
    Vec log() const {
        Vec r(size());
        for (int i = 0; i < size(); i++) r.d_[i] = std::log(d_[i]);
        return r;
    }
    Vec sqrt() const {
        Vec r(size());
        for (int i = 0; i < size(); i++) r.d_[i] = std::sqrt(d_[i]);
        return r;
    }
    Vec square() const {
        Vec r(size());
        for (int i = 0; i < size(); i++) r.d_[i] = d_[i] * d_[i];
        return r;
    }

    // --- Reductions ---
    double sum() const {
        double s = 0.0;
        for (double x : d_) s += x;
        return s;
    }
    double max() const {
        double m = d_[0];
        for (double x : d_) if (x > m) m = x;
        return m;
    }
    double squaredNorm() const {
        double s = 0.0;
        for (double x : d_) s += x * x;
        return s;
    }
    double dot(const Vec& b) const {
        double s = 0.0;
        for (int i = 0; i < size(); i++) s += d_[i] * b.d_[i];
        return s;
    }

    // --- Clamp / mask ---
    Vec cwiseMax(double s) const {
        Vec r(size());
        for (int i = 0; i < size(); i++) r.d_[i] = std::max(d_[i], s);
        return r;
    }
    Vec cwiseMax(const Vec& b) const {
        Vec r(size());
        for (int i = 0; i < size(); i++) r.d_[i] = std::max(d_[i], b.d_[i]);
        return r;
    }
    Vec cwiseMin(double s) const {
        Vec r(size());
        for (int i = 0; i < size(); i++) r.d_[i] = std::min(d_[i], s);
        return r;
    }
    Vec clamp(double lo, double hi) const {
        return cwiseMax(lo).cwiseMin(hi);
    }
    // Retorna 1.0 donde d_[i] > s, 0.0 en caso contrario
    Vec gt(double s) const {
        Vec r(size());
        for (int i = 0; i < size(); i++) r.d_[i] = d_[i] > s ? 1.0 : 0.0;
        return r;
    }
};

// ============================================================
// Mat — matriz de doubles (row-major)
// ============================================================

class Mat {
    int rows_, cols_;
    std::vector<double> d_;
public:
    Mat() : rows_(0), cols_(0) {}
    Mat(int r, int c) : rows_(r), cols_(c), d_(r * c, 0.0) {}

    static Mat Zero(int r, int c) { return Mat(r, c); }

    int rows() const { return rows_; }
    int cols() const { return cols_; }

    double& operator()(int r, int c)       { return d_[r * cols_ + c]; }
    double  operator()(int r, int c) const { return d_[r * cols_ + c]; }

    // --- Mat * Vec  (producto matricial) ---
    Vec operator*(const Vec& v) const {
        Vec r(rows_);
        for (int i = 0; i < rows_; i++) {
            double s = 0.0;
            for (int j = 0; j < cols_; j++) s += d_[i * cols_ + j] * v(j);
            r(i) = s;
        }
        return r;
    }

    // --- Mat^T * Vec  (evita copiar la transpuesta) ---
    Vec transposeMul(const Vec& v) const {
        Vec r(cols_);
        for (int j = 0; j < cols_; j++) {
            double s = 0.0;
            for (int i = 0; i < rows_; i++) s += d_[i * cols_ + j] * v(i);
            r(j) = s;
        }
        return r;
    }

    // --- Producto exterior: result(i,j) = a(i) * b(j) ---
    static Mat outer(const Vec& a, const Vec& b) {
        Mat r(a.size(), b.size());
        for (int i = 0; i < a.size(); i++)
            for (int j = 0; j < b.size(); j++)
                r.d_[i * b.size() + j] = a(i) * b(j);
        return r;
    }

    // --- Últimas n filas ---
    Mat bottomRows(int n) const {
        int start = rows_ - n;
        Mat r(n, cols_);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < cols_; j++)
                r.d_[i * cols_ + j] = d_[(start + i) * cols_ + j];
        return r;
    }

    // --- Mat op Mat ---
    Mat operator+(const Mat& b) const {
        Mat r(rows_, cols_);
        for (int i = 0; i < (int)d_.size(); i++) r.d_[i] = d_[i] + b.d_[i];
        return r;
    }
    Mat operator-(const Mat& b) const {
        Mat r(rows_, cols_);
        for (int i = 0; i < (int)d_.size(); i++) r.d_[i] = d_[i] - b.d_[i];
        return r;
    }
    Mat& operator+=(const Mat& b) {
        for (int i = 0; i < (int)d_.size(); i++) d_[i] += b.d_[i];
        return *this;
    }
    Mat& operator-=(const Mat& b) {
        for (int i = 0; i < (int)d_.size(); i++) d_[i] -= b.d_[i];
        return *this;
    }

    // --- Mat op scalar ---
    Mat operator*(double s) const {
        Mat r(rows_, cols_);
        for (int i = 0; i < (int)d_.size(); i++) r.d_[i] = d_[i] * s;
        return r;
    }
    Mat operator/(double s) const {
        Mat r(rows_, cols_);
        for (int i = 0; i < (int)d_.size(); i++) r.d_[i] = d_[i] / s;
        return r;
    }
    Mat operator+(double s) const {
        Mat r(rows_, cols_);
        for (int i = 0; i < (int)d_.size(); i++) r.d_[i] = d_[i] + s;
        return r;
    }
    Mat& operator*=(double s) {
        for (double& x : d_) x *= s;
        return *this;
    }

    friend Mat operator*(double s, const Mat& m) { return m * s; }

    // --- Element-wise ops ---
    Mat square() const {
        Mat r(rows_, cols_);
        for (int i = 0; i < (int)d_.size(); i++) r.d_[i] = d_[i] * d_[i];
        return r;
    }
    Mat sqrt() const {
        Mat r(rows_, cols_);
        for (int i = 0; i < (int)d_.size(); i++) r.d_[i] = std::sqrt(d_[i]);
        return r;
    }
    Mat cwiseDiv(const Mat& b) const {
        Mat r(rows_, cols_);
        for (int i = 0; i < (int)d_.size(); i++) r.d_[i] = d_[i] / b.d_[i];
        return r;
    }
};

using Vector = Vec;
using Matrix = Mat;
