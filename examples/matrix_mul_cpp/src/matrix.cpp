#include "matrix.hpp"

#include <cmath>

Matrix multiply(const Matrix &a, const Matrix &b) {
    if (a.empty() || b.empty()) {
        return Matrix();
    }
    std::size_t rows = a.size();
    std::size_t cols = b[0].size();
    std::size_t inner = b.size();
    Matrix result(rows, std::vector<double>(cols, 0.0));
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t k = 0; k < inner; ++k) {
            double value = a[i][k];
            for (std::size_t j = 0; j < cols; ++j) {
                result[i][j] += value * b[k][j];
            }
        }
    }
    return result;
}

Matrix make_identity(std::size_t n) {
    Matrix id(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        id[i][i] = 1.0;
    }
    return id;
}

Matrix make_test_matrix(std::size_t n) {
    Matrix m(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            m[i][j] = static_cast<double>((i + 1) * (j + 2));
        }
    }
    return m;
}

bool is_close(const Matrix &a, const Matrix &b, double tol) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) {
            return false;
        }
        for (std::size_t j = 0; j < a[i].size(); ++j) {
            if (std::fabs(a[i][j] - b[i][j]) > tol) {
                return false;
            }
        }
    }
    return true;
}
