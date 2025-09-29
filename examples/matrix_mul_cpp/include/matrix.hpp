#pragma once

#include <vector>

typedef std::vector<std::vector<double>> Matrix;

Matrix multiply(const Matrix &a, const Matrix &b);
Matrix make_identity(std::size_t n);
Matrix make_test_matrix(std::size_t n);
bool is_close(const Matrix &a, const Matrix &b, double tol = 1e-6);
