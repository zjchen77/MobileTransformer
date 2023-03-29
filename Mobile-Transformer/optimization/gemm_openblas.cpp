#include <iostream>
#include <vector>
#include <cblas.h>

// g++ -o gemm_openblas_example gemm_openblas_example.cpp -lopenblas


void gemm_openblas(const std::vector<std::vector<double>>& A,
                   const std::vector<std::vector<double>>& B,
                   std::vector<std::vector<double>>& C) {
    size_t m = A.size();
    size_t n = B[0].size();
    size_t k = A[0].size();

    std::vector<double> A_flat(m * k);
    std::vector<double> B_flat(k * n);
    std::vector<double> C_flat(m * n);

    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < k; ++j)
            A_flat[i * k + j] = A[i][j];

    for (size_t i = 0; i < k; ++i)
        for (size_t j = 0; j < n; ++j)
            B_flat[i * n + j] = B[i][j];

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A_flat.data(), k, B_flat.data(), n, 0.0, C_flat.data(), n);

    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            C[i][j] = C_flat[i * n + j];
}

int main() {
    std::vector<std::vector<double>> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    std::vector<std::vector<double>> B = {
        {9, 8, 7},
        {6, 5, 4},
        {3, 2, 1}
    };

    size_t m = A.size();
    size_t n = B[0].size();

    std::vector<std::vector<double>> C(m, std::vector<double>(n, 0));

    gemm_openblas(A, B, C);

    for (const auto& row : C) {
        for (const auto& element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
