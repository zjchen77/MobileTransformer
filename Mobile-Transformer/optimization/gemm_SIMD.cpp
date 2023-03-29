#include <iostream>
#include <vector>
#include <immintrin.h> // Include AVX intrinsics header

void gemm_avx(const std::vector<std::vector<double>>& A,
              const std::vector<std::vector<double>>& B,
              std::vector<std::vector<double>>& C) {
    size_t m = A.size();
    size_t n = B[0].size();
    size_t k = A[0].size();

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; j += 4) {
            __m256d c0 = _mm256_setzero_pd();
            for (size_t p = 0; p < k; ++p) {
                __m256d a0 = _mm256_broadcast_sd(&A[i][p]);
                __m256d b0 = _mm256_loadu_pd(&B[p][j]);
                c0 = _mm256_add_pd(c0, _mm256_mul_pd(a0, b0));
            }
            _mm256_storeu_pd(&C[i][j], c0);
        }
    }
}

int main() {
    std::vector<std::vector<double>> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    std::vector<std::vector<double>> B = {
        {9, 8, 7, 1},
        {6, 5, 4, 1},
        {3, 2, 1, 1}
    };

    size_t m = A.size();
    size_t n = B[0].size();

    std::vector<std::vector<double>> C(m, std::vector<double>(n, 0));

    gemm_avx(A, B, C);

    for (const auto& row : C) {
        for (const auto& element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
