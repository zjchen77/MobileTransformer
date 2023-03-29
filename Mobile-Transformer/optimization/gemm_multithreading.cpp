#include <iostream>
#include <vector>
#include <thread>

void gemm_threaded(const std::vector<std::vector<double>>& A,
                   const std::vector<std::vector<double>>& B,
                   std::vector<std::vector<double>>& C,
                   size_t start_row, size_t end_row) {
    size_t n = B[0].size();
    size_t k = A[0].size();

    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t p = 0; p < k; ++p) {
                sum += A[i][p] * B[p][j];
            }
            C[i][j] = sum;
        }
    }
}

void gemm_multithreaded(const std::vector<std::vector<double>>& A,
                        const std::vector<std::vector<double>>& B,
                        std::vector<std::vector<double>>& C) {
    size_t m = A.size();
    size_t num_threads = std::thread::hardware_concurrency();

    std::vector<std::thread> threads;
    size_t rows_per_thread = m / num_threads;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start_row = i * rows_per_thread;
        size_t end_row = (i == num_threads - 1) ? m : start_row + rows_per_thread;

        threads.emplace_back(gemm_threaded, std::ref(A), std::ref(B), std::ref(C), start_row, end_row);
    }

    for (auto& t : threads) {
        t.join();
    }
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

    gemm_multithreaded(A, B, C);

    for (const auto& row : C) {
        for (const auto& element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
