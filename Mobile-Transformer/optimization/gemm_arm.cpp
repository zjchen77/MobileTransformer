#include <iostream>
#include <vector>
#include <arm_compute/runtime/NEON/NEFunctions.h>
#include <arm_compute/runtime/Scheduler.h>

void gemm_arm_compute(const std::vector<std::vector<float>>& A,
                      const std::vector<std::vector<float>>& B,
                      std::vector<std::vector<float>>& C) {
    size_t m = A.size();
    size_t n = B[0].size();
    size_t k = A[0].size();

    arm_compute::TensorShape shape_A(k, m);
    arm_compute::TensorShape shape_B(n, k);
    arm_compute::TensorShape shape_C(n, m);

    arm_compute::Tensor a;
    arm_compute::Tensor b;
    arm_compute::Tensor c;

    a.allocator()->init(arm_compute::TensorInfo(shape_A, 1, arm_compute::DataType::F32));
    b.allocator()->init(arm_compute::TensorInfo(shape_B, 1, arm_compute::DataType::F32));
    c.allocator()->init(arm_compute::TensorInfo(shape_C, 1, arm_compute::DataType::F32));

    a.allocator()->allocate();
    b.allocator()->allocate();
    c.allocator()->allocate();

    arm_compute::Window window_A;
    window_A.use_tensor_dimensions(a.info()->tensor_shape());
    arm_compute::Window window_B;
    window_B.use_tensor_dimensions(b.info()->tensor_shape());

    arm_compute::Iterator it_A(&a, window_A);
    arm_compute::Iterator it_B(&b, window_B);

    arm_compute::execute_window_loop(window_A, [&](const arm_compute::Coordinates& coord) {
        size_t x = coord.x();
        size_t y = coord.y();
        *reinterpret_cast<float*>(it_A.ptr()) = A[y][x];
    });

    arm_compute::execute_window_loop(window_B, [&](const arm_compute::Coordinates& coord) {
        size_t x = coord.x();
        size_t y = coord.y();
        *reinterpret_cast<float*>(it_B.ptr()) = B[y][x];
    });

    arm_compute::NEGEMM gemm;
    gemm.configure(&a, &b, nullptr, &c, 1.0f, 0.0f);
    gemm.run();

    arm_compute::Window window_C;
    window_C.use_tensor_dimensions(c.info()->tensor_shape());
    arm_compute::Iterator it_C(&c, window_C);

    arm_compute::execute_window_loop(window_C, [&](const arm_compute::Coordinates& coord) {
        size_t x = coord.x();
        size_t y = coord.y();
        C[y][x] = *reinterpret_cast<float*>(it_C.ptr());
    });
}

int main() {
    std::vector<std::vector<float>> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    std::vector<std::vector<float>> B = {
       
