#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include "curand_utils.h"

void validate(int N, float* mat1, float* mat2, float* mat3) {
    float max_error = 0;
    float tmp;
    float sum = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            tmp = 0.000001;
            for (int k = 0; k < N; k++) {
                tmp += mat1[i * N + k] * mat2[k * N + j];
            }
            sum += tmp;
            max_error = std::max(std::abs(tmp - mat3[i * N + j]), max_error);
        }
    }

    std::cout << "Maxdiff: " << max_error << ", avg: " << sum / N / N << " (should be close to 1)" << std::endl;
}

void set_vector(long int N, float* array, const float alpha, long int seed) {
    curandGenerator_t gen;
    curandRngType_t uniform_gen = CURAND_RNG_PSEUDO_XORWOW;

    CURAND_CHECK(curandCreateGenerator(&gen, uniform_gen));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CURAND_CHECK(curandGenerateUniform(gen, array, N));

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSscal(handle, N, &alpha, array, 1);

    curandDestroyGenerator(gen);
    cublasDestroy(handle);
}

__global__ void matmatmul_mykernel(int N, float* mat1, float* mat2, float* mat3) {
    float tmp = 0;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    for (int k = 0; k < N; k++) {
        if (row < N and col < N) tmp += mat1[row * N + k] * mat2[k * N + col];
    }

    if (row < N and col < N) mat3[row * N + col] = tmp;
}

void matmatmul_cublas(int N, float* mat1, float* mat2, float* mat3) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0, beta = 0.0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, mat2, N, mat1, N, &beta, mat3, N);
    cublasDestroy(handle);
}

void benchmark_matmatmul(int N, long repeat) {

    float* mat1_device, * mat2_device, * mat3_device;

    cudaMalloc(&mat1_device, N * N * sizeof(float));
    cudaMalloc(&mat2_device, N * N * sizeof(float));
    cudaMalloc(&mat3_device, N * N * sizeof(float));

    float* mat1_host = (float*)malloc(N * N * sizeof(float));
    float* mat2_host = (float*)malloc(N * N * sizeof(float));
    float* mat3_host = (float*)malloc(N * N * sizeof(float));

    float alpha = std::sqrt((float)4 / N);
    set_vector(N * N, mat1_device, alpha, 123);
    set_vector(N * N, mat2_device, alpha, 456);

    CUDA_CHECK(cudaMemcpy(mat1_host, mat1_device, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mat2_host, mat2_device, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    const unsigned int n_tests = 20;
    const unsigned long long int n_repeat = repeat > 0 ? repeat : std::max(1, 10000 / (int)N);
    double best = 1e10, worst = 0, avg = 0;

    // kernel configuration for optimized version
    dim3 blocksize(32, 32);
    dim3 gridsize((N + 31) / 32, (N + 31) / 32);

    for (unsigned int t = 0; t < n_tests; t++) {
        const auto t1 = std::chrono::steady_clock::now();
        for (unsigned rep = 0; rep < n_repeat; rep++) {
            matmatmul_cublas(N, mat1_device, mat2_device, mat3_device);
            // matmatmul_mykernel<<<gridsize, blocksize>>>(N, mat1_device, mat2_device, mat3_device);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
        }
        const double time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - t1).count();

        best = std::min(best, time / n_repeat);
        worst = std::max(worst, time / n_repeat);
        avg += time / n_repeat;
    }

    CUDA_CHECK(cudaMemcpy(mat3_host, mat3_device, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // validate(N, mat1_host, mat2_host, mat3_host);

    cudaFree(mat1_device);
    cudaFree(mat2_device);
    cudaFree(mat3_device);
    free(mat1_host);
    free(mat2_host);
    free(mat3_host);

    std::cout << "STREAM triad of size " << std::setw(12) << (long)N * N
        << " (" << std::setw(6) << N << " x " << std::setw(6) << N
        << ") : min/avg/max: " << std::setw(11) << best << " "
        << std::setw(11) << avg / n_tests << " " << std::setw(11) << worst
        << " seconds or " << std::setw(8) << 1e-9 * N * N * N * 2 / best
        << " GFlops/s or " << std::setw(8)
        << 1e-9 * sizeof(float) * (N * N * 3) / best << " GB/s" << std::endl;
}

int main(int argc, char** argv) {
    if (argc % 2 == 0) {
        std::cout << "Error, expected odd number of common line arguments" << std::endl;
        std::cout << "Expected line of the form" << std::endl;
        std::cout << "-min 100 -max 1e8 -repeat -1" << std::endl;
        std::abort();
    }

    long N_min = 8;
    long N_max = -1;
    long repeat = -1;

    for (int i = 1; i < argc; i += 2) {
        std::string opt = argv[i];
        if (opt == "-min") {
            N_min = static_cast<long>(std::stod(argv[i + 1]));
        }
        else if (opt == "-max") {
            N_max = static_cast<long>(std::stod(argv[i + 1]));
        }
        else if (opt == "-repeat") {
            repeat = static_cast<long>(std::stod(argv[i + 1]));
        }
        else {
            std::cout << "Unknown option " << opt << " - ignored!" << std::endl;
        }
    }

    if (N_min < 1) {
        std::cout << "Expected positive size for -min argument, got " << N_min << std::endl;
        return 0;
    }

    if (N_max < N_min) N_max = N_min;

    int N;
    for (N = N_min; N <= N_max; N = N * 1.1 + 1) {
        N = (N + 7) / 8 * 8;
        benchmark_matmatmul(N, repeat);
    }
}
