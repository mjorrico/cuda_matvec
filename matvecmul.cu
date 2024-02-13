#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include "curand_utils.h"

#define BLOCKSIZE 1024
#define WARPSIZE 32
constexpr int WARPS_PER_BLOCK = BLOCKSIZE / WARPSIZE;

void validate(int M, int N, float* mat, float* vec_in, float* vec_out) {
    float max_error = 0;
    float tmp;
    float sum = 0;

    for (int i = 0; i < M; i++) {
        tmp = 0;
        for (int j = 0; j < N; j++) {
            tmp += mat[i * N + j] * vec_in[j];
        }
        sum += vec_out[i];
        max_error = std::max(std::abs(tmp - vec_out[i]), max_error);
    }

    std::cout << "Maxdiff: " << max_error << ", avg: " << sum / M << " (should be close to 1)" << std::endl;
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

__global__ void matvecmul_naive(int M, int N, float* mat, float* vec_in, float* vec_out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    float tmp = 0;
    if (idx < M) {
        for (int i = 0; i < N; i++) {
            tmp += mat[idx * N + i] * vec_in[i];
        }
    }
    vec_out[idx] = tmp;
}

__global__ void matvecmul_mykernel(int M, int N, float* mat, float* vec_in, float* vec_out) {
    extern __shared__ float shmem_vec_in[];
    register float tmp = 0.0;

    const int warp_id = threadIdx.x / WARPSIZE; // warp index within a block
    const int thr_id = threadIdx.x % WARPSIZE; // thread index within a warp
    const int mat_row = blockIdx.x * WARPS_PER_BLOCK + warp_id; // one wrap handles one row
    const int thr_len = (N + WARPSIZE - 1) / WARPSIZE; //  the # of scalar multiplication performed by a single thread

    // populating shared memory
    int shmem_idx;
    for (int i = 0; i < N; i += WARPSIZE) {
        shmem_idx = i + thr_id;
        shmem_vec_in[shmem_idx] = (shmem_idx < N) ? vec_in[shmem_idx] : 0;
    }
    __syncthreads();

    // moves pointer to assigned partition
    int product_idx;
    mat += mat_row * N;
    for (int i = 0; i < thr_len; i++) {
        product_idx = i * WARPSIZE + thr_id;
        if (product_idx < N && mat_row < M) tmp += mat[product_idx] * shmem_vec_in[product_idx];
    }
    __syncthreads();

    // atomicAdd but within wrap https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec4.pdf
    #pragma unroll
    for (int j = 1; j < WARPSIZE; j *= 2) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, j);
    }

    if (thr_id == 0 && mat_row < M) {
        vec_out[mat_row] = tmp;
    }
}

void matvecmul_cublas(int M, int N, float* mat, float* vec_in, float* vec_out) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0, beta = 0.0;
    cublasSgemv(handle, CUBLAS_OP_T,N, M, &alpha, mat, N, vec_in, 1, &beta, vec_out, 1);
    cublasDestroy(handle);
}

void benchmark_matvecmul(int M, int N, long repeat) {
    int shared_memory_size = ((N + 31) / 32 * 32) * sizeof(float);
    // std::cout << "Max memory: " << shared_memory_size << std::endl;
    // cudaFuncSetAttribute(matvecmul, cudaFuncAttributeMaxDynamicSharedMemorySize, 63000); // 48KB -> 65KB

    float* mat_device, * vec_in_device, * vec_out_device;

    cudaMalloc(&mat_device, M * N * sizeof(float));
    cudaMalloc(&vec_in_device, N * sizeof(float));
    cudaMalloc(&vec_out_device, M * sizeof(float));

    float* mat_host = (float*)malloc(M * N * sizeof(float));
    float* vec_in_host = (float*)malloc(N * sizeof(float));
    float* vec_out_host = (float*)malloc(M * sizeof(float));

    float alpha = std::sqrt((float)4 / N);
    set_vector(M * N, mat_device, alpha, 123);
    set_vector(N, vec_in_device, alpha, 456);

    CUDA_CHECK(cudaMemcpy(mat_host, mat_device, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vec_in_host, vec_in_device, N * sizeof(float), cudaMemcpyDeviceToHost));

    const unsigned int n_tests = 2;
    const unsigned long long int n_repeat = repeat > 0 ? repeat : std::max(1, 10000 / (int)N);
    double best = 1e10, worst = 0, avg = 0;

    // kernel configuration for naive
    dim3 blocksize_naive(32);
    dim3 gridsize_naive((M + 31) / 32);

    // kernel configuration for optimized version
    dim3 blocksize(BLOCKSIZE);
    dim3 gridsize((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    for (unsigned int t = 0; t < n_tests; t++) {
        const auto t1 = std::chrono::steady_clock::now();
        for (unsigned rep = 0; rep < n_repeat; rep++) {
            // matvecmul_naive<<<gridsize_naive, blocksize_naive>>>(M, N, mat_device, vec_in_device, vec_out_device);
            // matvecmul_cublas(M, N, mat_device, vec_in_device, vec_out_device);
            matvecmul_mykernel<<<gridsize, blocksize, shared_memory_size>>>(M, N, mat_device, vec_in_device, vec_out_device);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
        }
        const double time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - t1).count();

        best = std::min(best, time / n_repeat);
        worst = std::max(worst, time / n_repeat);
        avg += time / n_repeat;
    }

    CUDA_CHECK(cudaMemcpy(vec_out_host, vec_out_device, M * sizeof(float), cudaMemcpyDeviceToHost));

    // validate(M, N, mat_host, vec_in_host, vec_out_host);

    cudaFree(mat_device);
    cudaFree(vec_in_device);
    cudaFree(vec_out_device);
    free(mat_host);
    free(vec_in_host);
    free(vec_out_host);

    std::cout << "STREAM triad of size " << std::setw(12) << (long)M * N
        << " (" << std::setw(6) << M << " x " << std::setw(6) << N
        << ") : min/avg/max: " << std::setw(11) << best << " "
        << std::setw(11) << avg / n_tests << " " << std::setw(11) << worst
        << " seconds or " << std::setw(8) << 1e-6 * M / best
        << " MUPD/s or " << std::setw(8)
        << 1e-9 * sizeof(float) * (M * N + M + N) / best << " GB/s" << std::endl;
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

    int m, n;
    int count = 0;
    for (m = N_min; m <= N_max; m = m * 1.02 + 1) {
        m = (m + 7) / 8 * 8;
        n = 10000;
        benchmark_matvecmul(m, n, repeat);
    }
}
