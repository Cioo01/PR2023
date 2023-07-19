#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define BLOCK 32
#define MATRIX_SIZE 3072
#define N_ITER 3

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float* C, float* A, float* B, int wA_B, int size_C)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA_B * BLOCK_SIZE * 4 * by;
    int aEnd = aBegin + wA_B - 1;
    int aStep = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * 2 * bx;
    int bStep = BLOCK_SIZE * 2 * wA_B;

    float Csub[8] = { 0 };

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {
        __shared__ float As[4 * BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][2 * BLOCK_SIZE];

        if (a + wA_B * ty + tx < wA_B * wA_B)
        {
            As[ty][tx] = A[a + wA_B * ty + tx];
            if (ty + BLOCK_SIZE < 2 * wA_B && a + wA_B * (ty + BLOCK_SIZE) + tx < wA_B * wA_B)
            {
                As[ty + BLOCK_SIZE][tx] = A[a + wA_B * (ty + BLOCK_SIZE) + tx];
                if (ty + 2 * BLOCK_SIZE < 3 * wA_B && a + wA_B * (ty + 2 * BLOCK_SIZE) + tx < wA_B * wA_B)
                {
                    As[ty + 2 * BLOCK_SIZE][tx] = A[a + wA_B * (ty + 2 * BLOCK_SIZE) + tx];
                    if (ty + 3 * BLOCK_SIZE < 4 * wA_B && a + wA_B * (ty + 3 * BLOCK_SIZE) + tx < wA_B * wA_B)
                    {
                        As[ty + 3 * BLOCK_SIZE][tx] = A[a + wA_B * (ty + 3 * BLOCK_SIZE) + tx];
                    }
                }
            }
        }

        if (b + wA_B * ty + tx < wA_B * wA_B)
        {
            Bs[ty][tx] = B[b + wA_B * ty + tx];
            if (tx + BLOCK_SIZE < wA_B && b + wA_B * ty + tx + BLOCK_SIZE < wA_B * wA_B)
            {
                Bs[ty][tx + BLOCK_SIZE] = B[b + wA_B * ty + tx + BLOCK_SIZE];
            }
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            if (k + ty < wA_B && k + tx < wA_B)
            {
                Csub[0] += As[ty][k] * Bs[k][tx];
                Csub[4] += As[ty + 2 * BLOCK_SIZE][k] * Bs[k][tx];
                if (k + tx + BLOCK_SIZE < wA_B)
                {
                    Csub[2] += As[ty][k] * Bs[k][tx + BLOCK_SIZE];
                    Csub[6] += As[ty + 2 * BLOCK_SIZE][k] * Bs[k][tx + BLOCK_SIZE];
                }
            }
            if (k + ty + BLOCK_SIZE < 2 * wA_B && k + tx < wA_B)
            {
                Csub[1] += As[ty + BLOCK_SIZE][k] * Bs[k][tx];
                Csub[5] += As[ty + 3 * BLOCK_SIZE][k] * Bs[k][tx];
                if (k + tx + BLOCK_SIZE < wA_B)
                {
                    Csub[3] += As[ty + BLOCK_SIZE][k] * Bs[k][tx + BLOCK_SIZE];
                    Csub[7] += As[ty + 3 * BLOCK_SIZE][k] * Bs[k][tx + BLOCK_SIZE];
                }
            }
        }

        __syncthreads();
    }

    int c = wA_B * BLOCK_SIZE * 4 * by + BLOCK_SIZE * 2 * bx;

    if (c + wA_B * ty + tx < wA_B * wA_B)
    {
        C[c + wA_B * ty + tx] = Csub[0];
        if (c + wA_B * (ty + BLOCK_SIZE) + tx < wA_B * wA_B)
        {
            C[c + wA_B * (ty + BLOCK_SIZE) + tx] = Csub[1];
            if (c + wA_B * (ty + 2 * BLOCK_SIZE) + tx < wA_B * wA_B)
            {
                C[c + wA_B * (ty + 2 * BLOCK_SIZE) + tx] = Csub[4];
                if (c + wA_B * (ty + 3 * BLOCK_SIZE) + tx < wA_B * wA_B)
                {
                    C[c + wA_B * (ty + 3 * BLOCK_SIZE) + tx] = Csub[5];
                }
            }
        }
        if (c + wA_B * ty + tx + BLOCK_SIZE < wA_B * wA_B)
        {
            C[c + wA_B * ty + tx + BLOCK_SIZE] = Csub[2];
            if (c + wA_B * (ty + BLOCK_SIZE) + tx + BLOCK_SIZE < wA_B * wA_B)
            {
                C[c + wA_B * (ty + BLOCK_SIZE) + tx + BLOCK_SIZE] = Csub[3];
                if (c + wA_B * (ty + 2 * BLOCK_SIZE) + tx + BLOCK_SIZE < wA_B * wA_B)
                {
                    C[c + wA_B * (ty + 2 * BLOCK_SIZE) + tx + BLOCK_SIZE] = Csub[6];
                    if (c + wA_B * (ty + 3 * BLOCK_SIZE) + tx + BLOCK_SIZE < wA_B * wA_B)
                    {
                        C[c + wA_B * (ty + 3 * BLOCK_SIZE) + tx + BLOCK_SIZE] = Csub[7];
                    }
                }
            }
        }
    }
}

void constantInit(float* data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int matrixMultiply(int argc, char** argv, int block_size, dim3& dimsA, dim3& dimsB)
{
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A;
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B;
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));

    // Initialize host memory
    const float valB = 0.01f;
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);

    // Allocate device memory
    float* d_A, * d_B, * d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float* h_C;
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

    if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C), mem_size_C));
    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // copy host memory to device
    checkCudaErrors(
        cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    // Setup execution parameters
    dim3 threads(block_size, block_size); // matrix of threads
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y / 2); // matrix of blocks

    int size_C = dimsC.x * dimsC.y;

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel 
    if (block_size == 8)
    {
        matrixMulCUDA<8> << < grid, threads >> > (d_C, d_A, d_B, dimsA.x, size_C);
    }
    else if (block_size == 16)
    {
        matrixMulCUDA<16> << < grid, threads >> > (d_C, d_A, d_B, dimsA.x, size_C);
    }
    else
    {
        matrixMulCUDA<32> << < grid, threads >> > (d_C, d_A, d_B, dimsA.x, size_C);
    }

    printf("done\n");
    cudaDeviceSynchronize();

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    // Execute the kernel
    for (int j = 0; j < N_ITER; j++) {
        if (block_size == 8)
        {
            matrixMulCUDA<8> << < grid, threads >> > (d_C, d_A, d_B, dimsA.x, size_C);
        }
        else if (block_size == 16)
        {
            matrixMulCUDA<16> << < grid, threads >> > (d_C, d_A, d_B, dimsA.x, size_C);
        }
        else
        {
            matrixMulCUDA<32> << < grid, threads >> > (d_C, d_A, d_B, dimsA.x, size_C);
        }
    }

    // Record the stop event

    checkCudaErrors(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / N_ITER;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
        static_cast<double>(dimsA.y) *
        static_cast<double>(dimsB.x);
    double gigaFlops =
        (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
        " WorkgroupSize= %u threads/block\n",
        gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

    // Copy result from device to host
    checkCudaErrors(
        cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6;  // machine zero

    for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
        double abs_err = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                i, h_C[i], dimsA.x * valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    printf(
        "\nNOTE: The CUDA Samples are not meant for performance "
        "measurements. Results may vary when GPU Boost is enabled.\n");

    if (correct) {
        return EXIT_SUCCESS;
    }
    else {
        return EXIT_FAILURE;
    }
}


/**
 * Program main
 */
int main(int argc, char** argv)
{
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char**)argv, "help") ||
        checkCmdLineFlag(argc, (const char**)argv, "?"))
    {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices must be equal.\n");
        exit(EXIT_SUCCESS);
    }

    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;

    dim3 dimsA(MATRIX_SIZE, MATRIX_SIZE, 1);
    dim3 dimsB(MATRIX_SIZE, MATRIX_SIZE, 1);

    // width of Matrix A
    if (checkCmdLineFlag(argc, (const char**)argv, "wA")) {
        dimsA.x = getCmdLineArgumentInt(argc, (const char**)argv, "wA");
    }

    // height of Matrix A
    if (checkCmdLineFlag(argc, (const char**)argv, "hA")) {
        dimsA.y = getCmdLineArgumentInt(argc, (const char**)argv, "hA");
    }

    // width of Matrix B
    if (checkCmdLineFlag(argc, (const char**)argv, "wB")) {
        dimsB.x = getCmdLineArgumentInt(argc, (const char**)argv, "wB");
    }

    // height of Matrix B
    if (checkCmdLineFlag(argc, (const char**)argv, "hB")) {
        dimsB.y = getCmdLineArgumentInt(argc, (const char**)argv, "hB");
    }

    if (dimsA.x != dimsB.y) {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
            dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
        dimsB.x, dimsB.y);

    checkCudaErrors(cudaProfilerStart());
    int matrix_result = matrixMultiply(argc, argv, BLOCK, dimsA, dimsB);
    checkCudaErrors(cudaProfilerStop());

    exit(matrix_result);
}