#include <assert.h>
#include <stdio.h>

__global__ void computeKernel(int n, int* arr) {
    for (int i = threadIdx.x+(blockDim.x*blockIdx.x)+2; i <= n; i+=64*16) {
        bool isPrime = true;
        for (int j = 2; j * j <= i; j++) {
            if (i % j == 0) {
                isPrime = false;
            }
        }
        if (isPrime) {
            arr[i] = i;
        } else {
            arr[i] = -i;
        }
    }
}

__global__ void sumKernel(int n, int* arr, int* result) {
    for (int i = threadIdx.x+(blockDim.x*blockIdx.x); i <= n; i+=64*16) {

        atomicAdd(result, arr[i]);
    }
}

int main(int argc, char** argv) {
    assert(argc == 2);
    int n = atoll(argv[1]);
    int* devArr;
    cudaMalloc(&devArr, sizeof(int) * (n + 1));
    computeKernel<<<16, 64>>>(n, devArr);
    int* devResult;
    // ???
    cudaMalloc(&devResult, sizeof(int));

    sumKernel<<<16, 64>>>(n, devArr, devResult);
    int hostResult;
    cudaMemcpy(&hostResult, devResult, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", hostResult);
    cudaFree(devArr);
    cudaFree(devResult);
}