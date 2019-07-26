#include <stdio.h>
#include <stdlib.h>

const int INF = ((1 << 30) - 1);
const int V = 50010;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);
// void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

__global__ void cal_kernal_phase1(int B, int Round, int block_start_x, int block_start_y,  int n, int *Dist_gpu);

__global__ void cal_kernel(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int* Dist_gpu);

int n, m;
int *Dist;
int *Dist_gpu;

int main(int argc, char* argv[]) {
    input(argv[1]);
    int B = 32;
    block_FW(B);
    output(argv[2]);
    cudaFreeHost(Dist);
    cudaFree(Dist_gpu);
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

     cudaMallocHost(&Dist, sizeof(int)*n*n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i*n + j] = 0;
            } else {
                Dist[i*n + j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]*n + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i*n + j] >= INF) Dist[i*n + j] = INF;
        }
    }
    fwrite(Dist, sizeof(int), n*n, outfile);
    fclose(outfile);

}

int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW(int B) {
    int round = ceil(n, B);
    int num_thread = B * B;
    cudaMalloc((void**)&Dist_gpu, sizeof(int)*n*n);
    cudaMemcpy(Dist_gpu, Dist, sizeof(int)*n*n, cudaMemcpyHostToDevice);


   // dim3 block(B, B);

    for (int r = 0; r < round; ++r) {
        printf("%d %d\n", r, round);
        fflush(stdout);
        /* Phase 1*/
        //cal(B, r, r, r, 1, 1);
        cal_kernel<<<1, num_thread>>>(B, r, r, r, 1, 1, n, Dist_gpu);
        /* Phase 2*/
        cal_kernel<<<           r*1, num_thread >>>(B, r,     r,     0,             r,             1, n, Dist_gpu);
                cal_kernel<<< (round-r-1)*1, num_thread >>>(B, r,     r,  r +1,  round - r -1,             1, n, Dist_gpu);
                cal_kernel<<<           1*r, num_thread >>>(B, r,     0,     r,                  1,                     r, n, Dist_gpu);
                cal_kernel<<< 1*(round-r-1), num_thread >>>(B, r,  r +1,     r,             1,  round - r -1, n, Dist_gpu);

        /* Phase 3*/
        cal_kernel<<<                     r*r, num_thread >>>(B, r,     0,     0,            r,             r, n, Dist_gpu);
                cal_kernel<<<           (round-r-1)*r, num_thread >>>(B, r,     0,  r +1,  round -r -1,             r, n, Dist_gpu);
                cal_kernel<<<           r*(round-r-1), num_thread >>>(B, r,  r +1,     0,            r,  round - r -1, n, Dist_gpu);
                cal_kernel<<< (round-r-1)*(round-r-1), num_thread >>>(B, r,  r +1,  r +1,  round -r -1,  round - r -1, n, Dist_gpu);
    }
    cudaMemcpy(Dist, Dist_gpu, sizeof(int)*n*n, cudaMemcpyDeviceToHost);
}

// void cal(
//     int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
//     int block_end_x = block_start_x + block_height;
//     int block_end_y = block_start_y + block_width;

//     for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
//         for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
//             // To calculate B*B elements in the block (b_i, b_j)
//             // For each block, it need to compute B times
//             for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
//                 // To calculate original index of elements in the block (b_i, b_j)
//                 // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
//                 int block_internal_start_x = b_i * B;
//                 int block_internal_end_x = (b_i + 1) * B;
//                 int block_internal_start_y = b_j * B;
//                 int block_internal_end_y = (b_j + 1) * B;

//                 if (block_internal_end_x > n) block_internal_end_x = n;
//                 if (block_internal_end_y > n) block_internal_end_y = n;

//                 for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
//                     for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
//                         if (Dist[i][k] + Dist[k][j] < Dist[i][j]) {
//                             Dist[i][j] = Dist[i][k] + Dist[k][j];
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }


// __global__ void cal_kernal_phase1(int B, int Round, int block_start_x, int block_start_y,  int n, int* Dist_gpu)
// {

//     //only one blocks
//     int b_i = block_start_x ;
//      int b_j = block_start_y ;

//     int block_internal_start_x = b_i * B;
//     int block_internal_start_y = b_j * B;

//     for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
//         // To calculate original index of elements in the block (b_i, b_j)
//         // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2

//         __syncthreads();


//         int i = block_internal_start_x + threadIdx.x;
//         int j = block_internal_start_y + threadIdx.y;
//         if (i > n) i = n;
//         if (j > n) j = n;


//         if (Dist_gpu[i*n + k] + Dist_gpu[k*n + j] < Dist_gpu[i*n + j]) {
//             Dist_gpu[i*n + j] = Dist_gpu[i*n + k] + Dist_gpu[k*n + j];
//         }


//     }
// }

//general kernel
__global__ void cal_kernel(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int* Dist_gpu) {

        int b_i = block_start_x + blockIdx.x / block_width;
        int b_j = block_start_y + blockIdx.x % block_width;

        int inner_round = (B*B-1)/blockDim.x + 1;



        for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {

                for(int r=0; r<inner_round; r++){

                        int i = b_i * B + (threadIdx.x + r*blockDim.x) / B;
                        int j = b_j * B + (threadIdx.x + r*blockDim.x) % B;

                        if ((i>=n) | (j>=n)) continue ;

                        if (Dist_gpu[i*n+k] + Dist_gpu[k*n+j] < Dist_gpu[i*n+j]) {
                                Dist_gpu[i*n+j] = Dist_gpu[i*n+k] + Dist_gpu[k*n+j];
                        }
                }
                __syncthreads();
        }

}