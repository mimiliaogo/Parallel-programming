/* hw3b initial version  */
#include <stdio.h>
#include <stdlib.h>

const int INF = ((1 << 30) - 1);
const int V = 50010;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);

__global__ void cal_kernel(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int* Dist_gpu, int pitch);

__global__ void cal_kernel_phase1(int B, int Round,  int n, int* Dist_gpu, int pitch) ;
int n, m;
int *Dist;
int *Dist_gpu;

int main(int argc, char* argv[]) {
    input(argv[1]);
    int B = 16;
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
    // cudaMalloc((void**)&Dist_gpu, sizeof(int)*n*n);
    // cudaMemcpy(Dist_gpu, Dist, sizeof(int)*n*n, cudaMemcpyHostToDevice);

    size_t pitch;

    cudaMallocPitch((void**)&Dist_gpu, &pitch ,n*sizeof(int), n);
    cudaMemcpy2D(Dist_gpu, pitch, Dist, n*sizeof(int), n*sizeof(int), n, cudaMemcpyHostToDevice);
    pitch = pitch / sizeof(int);
    
    //dim3 block(B, B);

    for (int r = 0; r < round; ++r) {
        printf("%d %d\n", r, round);
        fflush(stdout);
        /* Phase 1*/
        
         //cal_kernel_phase1<<<1, num_thread, B*B*sizeof(int)>>>(B, r, n, Dist_gpu, pitch);
        cal_kernel<<<1, num_thread>>>(B, r, r, r, 1, 1, n, Dist_gpu, pitch);
        /* Phase 2*/
        if (r > 0)
            cal_kernel<<<           r*1, num_thread>>>(B, r,      r,     0,             r,              1, n, Dist_gpu,pitch);
        cal_kernel<<< (round-r-1)*1, num_thread >>>(B, r,     r,  r +1,  round - r -1,             1, n, Dist_gpu,pitch);
        if (r > 0)
            cal_kernel<<<           1*r, num_thread >>>(B, r,     0,     r,             1,                     r, n, Dist_gpu,pitch);
        cal_kernel<<< 1*(round-r-1), num_thread >>>(B, r,  r +1,     r,             1,  round - r -1, n, Dist_gpu, pitch);

        /* Phase 3*/
        if (r > 0) {
            cal_kernel<<<                     r*r, num_thread >>>(B, r,     0,     0,            r,             r, n, Dist_gpu, pitch);
            cal_kernel<<<           (round-r-1)*r, num_thread >>>(B, r,     0,  r +1,  round -r -1,             r, n, Dist_gpu, pitch);
            cal_kernel<<<           r*(round-r-1), num_thread >>>(B, r,  r +1,     0,            r,  round - r -1, n, Dist_gpu, pitch);
        }
        cal_kernel<<< (round-r-1)*(round-r-1), num_thread >>>(B, r,  r +1,  r +1,  round -r -1,  round - r -1, n, Dist_gpu, pitch);
    }
    //cudaMemcpy(Dist, Dist_gpu, sizeof(int)*n*n, cudaMemcpyDeviceToHost);
    pitch = pitch * sizeof(int);
    cudaMemcpy2D(Dist, n*sizeof(int), Dist_gpu, pitch, n*sizeof(int), n, cudaMemcpyDeviceToHost);
}




// phase 1
__global__ void cal_kernel_phase1(int B, int Round,  int n, int* Dist_gpu, int pitch) {

        extern __shared__ int Shared_Dist[];
       
        // Get the internal position
        int i = Round * B + threadIdx.x  / B;
        int j = Round * B + threadIdx.x  % B;
        if (i>=n || j>=n) {
            i = n - 1;
            j = n - 1;
        }

        int si = i - B*Round;
        int sj = j - B*Round;
        //move local memory to shared memory
        Shared_Dist[(si*B+sj)] = Dist_gpu[i*pitch+j];
        __syncthreads();
        
        
        // printf("%d", Shared_Dist[i*B+j]);
        for (int k = 0; k < B && k < n; ++k) {
        
            //if ((i>=n) | (j>=n)) continue ;
            int temp = Shared_Dist[(si*B+k)] + Shared_Dist[(k*B+sj)];
            if (temp < Shared_Dist[(si*B+sj)]) {
                    Shared_Dist[si*B+sj] = temp;
            }
            
            __syncthreads();
        }
        Dist_gpu[i*pitch+j] =  Shared_Dist[si*B+sj];

}


//general kernel for each phase
__global__ void cal_kernel(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int* Dist_gpu, int pitch) {

        // from blockId get the block 2d position
        int b_i = block_start_x + blockIdx.x / block_width;
        int b_j = block_start_y + blockIdx.x % block_width;

        int i = b_i * B + threadIdx.x / B;
        int j = b_j * B + threadIdx.x % B;
        int temp;
        #pragma unroll
        for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
        
            // int i = b_i * B + threadIdx.x  / B;
            // int j = b_j * B + threadIdx.x  % B;
            
            
            if ((i>=n) | (j>=n)) continue ;
            
            temp = Dist_gpu[i*pitch+k] + Dist_gpu[k*pitch+j];
            if (temp < Dist_gpu[i*pitch+j]) {
                    Dist_gpu[i*pitch+j] = temp;
            }
            
            __syncthreads();
        }

}