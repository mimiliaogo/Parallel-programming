/* hw3b initial version  */
#include <stdio.h>
#include <stdlib.h>
#define B 32 //int B = 16;
const int INF = ((1 << 30) - 1);
const int V = 50010;
void input(char* inFileName);
void output(char* outFileName);

void block_FW();
int ceil(int a, int b);

__global__ void cal_kernel( int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int* Dist_gpu, int pitch);

__global__ void cal_kernel_phase1(int Round,  int n, int* Dist_gpu, int pitch) ;
__global__ void cal_kernel_phase2( int Round,  int n, int* Dist_gpu, int pitch) ;
int n, m, bn;
int *Dist;
int *Dist_gpu;

int main(int argc, char* argv[]) {
    input(argv[1]);
    //int B = 16;
    block_FW();
    output(argv[2]);
    cudaFreeHost(Dist);
    cudaFree(Dist_gpu);
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    int round = (n + B - 1) / B; 
    bn = B*round; // border n
    // To avoid the border problem
    cudaMallocHost(&Dist, sizeof(int)*bn*bn);
    for (int i = 0; i < bn; ++i) {
        for (int j = 0; j < bn; ++j) {
            if (i == j) {
                Dist[i*bn + j] = 0;
            } else {
                Dist[i*bn + j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]*bn + pair[1]] = pair[2];
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

void block_FW() {
    int round = ceil(n, B);
    int num_thread = B * B;
    // cudaMalloc((void**)&Dist_gpu, sizeof(int)*n*n);
    // cudaMemcpy(Dist_gpu, Dist, sizeof(int)*n*n, cudaMemcpyHostToDevice);

    size_t pitch;

    cudaMallocPitch((void**)&Dist_gpu, &pitch ,bn*sizeof(int), bn);
    cudaMemcpy2D(Dist_gpu, pitch, Dist, bn*sizeof(int), bn*sizeof(int), bn, cudaMemcpyHostToDevice);
    pitch = pitch / sizeof(int);
    
    dim3 block(B, B);
    dim3 grid2(round-1, 2);
    for (int r = 0; r < round; ++r) {
        printf("%d %d\n", r, round);
        fflush(stdout);
        /* Phase 1*/
        
        cal_kernel_phase1<<<1, block, B*B*sizeof(int)>>>( r, n, Dist_gpu, pitch);
        //cal_kernel<<<1, num_thread>>>(B, r, r, r, 1, 1, n, Dist_gpu, pitch);
        /* Phase 2*/
        cal_kernel_phase2<<<grid2, block>>>(r, n, Dist_gpu, pitch);
        // if (r > 0)
        //     cal_kernel<<<           r*1, num_thread>>>(B, r,      r,     0,             r,              1, n, Dist_gpu,pitch);
        // cal_kernel<<< (round-r-1)*1, num_thread >>>(B, r,     r,  r +1,  round - r -1,             1, n, Dist_gpu,pitch);
        // if (r > 0)
        //     cal_kernel<<<           1*r, num_thread >>>(B, r,     0,     r,             1,                     r, n, Dist_gpu,pitch);
        // cal_kernel<<< 1*(round-r-1), num_thread >>>(B, r,  r +1,     r,             1,  round - r -1, n, Dist_gpu, pitch);

        /* Phase 3*/
        if (r > 0) {
            cal_kernel<<<                     r*r, num_thread >>>( r,     0,     0,            r,             r, n, Dist_gpu, pitch);
            cal_kernel<<<           (round-r-1)*r, num_thread >>>( r,     0,  r +1,  round -r -1,             r, n, Dist_gpu, pitch);
            cal_kernel<<<           r*(round-r-1), num_thread >>>( r,  r +1,     0,            r,  round - r -1, n, Dist_gpu, pitch);
        }
        cal_kernel<<< (round-r-1)*(round-r-1), num_thread >>>( r,  r +1,  r +1,  round -r -1,  round - r -1, n, Dist_gpu, pitch);
    }
    //cudaMemcpy(Dist, Dist_gpu, sizeof(int)*n*n, cudaMemcpyDeviceToHost);
    pitch = pitch * sizeof(int);
    cudaMemcpy2D(Dist, n*sizeof(int), Dist_gpu, pitch, n*sizeof(int), n, cudaMemcpyDeviceToHost);
}




// phase 1
__global__ void cal_kernel_phase1(int Round,  int n, int* Dist_gpu, int pitch) {

        extern __shared__ int Shared_Dist[];
        int si = threadIdx.y;
        int sj = threadIdx.x;
        // Get the internal position
        int i = Round * B + threadIdx.y;
        int j = Round * B + threadIdx.x;
       
        // int si = i - B*Round;
        // int sj = j - B*Round;
        //move local memory to shared memory
        Shared_Dist[(si*B+sj)] = Dist_gpu[i*pitch+j];
        __syncthreads();
        
        if (i>=n || j>=n) {
            return;
        }

        int temp;
        // printf("%d", Shared_Dist[i*B+j]);
        for (int k = 0; k < B ; ++k) {
        
            //if ((i>=n) | (j>=n)) continue ;
            temp = Shared_Dist[(si*B+k)] + Shared_Dist[(k*B+sj)];
            if (temp < Shared_Dist[(si*B+sj)]) {
                    Shared_Dist[si*B+sj] = temp;
            }
            
            __syncthreads();
        }
        Dist_gpu[i*pitch+j] =  Shared_Dist[si*B+sj];

}

// phase 2
__global__ void cal_kernel_phase2(int Round,  int n, int* Dist_gpu, int pitch) {
        
        // the position of the block i'm in 
        int block_start_x, block_start_y;
        
        // for row
        block_start_y = Round; 
        if (blockIdx.x < Round*1) { // row left
            /* Start at (r, 0) */
            block_start_x = blockIdx.x;
            
        } else { // row right
            /* Start at (r, r+1) */
            block_start_x = Round + 1 + blockIdx.x - Round;
        }

        //for col
        if (blockIdx.y == 1) {
            int temp = block_start_x;
            block_start_x = block_start_y;
            block_start_y = temp; 
        }

        // the position in block
        //int x = threadIdx.y;
        //int y = threadIdx.x;

        //the position in Dist
        int i = block_start_x * B + threadIdx.x;
        int j = block_start_y * B + threadIdx.y;
        

        //extern __shared__ int Shared_Dist[];

        // global to shared memory 
        // itself
        //Shared_Dist[] = Dist_gpu[];
        // pivot


        // if (i>=n || j >= n) return;
        int temp;
        for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
            
            if ((i>=n) | (j>=n)) continue ;
            temp = Dist_gpu[i*pitch+k] + Dist_gpu[k*pitch+j];
            if (temp < Dist_gpu[i*pitch+j]) {
                    Dist_gpu[i*pitch+j] = temp;
            }
            
            __syncthreads();
        }


        
}


//general kernel for each phase
__global__ void cal_kernel(int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int* Dist_gpu, int pitch) {

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