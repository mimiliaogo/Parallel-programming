#include <stdio.h>
#include <stdlib.h>
#define B 64 //int B = 16;
const int INF = ((1 << 30) - 1);
const int V = 50010;
void input(char* inFileName);
void output(char* outFileName);

void block_FW();
int ceil(int a, int b);

__global__ void cal_kernel( int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int* Dist_gpu, int pitch);

__global__ void cal_kernel_phase1(int Round,  int n, int* Dist_gpu, int pitch) ;
__global__ void cal_kernel_phase2( int Round,  int n, int* Dist_gpu, int pitch) ;
__global__ void cal_kernel_phase3(int Round,  int n, int* Dist_gpu, int pitch) ;

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
    
    dim3 block(32, 32);
    dim3 grid2(round-1, 2);
    dim3 grid3(round, round); //to avoid border problem
    for (int r = 0; r < round; ++r) {
        //printf("%d %d\n", r, round);
        //fflush(stdout);
        /* Phase 1*/
        
        cal_kernel_phase1<<<1, block, B*B*sizeof(int)>>>( r, n, Dist_gpu, pitch);
        //cal_kernel<<<1, num_thread>>>(B, r, r, r, 1, 1, n, Dist_gpu, pitch);
        /* Phase 2*/
        cal_kernel_phase2<<<grid2, block, B*B*2*sizeof(int)>>>(r, n, Dist_gpu, pitch);
        // if (r > 0)
        //     cal_kernel<<<           r*1, num_thread>>>(B, r,      r,     0,             r,              1, n, Dist_gpu,pitch);
        // cal_kernel<<< (round-r-1)*1, num_thread >>>(B, r,     r,  r +1,  round - r -1,             1, n, Dist_gpu,pitch);
        // if (r > 0)
        //     cal_kernel<<<           1*r, num_thread >>>(B, r,     0,     r,             1,                     r, n, Dist_gpu,pitch);
        // cal_kernel<<< 1*(round-r-1), num_thread >>>(B, r,  r +1,     r,             1,  round - r -1, n, Dist_gpu, pitch);

        /* Phase 3*/
        cal_kernel_phase3<<<grid3, block, B*B*2*sizeof(int)>>>(r, n, Dist_gpu, pitch);
        //calPhase3<<<grid3, block, B*B*2*sizeof(int)>>>(r, Dist_gpu, n, pitch);
       
        // if (r > 0) {
        //     cal_kernel<<<                     r*r, num_thread >>>( r,     0,     0,            r,             r, n, Dist_gpu, pitch);
        //     cal_kernel<<<           (round-r-1)*r, num_thread >>>( r,     0,  r +1,  round -r -1,             r, n, Dist_gpu, pitch);
        //     cal_kernel<<<           r*(round-r-1), num_thread >>>( r,  r +1,     0,            r,  round - r -1, n, Dist_gpu, pitch);
        // }
        // cal_kernel<<< (round-r-1)*(round-r-1), num_thread >>>( r,  r +1,  r +1,  round -r -1,  round - r -1, n, Dist_gpu, pitch);
    }
    //cudaMemcpy(Dist, Dist_gpu, sizeof(int)*n*n, cudaMemcpyDeviceToHost);
    pitch = pitch * sizeof(int);
    cudaMemcpy2D(Dist, n*sizeof(int), Dist_gpu, pitch, n*sizeof(int), n, cudaMemcpyDeviceToHost);
}




// phase 1
__global__ void cal_kernel_phase1(int Round,  int n, int* Dist_gpu, int pitch) {

        
        extern __shared__ int Shared_Dist[];
        int si = threadIdx.y*2;
        int sj = threadIdx.x*2;
        // Get the internal position (the left corner)
        int i = Round * B + threadIdx.y*2;
        int j = Round * B + threadIdx.x*2;
     
        //move local memory to shared memory
        Shared_Dist[(si*B+sj)] = Dist_gpu[i*pitch+j];
        Shared_Dist[(si+1)*B+sj] = Dist_gpu[(i+1)*pitch+j];
        Shared_Dist[(si*B+sj+1)] = Dist_gpu[i*pitch+j+1];
        Shared_Dist[(si+1)*B+sj+1] = Dist_gpu[(i+1)*pitch+j+1];
        __syncthreads();
        
        if (i>=n || j>=n) {
            return;
        }

        int temp;
        #pragma unroll 64
        for (int k = 0; k < B ; ++k) {
        
            //if ((i>=n) | (j>=n)) continue ;
            temp = Shared_Dist[(si*B+k)] + Shared_Dist[(k*B+sj)];
            if (temp < Shared_Dist[(si*B+sj)]) {
                    Shared_Dist[si*B+sj] = temp;
            }
            temp = Shared_Dist[(si+1)*B+k] + Shared_Dist[(k*B+sj)];
            if (temp < Shared_Dist[(si+1)*B+sj]) {
                    Shared_Dist[(si+1)*B+sj] = temp;
            }
            temp = Shared_Dist[(si*B+k)] + Shared_Dist[(k*B+sj+1)];
            if (temp < Shared_Dist[(si*B+sj+1)]) {
                    Shared_Dist[(si*B+sj+1)] = temp;
            }
            temp = Shared_Dist[(si+1)*B+k] + Shared_Dist[(k*B+sj+1)];
            if (temp < Shared_Dist[(si+1)*B+sj+1]) {
                    Shared_Dist[(si+1)*B+sj+1] = temp;
            }
            
            __syncthreads();
        }
        Dist_gpu[i*pitch+j] =  Shared_Dist[si*B+sj];
        Dist_gpu[i*pitch+j] = Shared_Dist[(si*B+sj)] ;
        Dist_gpu[(i+1)*pitch+j] = Shared_Dist[(si+1)*B+sj];
        Dist_gpu[i*pitch+j+1] = Shared_Dist[(si*B+sj+1)];
        Dist_gpu[(i+1)*pitch+j+1] = Shared_Dist[(si+1)*B+sj+1];
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
        int x = threadIdx.y*2;
        int y = threadIdx.x*2;

        //the position in Dist
        int i = block_start_x * B + x;
        int j = block_start_y * B + y;
        

        extern __shared__ int Shared_Dist[];
        int * Shared_Pivot = &Shared_Dist[B*B];

        // local to shared memory 
        // itself
        Shared_Dist[x*B+y] = Dist_gpu[i*pitch+j];
        Shared_Dist[(x+1)*B+y] = Dist_gpu[(i+1)*pitch+j];
        Shared_Dist[x*B+y+1] = Dist_gpu[i*pitch+j+1];
        Shared_Dist[(x+1)*B+y+1] = Dist_gpu[(i+1)*pitch+j+1];
        // pivot
        int px = Round*B + x;
        int py = Round*B + y;
        Shared_Pivot[x*B+y] = Dist_gpu[px*pitch + py];
        Shared_Pivot[(x+1)*B+y] = Dist_gpu[(px+1)*pitch + py];
        Shared_Pivot[x*B+y+1] = Dist_gpu[px*pitch + py+1];
        Shared_Pivot[(x+1)*B+y+1] = Dist_gpu[(px+1)*pitch + py+1];
        __syncthreads();
        
        
        int* a =(blockIdx.y == 0)?&Shared_Dist[0]:Shared_Pivot;
        int* b =(blockIdx.y == 1)?&Shared_Dist[0]:Shared_Pivot;

        int temp;
        #pragma unroll 64
        for (int k = 0; k < B ; ++k) 
        {
            
            temp =a[x*B+k] + b[k*B+y];
            if ( temp < Shared_Dist[x*B+y])
            {
                Shared_Dist[x*B+y] = temp;
            }
            temp =a[(x+1)*B+k] + b[k*B+y];
            if ( temp < Shared_Dist[(x+1)*B+y])
            {
                Shared_Dist[(x+1)*B+y] = temp;
            }
            temp =a[x*B+k] + b[k*B+y+1];
            if ( temp < Shared_Dist[x*B+y+1])
            {
                Shared_Dist[x*B+y+1] = temp;
            }
            temp =a[(x+1)*B+k] + b[k*B+y+1];
            if ( temp < Shared_Dist[(x+1)*B+y+1])
            {
                Shared_Dist[(x+1)*B+y+1] = temp;
            }
            __syncthreads();
        }
        Dist_gpu[i*pitch + j]=Shared_Dist[x*B+y];
        Dist_gpu[(i+1)*pitch + j]=Shared_Dist[(x+1)*B+y];
        Dist_gpu[i*pitch + j+1]=Shared_Dist[x*B+y+1];
        Dist_gpu[(i+1)*pitch + j+1]=Shared_Dist[(x+1)*B+y+1];


        
}

// phase 3
__global__ void cal_kernel_phase3(int Round,  int n, int* Dist_gpu, int pitch) {
        
        // grid (round, round) is to avoid border problem
        // This is the pivot and col/row
        if (blockIdx.x==Round || blockIdx.y== Round) return;

        // x, y position in grim    
        int gx = blockIdx.y;
        int gy = blockIdx.x;
        
        // sx, sy position in block
        int sx = threadIdx.y*2;
        int sy = threadIdx.x*2;

        // i, j position in Dist_gpu
        int i = gx*B + sx;
        int j = gy*B + sy;
        
        extern __shared__ int Shared_Dist[];
        int * a = &Shared_Dist[0]; // for row related block
        int * b = &Shared_Dist[B*B];// for column related block

        a[sx*B+sy] = Dist_gpu[i*pitch+(Round*B+sy)];
        a[(sx+1)*B+sy] = Dist_gpu[(i+1)*pitch+(Round*B+sy)];
        a[sx*B+sy+1] = Dist_gpu[i*pitch+(Round*B+sy+1)];
        a[(sx+1)*B+sy+1] = Dist_gpu[(i+1)*pitch+(Round*B+sy+1)];

        b[sx*B+sy] = Dist_gpu[(Round*B+sx)*pitch+j];
        b[(sx+1)*B+sy] = Dist_gpu[(Round*B+sx+1)*pitch+j];
        b[sx*B+sy+1] = Dist_gpu[(Round*B+sx)*pitch+j+1];
        b[(sx+1)*B+sy+1] = Dist_gpu[(Round*B+sx+1)*pitch+j+1];
        
        __syncthreads();
        if (i>=n || j>=n) return;

        int temp;
        int self_dist1 = Dist_gpu[i*pitch+j];
        int self_dist2 = Dist_gpu[(i+1)*pitch+j];
        int self_dist3 = Dist_gpu[i*pitch+j+1];
        int self_dist4 = Dist_gpu[(i+1)*pitch+j+1];
        #pragma unroll 64
        for (int k = 0; k < B ; ++k) {		
            temp=a[sx*B+k] + b[k*B+sy];
            if ( temp<self_dist1){
                self_dist1=temp;
            }
            temp=a[(sx+1)*B+k] + b[k*B+sy];
            if ( temp<self_dist2){
                self_dist2=temp;
            }
            temp=a[sx*B+k] + b[k*B+sy+1];
            if ( temp<self_dist3){
                self_dist3=temp;
            }
            temp=a[(sx+1)*B+k] + b[k*B+sy+1];
            if ( temp<self_dist4){
                self_dist4=temp;
            }
        }
        Dist_gpu[i*pitch+j] = self_dist1;
        Dist_gpu[(i+1)*pitch+j] = self_dist2;
        Dist_gpu[i*pitch+j+1] = self_dist3;
        Dist_gpu[(i+1)*pitch+j+1] = self_dist4;
            
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