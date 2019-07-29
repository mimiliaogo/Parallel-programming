/* hw3b initial version  */
#include <stdio.h>
#include <stdlib.h>

const int INF = ((1 << 30) - 1);
const int V = 50010;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);

// __global__ void cal_kernel( int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int* Dist_gpu, int pitch);

// __global__ void cal_kernel_phase1(int Round,  int n, int* Dist_gpu, int pitch) ;
// __global__ void cal_kernel_phase2( int Round,  int n, int* Dist_gpu, int pitch) ;
int n, m, bn;
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
    int B = 32;
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
__global__ 
void calPhase1(int B, int Round, int* Dist, int node, int pitch)
{
	extern __shared__ int sdata[];
	int x = threadIdx.y;//threadIdx.x;
    int y = threadIdx.x;//threadIdx.y;
	int sx = Round*B+x;
	int sy = Round*B+y;
	
	sdata[x*B+y]=Dist[sx*pitch+sy];
	__syncthreads();
	if(sx>=node||sy>=node)
		return;
	int tem;
	for (int k = 0; k < B ; ++k) 
	{		
		tem=sdata[x*B+k] + sdata[k*B+y];
		if (tem < sdata[x*B+y])
		{
			sdata[x*B+y] = tem;
		}	
		__syncthreads();
	}
	Dist[sx*pitch+sy]=sdata[x*B+y];
}
__global__ 
void calPhase2(int B, int Round, int* Dist, int node, int pitch)
{
	if(blockIdx.x==Round)
		return;
	extern __shared__ int sm[];
	int* p = &sm[B*B];
	
	int x = threadIdx.y;//threadIdx.x;
    int y = threadIdx.x;//threadIdx.y;
	
	unsigned int sx = Round*B+x;
	unsigned int sy = Round*B+y;	
	sm[x*B+y]=Dist[sx*pitch+sy];	
	
	unsigned int rx = blockIdx.x*B+x;
	unsigned int cy = blockIdx.x*B+y;
	unsigned int idx= (blockIdx.y == 1)?rx*pitch+sy:sx*pitch+cy;
	p[x*B+y]=Dist[idx];
	__syncthreads();
	
	
	int* a =(blockIdx.y == 0)?&sm[0]:p;
	int* b =(blockIdx.y == 1)?&sm[0]:p;
	int tem;
	for (int k = 0; k < B ; ++k) 
	{
		tem=a[x*B+k] + b[k*B+y];
		if ( tem < p[x*B+y])
		{
			p[x*B+y] = tem;
		}
		__syncthreads();
	}
	Dist[idx]=p[x*B+y];
	
}
__global__ 
void calPhase3(int B, int Round, int* Dist, int node, int pitch)
{
	int blockIdxx=blockIdx.y;//blockIdx.x;
	int blockIdxy=blockIdx.x;//blockIdx.y;
	if (blockIdxx == Round || blockIdxy == Round) 
		return;
	extern __shared__ int sm[];
	int* pr = &sm[0];
	int* pc = &sm[B*B];
	
	int x = threadIdx.y;//threadIdx.x;
    int y = threadIdx.x;//threadIdx.y;
	
	int rx = blockIdxx*blockDim.x+x;
	int ry = Round*B+y;
	
	int cx = Round*B+x;
	int cy = blockIdxy*blockDim.y+y;
	
	pr[x*B+y]=Dist[rx*pitch+ry];
	pc[x*B+y]=Dist[cx*pitch+cy];
	__syncthreads();
	
	if (rx >= node || cy >= node) 
		return;
	
	int tem;
	int ans=Dist[rx*pitch+cy] ;
	for (int k = 0; k < B ; ++k) {		
		tem=pr[x*B+k] + pc[k*B+y];
		if ( tem<ans){
			ans=tem;
		}
	}
	Dist[rx*pitch+cy] = ans;
	
}
void block_FW(int B) {
    int round = ceil(n, B);
    int num_thread = B * B;
    // cudaMalloc((void**)&Dist_gpu, sizeof(int)*n*n);
    // cudaMemcpy(Dist_gpu, Dist, sizeof(int)*n*n, cudaMemcpyHostToDevice);

    size_t pitch;

    cudaMallocPitch((void**)&Dist_gpu, &pitch ,bn*sizeof(int), bn);
    cudaMemcpy2D(Dist_gpu, pitch, Dist, bn*sizeof(int), bn*sizeof(int), bn, cudaMemcpyHostToDevice);
    pitch = pitch / sizeof(int);
    
    dim3 grid1(1, 1);
	dim3 grid2(round, 2);
    dim3 grid3(round, round);
	dim3 block(B, B);
	int sSize = B * B * sizeof(int);
	for (int r = 0; r < round; ++r) {
		calPhase1<<<grid1, block, sSize  >>>(B, r, Dist_gpu, n, pitch);
		calPhase2<<<grid2, block, sSize*2>>>(B, r, Dist_gpu, n, pitch);
		calPhase3<<<grid3, block, sSize*2>>>(B, r, Dist_gpu, n, pitch);
	}
    //cudaMemcpy(Dist, Dist_gpu, sizeof(int)*n*n, cudaMemcpyDeviceToHost);
    pitch = pitch * sizeof(int);
    cudaMemcpy2D(Dist, n*sizeof(int), Dist_gpu, pitch, n*sizeof(int), n, cudaMemcpyDeviceToHost);
}




