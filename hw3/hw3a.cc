#define PNG_NO_SETJMP
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <sched.h>
using namespace std;
const int INF = ((1 << 30) - 1);
const int V = 50010;
void input(char* inFileName);
void output(char* outFileName);

//void block_FW(int B);
//int ceil(int a, int b);
//void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m;
static int Dist[V][V];

int NUM_THREAD;

void Shortest_Path(int id, int k)
{
   
    for (int t = id; t < n; t += NUM_THREAD) {
        for (int j = 0; j < n; ++j) {
            if (Dist[t][k] + Dist[k][j] < Dist[t][j])  
            Dist[t][j] = Dist[t][k] + Dist[k][j]; 
        }
    }

}

int main(int argc, char* argv[]) {

    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    NUM_THREAD = CPU_COUNT(&cpuset);
    
    std::thread thread_arr[NUM_THREAD];
    //int id[NUM_THREAD];

    input(argv[1]);
    for (int k = 0; k < n; k++)  
    {  
        for (int d = 0; d < NUM_THREAD; d++) {
            thread_arr[d] =  thread(Shortest_Path, d, k);
        }
        for (int d=0; d < NUM_THREAD; d++) {
            thread_arr[d].join();
        }

    }  
    output(argv[2]);
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

