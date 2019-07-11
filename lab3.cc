#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int NUMTHREAD;
int slc;
void *Calpi(void* arg_)
{
    int *k = (int*)arg_;
    double *area =(double*) malloc(sizeof(float));
    *area = 0.0;
    for (double i = *k; i < slc; i += NUMTHREAD)
    {
        *area += (sqrt(1-(i/slc)*(i/slc)))/slc;
    }
    pthread_exit(area);
}

int main(int argc, char**argv)
{

    slc = atoi(argv[1]);
    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    NUMTHREAD = CPU_COUNT(&cpuset);
    pthread_t thread_ids[NUMTHREAD];
    int id[NUMTHREAD];
    for (int i=0; i<NUMTHREAD; i++) {
        id[i] = i;
    }
    double sum=0.0;
    for (int k = 0; k < NUMTHREAD; k++)
    {
        pthread_create(&thread_ids[k], 0, Calpi, &id[k]);
    }
    for (int k=0;k < NUMTHREAD; k++) {
        double* result;
        pthread_join(thread_ids[k], (void**)&result);
        sum+=*result;
    }
    sum = 4*sum;
    printf("%.6f", sum);
}