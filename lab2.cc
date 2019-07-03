#include <mpi.h>
#include <math.h>
int main(int argc,char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    int slc = atoi(argv[1]);
    float result=0.0, sum;
    float pi = 0.0 ;
    //Get process ID
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    //Get processes Number
    MPI_Comm_size (MPI_COMM_WORLD, &size);



    for (int i=rank; i<n; i+=size) {
        result += (sqrt(1-(rank/slc)*(rank/slc)))/slc;
    }
    

    MPI_Reduce(&result, &sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank==0) {
        pi =( 4 * sum);
        printf("%.6f", pi);
    }
    MPI_Finalize();

    return 0;
}