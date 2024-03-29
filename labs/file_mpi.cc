#include <stdio.h>
#include <mpi.h>
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_File f;
    MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
    float data;
    // offset: the order they eat data (bytes)
    MPI_File_read_at(f, sizeof(float)*rank, &data, 1, MPI_FLOAT, MPI_STATUS_IGNORE);
    printf("rank %d got float %f\n", rank, data);
    MPI_Finalize();    
}