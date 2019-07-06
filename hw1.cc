//MPI ODD-EVEN SORT
#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
using namespace std;
void Odd_Even_switch(float data_arr[], int local_size, int phase, int even_partner, int odd_parter, int rank, int size);
// qsort cmp function
int compare_float(const void *a, const void *b)
{
    const float *da = (const float *)a;
    const float *db = (const float *)b;
    return (*da > *db) - (*da < *db);
}
// Merge arr1[0..n1-1] and arr2[0..n2-1] into
// arr3[0..n1+n2-1]
void mergeArrays(float arr1[], float arr2[], int n1,
                 int n2, float arr3[])
{
    int i = 0.0, j = 0.0, k = 0.0;

    // Traverse both array
    while (i < n1 && j < n2)
    {
        // Check if current element of first
        // array is smaller than current element
        // of second array. If yes, store first
        // array element and increment first array
        // index. Otherwise do same with second array
        if (arr1[i] < arr2[j])
            arr3[k++] = arr1[i++];
        else
            arr3[k++] = arr2[j++];
    }

    // Store remaining elements of first array
    while (i < n1)
        arr3[k++] = arr1[i++];

    // Store remaining elements of second array
    while (j < n2)
        arr3[k++] = arr2[j++];
}
void ParallelSort(float data_arr[], int local_size, int rank, int size)
{
    int even_partner;
    int odd_partner;
    /* Find partners:  negative rank => do nothing during phase */
    if (rank % 2 != 0)
    { /* odd rank */
        even_partner = rank - 1;
        odd_partner = rank + 1;
        if (odd_partner == size)
            odd_partner = MPI_PROC_NULL; // Idle during odd phase
    }
    else
    { /* even rank */
        even_partner = rank + 1;
        if (even_partner == size)
            even_partner = MPI_PROC_NULL; // Idle during even phase
        odd_partner = rank - 1;
    }
    /* Find partner size */
    
    int phase;
    for (int phase = 0; phase < size + 1; phase++)
    {
        Odd_Even_switch(data_arr, local_size, phase, even_partner, odd_partner, rank, size);
    }
}

void Odd_Even_switch(float data_arr[], int local_size, int phase, int even_partner, int odd_partner, int rank, int size)
{
    if (phase % 2 == 0)
    { // even phase
        if (even_partner >= 0)
        {
            if (rank % 2 == 0)
            { // even rank send to odd rank
                // cout<<rank<<"send"<<'\n';
                MPI_Send(data_arr, local_size, MPI_FLOAT, even_partner, 99, MPI_COMM_WORLD);
                MPI_Recv(data_arr, local_size, MPI_FLOAT, even_partner, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                // cout<<rank<<"receive"<<'\n';
                // haven't add diff local_size
                //odd rank receive data from even rank and do merge sort
                float *rec_arr = (float *)malloc(sizeof(float) * local_size);
                MPI_Recv(rec_arr, local_size, MPI_FLOAT, even_partner, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                float *merge_arr = (float *)malloc(sizeof(float) * local_size * 2);
                mergeArrays(data_arr, rec_arr, local_size, local_size, merge_arr);
                for (int i = 0; i < local_size; i++)
                { // copy front smallest numbers
                    rec_arr[i] = merge_arr[i];
                }
                for (int i = local_size; i < 2 * local_size; i++)
                { //copy later bigger numbers to itself
                    data_arr[i - local_size] = merge_arr[i];
                }
                //send the small half number to even rank
                MPI_Send(rec_arr, local_size, MPI_FLOAT, even_partner, 99, MPI_COMM_WORLD);
            }
        }
    }
    else
    { // odd phase
        if (odd_partner >= 0)
        {
            if (rank % 2 != 0)
            { // odd rank send to even rank
                // cout<<rank<<"send"<<'\n';
                MPI_Send(data_arr, local_size, MPI_FLOAT, odd_partner, 99, MPI_COMM_WORLD);
                MPI_Recv(data_arr, local_size, MPI_FLOAT, odd_partner, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                // cout<<rank<<"receive"<<'\n';
                // haven't add diff local_size
                //odd rank receive data from even rank and do merge sort
                float *rec_arr = (float *)malloc(sizeof(float) * local_size);
                MPI_Recv(rec_arr, local_size, MPI_FLOAT, odd_partner, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                float *merge_arr = (float *)malloc(sizeof(float) * local_size * 2);
                mergeArrays(data_arr, rec_arr, local_size, local_size, merge_arr);
                for (int i = 0; i < local_size; i++)
                { // copy front smallest numbers
                    rec_arr[i] = merge_arr[i];
                }
                for (int i = local_size; i < 2 * local_size; i++)
                { //copy later bigger numbers to itself
                    data_arr[i - local_size] = merge_arr[i];
                }
                //send the small half number to even rank
                MPI_Send(rec_arr, local_size, MPI_FLOAT, odd_partner, 99, MPI_COMM_WORLD);
            }
        }
    }
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    int local_size;
    //get the array size
    int array_size = atoi(argv[1]);
    //Get process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Get processes Number
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //get local array size
    if (rank < array_size % size)
    {
        local_size = array_size / size + 1;
    }
    else
    {
        local_size = array_size / size;
    }
    //get the max add data size
    int max_add = array_size % size - 1;
    //open input file
    MPI_File f;
    MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f);

    //distribute numbers to each process
    float *data_arr = (float *)malloc(sizeof(float) * local_size); // each process has an array

    //decide offset
    int offset;
    if (rank <= max_add)
        offset = sizeof(float) * rank * local_size;
    else
    {
        offset = sizeof(float) * (max_add + 1) * (local_size + 1);
        offset += sizeof(float) * (rank - max_add - 1) * local_size;
    }

    MPI_File_read_at(f, offset, data_arr, local_size, MPI_FLOAT, MPI_STATUS_IGNORE);


    if (size >= array_size) size = array_size;
    if (rank < array_size)
    {

        //and then sort it
        qsort(data_arr, local_size, sizeof(float), compare_float);

       // print each process array
        // cout<<rank<<": ";
        // for (int i=0; i<local_size; i++) {
        //     cout<<data_arr[i]<<' ';
        // }
        // cout<<'\n';

        //start  parallel sorting
        ParallelSort(data_arr, local_size, rank, size);
        //print each process array

        // cout<<rank<<": ";
        // for (int i=0; i<local_size; i++) {
        //     cout<<data_arr[i]<<' ';
        // }
        // cout<<'\n';
    }
    MPI_File output;
    MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output);
    MPI_File_write_at(output, offset, data_arr, local_size, MPI_FLOAT, MPI_STATUS_IGNORE);

    MPI_Finalize();
}