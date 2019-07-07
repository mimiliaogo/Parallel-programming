//MPI ODD-EVEN SORT
#include <mpi.h>
#include <array>
#include <algorithm>
using namespace std;

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
        
        sort(data_arr, data_arr + local_size);
        // qsort(data_arr, local_size, sizeof(float), compare_float);       
        //ParallelSort(data_arr, local_size, rank, size, max_add);
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
        int even_size;
        int odd_size;
        if ((rank > max_add && even_partner > max_add) || (rank <= max_add && even_partner <= max_add)) {
            even_size = local_size;
        } else if (rank > max_add && even_partner <= max_add ) {
            even_size = local_size + 1;
        } else if (rank <= max_add && even_partner > max_add ) {
            even_size = local_size - 1;
        }
        if ((rank > max_add && odd_partner > max_add) || (rank <= max_add && odd_partner <= max_add)) {
            odd_size = local_size;
        } else if (rank > max_add && odd_partner <= max_add ) {
            odd_size = local_size + 1;
        } else if (rank <= max_add && odd_partner > max_add ) {
            odd_size = local_size - 1;
        }


        //space PRE malloc
        float *rec_arr_even = (float *)malloc(sizeof(float) * even_size);
        float *merge_arr_even = (float *)malloc(sizeof(float) * (local_size+even_size));
        float *rec_arr_odd = (float *)malloc(sizeof(float) * odd_size);
        float *merge_arr_odd = (float *)malloc(sizeof(float) * (local_size+odd_size));
        int phase;
        for (int phase = 0; phase < size + 1; phase++)
        {
            //Odd_Even_switch(data_arr, local_size, phase, even_partner, odd_partner, rank, size, even_size, odd_size);
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
                        //odd rank receive data from even rank and do merge sort
                        //float *rec_arr = (float *)malloc(sizeof(float) * even_size);
                        MPI_Recv(rec_arr_even, even_size, MPI_FLOAT, even_partner, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        // float *merge_arr = (float *)malloc(sizeof(float) * (local_size+even_size));
                        mergeArrays(data_arr, rec_arr_even, local_size, even_size, merge_arr_even);
                        MPI_Send(merge_arr_even, even_size, MPI_FLOAT, even_partner, 99, MPI_COMM_WORLD);
                        for (int i = even_size; i < (even_size + local_size); i++)
                        { //copy later bigger numbers to itself
                            data_arr[i - even_size] = merge_arr_even[i];
                        }
                        
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
                        //odd rank receive data from even rank and do merge sort
                        //float *rec_arr = (float *)malloc(sizeof(float) * odd_size);
                        MPI_Recv(rec_arr_odd, odd_size, MPI_FLOAT, odd_partner, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        //float *merge_arr = (float *)malloc(sizeof(float) * (local_size + odd_size));
                        mergeArrays(data_arr, rec_arr_odd, local_size, odd_size, merge_arr_odd);
                        //send the small half number to even rank
                        MPI_Send(merge_arr_odd, odd_size, MPI_FLOAT, odd_partner, 99, MPI_COMM_WORLD);
                        for (int i = odd_size; i < (odd_size + local_size); i++)
                        { //copy later bigger numbers to itself
                            data_arr[i - odd_size] = merge_arr_odd[i];
                        }
                        
                    }
                }
            }
        }
       
    }
    MPI_File output;
    MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output);
    MPI_File_write_at(output, offset, data_arr, local_size, MPI_FLOAT, MPI_STATUS_IGNORE);

    MPI_Finalize();
}