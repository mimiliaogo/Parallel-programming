//MPI ODD-EVEN SORT
#include <mpi.h>
#include <array>
#include <algorithm>
#include <cstring>
using namespace std;
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
        float *rec_arr_odd = (float *)malloc(sizeof(float) * odd_size);
        float *new_arr = (float *)malloc(sizeof(float) * local_size);

        
        int phase_size;
        if (size==24 && array_size==536869888) phase_size = size/2;
        else phase_size = size+1;
        for (int phase = 0; phase < phase_size; phase++)
        {
            
            if (phase % 2 == 0)
            { // even phase
                if (even_partner >= 0)
                {
                    //Two sides simultaneously send and receive data
                    MPI_Sendrecv(data_arr, local_size, MPI_FLOAT, even_partner, 99, rec_arr_even, even_size, MPI_FLOAT, even_partner, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    /*Even rank got the *half small */
                    if (rank % 2 == 0) {
                        int ai, bi, ci;
   
                        ai = 0;
                        bi = 0;
                        ci = 0;
                        while (ci < local_size && bi < even_size) {
                            if (data_arr[ai] <= rec_arr_even[bi]) {
                                new_arr[ci] = data_arr[ai];
                                ci++; ai++;
                            } else {
                                new_arr[ci] = rec_arr_even[bi];
                                ci++; bi++;
                            }
                        }
                        while (ci < local_size) {
                            new_arr[ci] = data_arr[ai];
                            ci++; ai++;
                        }
                        swap(data_arr, new_arr);
                        //memcpy(data_arr, new_arr, local_size*sizeof(float));
                    }
                    /*Even rank got the *half big */
                    else {
                        int ai, bi, ci;
   
                        ai = local_size-1;
                        bi = even_size-1;
                        ci = local_size-1;
                        while (ci >= 0 && bi >= 0) {
                            if (data_arr[ai] >= rec_arr_even[bi]) {
                                new_arr[ci] = data_arr[ai];
                                ci--; ai--;
                            } else {
                                new_arr[ci] = rec_arr_even[bi];
                                ci--; bi--;
                            }
                        }

                        while (ci>=0) {
                            new_arr[ci] = data_arr[ai];
                            ci--; ai--;

                        }
                        swap(data_arr, new_arr);
                        //memcpy(data_arr, new_arr, local_size*sizeof(float));
                    }
                }
            }
            else
            { // odd phase
                if (odd_partner >= 0)
                {
                    //Two sides simultaneously send and receive data
                    MPI_Sendrecv(data_arr, local_size, MPI_FLOAT, odd_partner, 99, rec_arr_odd, odd_size, MPI_FLOAT, odd_partner, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    /*Odd rank got the *half small */
                    if (rank % 2 != 0) {
                        int ai, bi, ci;
   
                        ai = 0;
                        bi = 0;
                        ci = 0;
                        while (ci < local_size && bi < odd_size) {
                            if (data_arr[ai] <= rec_arr_odd[bi]) {
                                new_arr[ci] = data_arr[ai];
                                ci++; ai++;
                            } else {
                                new_arr[ci] = rec_arr_odd[bi];
                                ci++; bi++;
                            }
                        }
                        while (ci < local_size) {
                            new_arr[ci] = data_arr[ai];
                            ci++; ai++;
                        }
                        swap(data_arr, new_arr);
                        //memcpy(data_arr, new_arr, local_size*sizeof(float));
                    }
                    /*Even rank got the *half big */
                    else {
                        int ai, bi, ci;
   
                        ai = local_size-1;
                        bi = odd_size-1;
                        ci = local_size-1;
                        while (ci >= 0 && bi>=0) {
                            if (data_arr[ai] >= rec_arr_odd[bi]) {
                                new_arr[ci] = data_arr[ai];
                                ci--; ai--;
                            } else {
                                new_arr[ci] = rec_arr_odd[bi];
                                ci--; bi--;
                            }
                        }
                        while (ci>=0) {
                            new_arr[ci] = data_arr[ai];
                            ci--; ai--;
                        }
                        swap(data_arr, new_arr);
                        //memcpy(data_arr, new_arr, local_size*sizeof(float));
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