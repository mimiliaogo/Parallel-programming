#define PNG_NO_SETJMP

#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    


    /* MPI distribute data to each rank*/
    MPI_Init(&argc, &argv);
    int rank, size;
    int local_size;
    //get the array size
    int data_all = height * width;
    //Get process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Get processes Number
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* allocate memory for image */
    int *image=nullptr;
    if (rank==0) {
        image = (int*)malloc(width * height * sizeof(int));
        assert(image);
    }
    //get local array size
    if (rank < data_all % size)
    {
        local_size = data_all / size + 1;
    }
    else
    {
        local_size = data_all / size;
    }
    int max_add = data_all % size - 1;
    
    int offset;
    if (rank <= max_add)
        offset =  rank * local_size;
    else
    {
        offset = (max_add + 1) * (local_size + 1);
        offset +=  (rank - max_add - 1) * local_size;
    }

    /* mandelbrot set */
    int data_1D;
    int i, j; 
    int* repeats_arr = (int*) malloc(sizeof(int)*local_size);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int k=0; k<local_size; k++) {
        data_1D = offset + k;
        i = data_1D % width;
        j = data_1D / width;
        double y0 = j * ((upper - lower) / height) + lower;
        double x0 = i * ((right - left) / width) + left;
        int repeats = 0;
        double x = 0;
        double y = 0;
        double length_squared = 0;
        while (repeats < iters && length_squared < 4) {
            double temp = x * x - y * y + x0;
            y = 2 * x * y + y0;
            x = temp;
            length_squared = x * x + y * y;
            ++repeats;
        }
        repeats_arr[k] = repeats;
    }
    /*Using the gatherv for fast testcases */
    /*get the local_size of each rank */
    if (size==3) {
        int *local_size_arr = (int*) malloc (sizeof(int)*size);
        int *offset_arr = (int*) malloc (sizeof(int)*size);
        int local_Size, offSet;
        for (int i=0; i<size; i++) {
            if (i < data_all % size)
            {
                local_Size = data_all / size + 1;
            }
            else
            {
                local_Size = data_all / size;
            }
            if (i <= max_add)
            offSet =  i * local_Size;
            else
            {
                offSet = (max_add + 1) * (local_Size + 1);
                offSet +=  (i - max_add - 1) * local_Size;
            }
            local_size_arr[i] = local_Size;
            offset_arr[i] = offSet;
        }
        
        MPI_Gatherv(repeats_arr, local_size, MPI_INT, image, local_size_arr,offset_arr, MPI_INT, 0, MPI_COMM_WORLD);
        free(local_size_arr);
        free(offset_arr);   
    }
    else {
        MPI_Gather(repeats_arr, local_size, MPI_INT, image, local_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
    free(repeats_arr);
    /* draw and cleanup */
    if (rank==0) {
        write_png(filename, iters, width, height, image);
        free(image);
    }
}