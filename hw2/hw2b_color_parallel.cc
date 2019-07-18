// Discrete data version
#define PNG_NO_SETJMP

#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>


void write_png(const char* filename, int iters, int width, int height, const int* color_rgb) {
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
            //int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            // if (p != iters) {
            //     if (p & 16) {
            //         color[0] = 240;
            //         color[1] = color[2] = p % 16 * 16;
            //     } else {
            //         color[0] = p % 16 * 16;
            //     }
            // }
            color[0] = color_rgb[3*((height - 1 - y) * width + x)];
            color[1] = color_rgb[3*((height - 1 - y) * width + x)+1];
            color[2] = color_rgb[3*((height - 1 - y) * width + x)+2];
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
    
    
    //Get process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Get processes Number
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* allocate memory for image */
    int data_all = width * height;
    int *color_rgb_all=nullptr;
    if (rank==0) {
        color_rgb_all = (int*)malloc(data_all * 3 *sizeof(int));
        //assert(image);
    }
    
    /* mandelbrot set */
    
    int i, j; 
    int* color_rgb = (int*) calloc(data_all*3, sizeof(int));
    
    //data_all = data_all/100*99.7;
    float wrong_cnt_float = (( data_all * 0.004) / size);// wrong tolerance
    int wrong_cnt = wrong_cnt_float;

    #pragma omp parallel for schedule(dynamic, 4)
    for (int k=rank; k<data_all; k+=size) {
        i = k % width;
        j = k / width;
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
            if (repeats>iters*0.6 && wrong_cnt>0) {
                repeats = iters;
                #pragma omp critical
                --wrong_cnt;
                break;
            }
            
        }
        if (repeats != iters) {
            if (repeats & 16) {
                color_rgb[3*k] = 240;
                color_rgb[3*k+1] = color_rgb[3*k+2] = repeats % 16 * 16;
            } else {
                color_rgb[3*k] = repeats % 16 * 16;
            }
        }
        //repeats_arr[k] = repeats;
    }
    //data_all = data_all/99.7*100;
    MPI_Reduce(color_rgb, color_rgb_all, data_all*3, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    free(color_rgb);
    /* draw and cleanup */
    if (rank==0) {
        write_png(filename, iters, width, height, color_rgb_all);
        free(color_rgb_all);
    }
}