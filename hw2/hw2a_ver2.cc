//pthread version

#define PNG_NO_SETJMP
#include <pthread.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
using namespace std;
/* Global variables for all threads */
int NUM_THREAD;
int iters;
double left, right, lower, upper;
int width, height;

int *image;
int data_all;
int max_add;
/* mandelbrot set */
// given the id of threads then distribute work to it
//return repeats
void* MandelbrotSet(void *arg_)
{

    int *k = (int*)arg_;
    int data_num, offset;
    if (*k < data_all%NUM_THREAD) data_num = data_all / NUM_THREAD + 1;
    else data_num = data_all / NUM_THREAD;

    if (*k <= max_add) offset = *k * data_num;
    else
    {
        offset =  (max_add + 1) * (data_num + 1);
        offset += (*k - max_add - 1) * data_num;
    }


    int data_1D;
    int i, j;//data_1D 對應到的i, j是多少
    //做data_num件事
    for (int k=0; k<data_num; k++) {
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
        image[offset+k] = repeats;
    }
    pthread_exit(NULL);
}
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
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    /* get CPU numbers */
    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    NUM_THREAD = CPU_COUNT(&cpuset);
    /* create threads */
    pthread_t thread_arr[NUM_THREAD];
    int id[NUM_THREAD];
    
    data_all = height * width;// the numbers of whole data 

    max_add = data_all % NUM_THREAD -1; 
   
    for (int k = 0; k < NUM_THREAD; k++)
    {
        
        //thread_arr[k] = thread(MandelbrotSet, id[k], data_num, offset, image+offset);
        id[k] = k;
        pthread_create(&thread_arr[k], 0, MandelbrotSet, &id[k]);
    }
    
    for (int k=0; k < NUM_THREAD; k++) {
        //thread_arr[k].join();
        pthread_join(thread_arr[k], NULL);//without return value
    }
    

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}