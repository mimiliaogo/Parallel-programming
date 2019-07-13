//C style
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

struct ThreadArgunent {
    int a;
    int b;
};

void* threadRoutine(void* arg_) {
    struct ThreadArgunent* arg = arg_;
    int* c = malloc(sizeof(int));
    *c = arg->a + arg->b;
    pthread_exit(c);
}

int main() {
    pthread_t thread;
    struct ThreadArgunent arg;
    arg.a = 1;
    arg.b = 2;
    pthread_create(&thread, 0, threadRoutine, &arg);
    int* result;
    pthread_join(thread, (void**)&result);
    printf("result: %d\n", *result);
    free(result);
}