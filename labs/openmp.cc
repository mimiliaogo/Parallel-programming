#include <omp.h>
#include <stdio.h>
int main()
{
    int i, n, chunk, a[100], b[100], result;
    result=0; chunk=2; n=10;//10 cores

    for (int i=0; i<n; i++) a[i] = b[i] = 1;

    #pragma omp parallel for default(shared) private(i) schedule(static, chunk) reduction(+:result)
    {
        for (i=0; i<n; i++) result = result + (a[i]*b[i]);
    }
    printf("%f", result);
}