//
// Created by shiting on 2023-12-22.
//

#include <stdlib.h>
#include "stdio.h"
#include "mpi/mpi.h"
#include <math.h>
#include <stdbool.h>
#include "array_helpers.h"

void initialize_matrix(double * v, double * f, int NX, int NY) {
    int cnt = 0;
    // Initialise input B
//#pragma omp parallel for default(none) shared(v, f)
    for (int iy = 0; iy < NY; iy++)
    {
        for (int ix = 0; ix < NX; ix++)
        {
            // x = x0
            v[NX * iy + ix] = 0.0;

            const double x = 2.0 * ix / (NX - 1.0) - 1.0;
            const double y = 2.0 * iy / (NY - 1.0) - 1.0;
            // f = b
            f[NX * iy + ix] = sin(x + y);
        }
    }
}

bool check_matrix_init(double *local_v, double *local_f, int q, int NX, int NY, int I, int J){
    int nx = NX / q;
    int ny = NY / q;
    int coords[2];

    double *v = (double *) malloc(NX * NY * sizeof(double));
    double *f = (double *) malloc(NX * NY * sizeof(double));

    initialize_matrix(&v[0], &f[0], NX, NY);

    for(int iy = 0; iy < ny; iy ++)
    {
        for(int ix = 0; ix < nx; ix++)
        {
            get_big_matrix_coords(ix, iy, I, J, nx, ny, &coords[0]);
            if(local_v[nx * iy + ix] != v[NX * coords[1] + coords[0]] ||
               local_f[nx * iy + ix] != f[NX * coords[1] + coords[0]])
            {
                free(v);
                free(f);
                return false;
            }
        }
    }

    //printf("coord %d, %d: nx, ny: %d, %d \n", I, J, nx, ny);
    //print_arr(local_f, nx * ny);

    free(v);
    free(f);
    return true;
}

void update_single_matrix(double *v, int nx, int ny, double *w){
    for (int ix = 1; ix < (nx-1); ix++)
    {
        v[nx*0      + ix] = v[nx*(ny-2) + ix];
        v[nx*(ny-1) + ix] = v[nx*1      + ix];
        *w += fabs(v[nx*0+ix]) + fabs(v[nx*(ny-1)+ix]);
    }
    for (int iy = 1; iy < (ny-1); iy++)
    {
        v[nx*iy + 0]      = v[nx*iy + (nx-2)];
        v[nx*iy + (nx-1)] = v[nx*iy + 1     ];
        *w += fabs(v[nx*iy+0]) + fabs(v[nx*iy+(nx-1)]);
    }
}

bool test_sum(const double *arr, double sum, int len)
{
    double tmp = 0.0;
    for(int i = 0; i < len; i++)
    {
        tmp += fabs(arr[i]);
    }
    if(sum == tmp)
    {
        return true;
    }
    printf("%f %f\n", sum, tmp);
    return false;
}