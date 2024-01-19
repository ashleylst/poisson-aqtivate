/*
 * Copyright (c) 2021, Dirk Pleiter, KTH
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "omp.h"
#include "mpi/mpi.h"
#include <stdlib.h>
#include "stdio.h"
#include <math.h>
#include "test.h"
#include "log.h"
#include "sys/time.h"

#ifndef NX
#define NX 16
#endif
#ifndef NY
#define NY 1024
#endif
#define NMAX 200000
#define EPS 1e-5

int solver(double *, double *, int, int, double, int);
int mpi_solver(double *v, double *f, int x, int y, double eps, int nmax, int block_num, MPI_Comm comm);

void print_arr(double * arr, int len);

void init(double * v, double * f, const int grid_size, MPI_Comm comm){
    // rank in grid
    int rank, size;
    int coords[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Determines process coords in cartesian topology given rank in group, coords store i,j coord in grid
    MPI_Cart_coords(comm, rank, 2, coords);
    // coords[i][j] = coords[0], coords[1]

    int block_x = NX / grid_size;
    int block_y = NY / grid_size;

#pragma omp parallel for default(none) shared(block_y, block_x, coords, f, v)
    for(int i = 0; i < block_y; i++)
    {
        for(int j = 0; j < block_x; j++)
        {
            v[block_x * i + j] = 0.0; //v[i][j] = 0
            int X = block_x * coords[1] + j;
            int Y = block_y * coords[0] + i;
            const double x = 2.0 * X / (NX - 1.0) - 1.0;
            const double y = 2.0 * Y / (NY - 1.0) - 1.0;
            // f = b
            f[block_x * i + j] = sin(x + y); // f[i][j] = sin(...)
        }
    }
    //printf("coord %d, %d: \n", coords[0], coords[1]);
    //print_arr(f, block_x * block_y);
}

int main(int argc, char *argv[])
{
    double *v;
    double *f;

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // get the dimensions
    int q = (int)sqrt(p);
    if (p != q*q || NX % q != 0 || NY % q != 0)
    {
        // exit if processes cannot be treated as grid
        log_error("processes cannot be treated as grid or the problem cannot be divided evenly");
        exit(1);
    }

    // Allocate memory, v, f local to processor
    v = (double *) malloc(NX/q * NY/q * sizeof(double));
    f = (double *) malloc(NX/q * NY/q * sizeof(double));

    // create 2D cartesian grid for the processors (enable reordering)
    MPI_Comm grid_comm;
    int dims[2] = {q, q};
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

    struct timeval begin, end;
    gettimeofday(&begin, 0);

    // initialize the grid
    init(&v[0], &f[0], q, grid_comm);

    /*
    // check initialization correctness
    int size;
    int coords[2];
    MPI_Comm_rank(grid_comm, &rank);
    MPI_Comm_size(grid_comm, &size);
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    if(!check_matrix_init(&v[0], &f[0], q, NX, NY, coords[0], coords[1]))
    {
        printf("failed init");
    }
    */

    mpi_solver(&v[0], &f[0], NX, NY, EPS, NMAX, q, grid_comm);

    // Stop measuring time and calculate the elapsed time
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds*1e-6;

    printf("Time measured: %.3f seconds.\n", elapsed);

    //distribute_vector(NY*NX, &v[0], &local_v, grid_comm);

    // Call solver
   // solver(v, f, NX, NY, EPS, NMAX);

    //for (int iy = 0; iy < NY; iy++)
    //    for (int ix = 0; ix < NX; ix++)
    //        printf("%d,%d,%e\n", ix, iy, v[iy*NX+ix]);

    // Clean-up
    free(v);
    free(f);
    MPI_Finalize();

    return 0;
}
