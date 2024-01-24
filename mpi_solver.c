#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi/mpi.h"
#include "array_helpers.h"
#include "log.h"
#include "test.h"
#include "omp.h"

void evaluate_e(double *e, const double k1, const double k){
    double d = fabs(k1 - k);
    *e = (d > *e) ? d : *e;
}

void prepare_halo_buf(const double *v, double* sendbuf, int nx, int ny){
    //print_arr(v, nx*ny);
    for(int i = 0; i < nx; i++)
    {
        sendbuf[i] = v[i]; // top v[0][i]
        sendbuf[nx + i] = v[nx * (ny - 1) + i]; //bottom v[ny-1][i]
    }
    for(int i = 0; i < ny; i++)
    {
        sendbuf[2 * nx + i] = v[i * nx]; // left v[i][0]
        sendbuf[2 * nx + ny + i] = v[i * nx + nx - 1]; //right v[i][nx-1]
    }
    //print_arr(sendbuf, 2 * (nx+ny));
}

void prepare_column(const double *arr, double *buf, int nx, int ny, int col){
    for(int i = 0; i < ny; i++)
    {
        buf[i] = arr[i * nx + col];
    }
}

void update_v(double *vp, const double *f, const double *v, const double* recvbuf, \
              int ix, int iy, int nx, int ny){

    vp[iy * nx + ix] = -0.25 * (f[iy * nx + ix] - get_neighbours(ix, iy, nx, ny, &v[0], &recvbuf[0]));

}

void update_boundary(int block_num, int nx, int ny, const int *coords,
                     double *v, double *local_w, MPI_Comm comm){
    MPI_Status status_send[block_num * 4];

    double *from_bottom = (double *) malloc(nx * sizeof(double ));
    double *from_top = (double *) malloc(nx * sizeof(double ));
    double *from_left = (double *) malloc(ny * sizeof(double ));
    double *from_right = (double *) malloc(ny * sizeof(double ));
    int sendrecv_rank;
    int sendrecv_coords[2];

    if(block_num == 1)
    {
        update_single_matrix(&v[0], nx, ny, local_w);
        return;
    }

    for(int i = 0; i < block_num; i++)
    {
        // update first row of the big matrix, v[0][i] = v[y-2][i]
        if(coords[0] == 0 && coords[1] == i)
        {
            sendrecv_coords[0] = block_num - 1;
            sendrecv_coords[1] = i;//{block_num - 1, i};
            MPI_Cart_rank(comm, sendrecv_coords, &sendrecv_rank);

            //printf("coord %d, %d send to %d, %d\n", coords[0], coords[1], sendrecv_coords[0], sendrecv_coords[1]);
            //coord (0,i) send the second row to coord(block_num-1, i)
            MPI_Sendrecv(&v[nx], nx, MPI_DOUBLE, sendrecv_rank, 0,
                         from_bottom, nx, MPI_DOUBLE, sendrecv_rank, 0,
                         comm, &status_send[i]);

            substitute_part_arr(&v[0], from_bottom, nx, ny,0, ROW);
        }

        // update last row of the big matrix, v[y-1][i] = v[1][i]
        if(coords[0] == block_num - 1 && coords[1] == i)
        {
            sendrecv_coords[0] = 0;
            sendrecv_coords[1] = i; //{0, i};
            MPI_Cart_rank(comm, sendrecv_coords, &sendrecv_rank);

            //printf("coord %d, %d send to %d, %d\n", coords[0], coords[1], sendrecv_coords[0], sendrecv_coords[1]);
            //coord (block-1,i) send the second last row to coord(0, i)
            MPI_Sendrecv(&v[(ny - 2) * nx], nx, MPI_DOUBLE, sendrecv_rank, 0,
                         from_top, nx, MPI_DOUBLE, sendrecv_rank, 0,
                         comm, &status_send[block_num + i]);

            substitute_part_arr(&v[0], from_top, nx, ny, ny - 1, ROW);
        }
    }

    double *col_bufr = (double *) malloc(ny * sizeof(double));
    double *col_buf = (double *) malloc(ny * sizeof(double));

    for(int i = 0; i < block_num; i++)
    {
        //update left column of the big matrix, v[i][0] = v[i][x-2]
        if(coords[0] == i && coords[1] == 0)
        {
            sendrecv_coords[0] = i;
            sendrecv_coords[1] = block_num - 1;
            MPI_Cart_rank(comm, sendrecv_coords, &sendrecv_rank);

            //coord (i,0) send the second column to coord (i,block_num-1)
            prepare_column(&v[0], &col_buf[0], nx, ny, 1);

            MPI_Sendrecv(&col_buf[0], ny, MPI_DOUBLE, sendrecv_rank, 0,
                         from_right, ny, MPI_DOUBLE, sendrecv_rank, 0,
                         comm, &status_send[2*block_num + i]);

            substitute_part_arr(&v[0], from_right, nx, ny, 0, COLUMN);
        }

        //update right column of the big matrix, v[i][x-1] = v[i][1]
        if(coords[0] == i && coords[1] == block_num - 1)
        {
            sendrecv_coords[0] = i;
            sendrecv_coords[1] = 0;
            MPI_Cart_rank(comm, sendrecv_coords, &sendrecv_rank);

            // coord (i, block_num-1) send the second last column to coord (i, 0)
            prepare_column(&v[0], &col_bufr[0], nx, ny, nx - 2);

            MPI_Sendrecv(&col_bufr[0], ny, MPI_DOUBLE, sendrecv_rank, 0,
                         from_left, ny, MPI_DOUBLE, sendrecv_rank, 0,
                         comm, &status_send[3*block_num + i]);

            substitute_part_arr(&v[0], from_left, nx, ny, nx - 1, COLUMN);
        }
    }

    add_zeros(&coords[0], block_num, &v[0], nx, ny);

    get_w(&v[0], local_w, nx * ny);

    free(col_buf);
    free(col_bufr);
    free(from_top);
    free(from_bottom);
    free(from_left);
    free(from_right);
}


int mpi_solver(double *v, double *f, int x, int y, double eps, int nmax, int block_num, MPI_Comm comm){
    int n = 0;
    int nx = x/block_num;
    int ny = y/block_num;

    // e = error
    double local_e = 2. * eps;
    double *vp = (double *) malloc(nx * ny * sizeof(double));
    double* sendbuf = (double *) malloc((nx + ny) * 2 * sizeof(double));
    double* recvbuf = (double *) malloc((nx + ny) * 2 * sizeof(double));

    // prepare the data structures for mpi neighbor all to all send
    const int sendcounts[4] = {nx, nx, ny, ny};
    const int recvcounts[4] = {nx, nx, ny, ny};
    const int sdispls[4] = {0, nx, 2*nx, 2*nx+ny};
    const int rdispls[4] = {0, nx, 2*nx, 2*nx+ny};

    // rank in grid
    int rank, size;
    int coords[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    // Determines process coords in cartesian topology given rank in group, coords store i,j coord in grid
    MPI_Cart_coords(comm, rank, 2, coords);

    while ((local_e > eps) && (n < nmax))
    {
        local_e = 0.0;

        prepare_halo_buf(&v[0], &sendbuf[0], nx, ny);
        // halo exchange
        MPI_Neighbor_alltoallv(sendbuf, sendcounts, sdispls, MPI_DOUBLE, \
                           recvbuf, recvcounts, rdispls, MPI_DOUBLE, comm);

        //char *msg = (char *) malloc(nx * ny * (sizeof (double) * 2));
        //char pre[25] = "proc recvbuf ";
        //prepare_log(&pre[0], &msg[0], recvbuf, (nx + ny) * 2, &coords[0]);
        //log_info(msg);

        //print_arr(f, nx*ny);
        //printf("coord %d, %d: recvbuf: \n", coords[0], coords[1]);
        //print_arr(recvbuf, 2*(nx+ny));

        // update middle part
        omp_set_num_threads(2);
#pragma omp parallel for default(none) shared(nx, ny, vp, f, v, recvbuf)
        for(int ix = 1; ix < (nx-1); ix++)
        {
            for(int iy = 1; iy < (ny-1); iy++)
            {
                // x(k+1)
                update_v(&vp[0], &f[0], &v[0], &recvbuf[0], ix, iy, nx, ny);
            }
        }

        // update top part
        if(coords[0] != 0)
        {
#pragma omp parallel for default(none) shared(nx, ny, vp, f, v, recvbuf, coords, x, y)
            for(int i = 0; i < nx; i++)
            {
                if(is_big_boundary(i, 0, coords[0], coords[1], nx, ny, x, y))
                {
                    continue;
                }
                update_v(&vp[0], &f[0], &v[0], &recvbuf[0], i, 0, nx, ny);
            }
        }

        //update bottom part
        if(coords[0] != block_num - 1)
        {
#pragma omp parallel for default(none) shared(nx, ny, vp, f, v, recvbuf, coords, x, y)
            for(int i = 0; i < nx; i++)
            {
                if(is_big_boundary(i, ny - 1, coords[0], coords[1], nx, ny, x, y))
                {
                    continue;
                }
                update_v(&vp[0], &f[0], &v[0], &recvbuf[0], i, ny - 1, nx, ny);
            }
        }

        // update left part
        if(coords[1] != 0)
        {
#pragma omp parallel for default(none) shared(nx, ny, vp, f, v, recvbuf, coords, x, y)
            for(int i = 0; i < ny; i++)
            {
                if(is_big_boundary(0, i, coords[0], coords[1], nx, ny, x, y))
                {
                    continue;
                }
                update_v(&vp[0], &f[0], &v[0], &recvbuf[0], 0, i, nx, ny);
            }
        }

        // update right part
        if(coords[1] != block_num - 1)
        {
#pragma omp parallel for default(none) shared(nx, ny, vp, f, v, recvbuf, coords, x, y)
            for(int i = 0; i < ny; i++)
            {
                if(is_big_boundary(nx - 1, i, coords[0], coords[1], nx, ny, x, y))
                {
                    continue;
                }
                update_v(&vp[0], &f[0], &v[0], &recvbuf[0], nx - 1, i, nx, ny);
            }
        }

        //printf("coord %d, %d: after update: \n", coords[0], coords[1]);
        //print_arr(vp, nx*ny);

        double local_w = 0.0;
//#pragma omp parallel for default(none) shared(nx, ny, coords, x, y, local_e, vp, v, local_w)
        for(int ix = 0; ix < nx; ix++)
        {
            for(int iy = 0; iy < ny; iy++)
            {
                if(is_big_boundary(ix, iy, coords[0], coords[1], nx, ny, x, y))
                {
                    continue;
                }
                // e = max(x(k+1) - x(k))
                evaluate_e(&local_e, vp[nx * iy + ix], v[nx * iy + ix]);
                // v = vp
                v[nx * iy + ix] = vp[nx * iy + ix];
                // maintain correctness for thread num = 1
                local_w += fabs(v[nx * iy + ix]);
            }
        }

        //char pre1[25] = "proc v after update ";
        //prepare_log(&pre1[0], &msg[0], v, nx * ny, &coords[0]);
        //log_info(msg);

        //printf("coord %d, %d: after prune: \n", coords[0], coords[1]);
        //print_arr(v, nx*ny);
        //printf("%f\n", local_w);

        update_boundary(block_num, nx, ny, &coords[0], &v[0], &local_w, comm);

        //char pre2[25] = "proc v boundary update ";
        //prepare_log(&pre2[0], &msg[0], v, nx * ny, &coords[0]);
        //log_info(msg);
        //printf("coord %d, %d: after substitution \n", coords[0], coords[1]);
        //print_arr(v, nx*ny);
        //printf("local_w = %f\n", local_w);
        //printf("local_e = %f\n", local_e);

        int rank00;
        int coords00[2] = {0, 0};
        MPI_Cart_rank(comm, coords00, &rank00);

        double global_w;
        double global_e;
        MPI_Reduce(&local_w, &global_w, 1, MPI_DOUBLE, MPI_SUM, rank00, comm);
        MPI_Reduce(&local_e, &global_e, 1, MPI_DOUBLE, MPI_MAX, rank00, comm);

        if(rank == rank00)
        {
            global_w /= (x*y);
            global_e /= global_w;
            //printf("global_w = %f\n", global_w);
            printf("global_e = %f\n", global_e);
        }

        MPI_Bcast(&global_e, 1, MPI_DOUBLE, rank00, comm);
        local_e = global_e;
        //printf("local_e = %f\n", local_e);

        //free(msg);
        MPI_Barrier(comm);
        n++;
    }

    free(vp);
    free(sendbuf);
    free(recvbuf);

}