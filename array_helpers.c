//
// Created by shiting on 2023-12-21.
//

#include <stdlib.h>
#include "stdio.h"
#include <math.h>
#include <string.h>
#include "stdbool.h"
#include "mpi/mpi.h"
#include "array_helpers.h"

void print_arr(double * arr, int len){
    for(int i = 0; i < len; i++)
    {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

void transform_arr_to_string(char *str, double *arr, int len){
    for(int i = 0; i < len; i++)
    {
        char tmp[sizeof(arr[i])+1];
        sprintf(tmp, "%f", arr[i]);
        //printf("%s\n", tmp);
        strcat(str, tmp);

        if(i == len - 1)
        {
            char end = '\0';
            strcat(str, &end);
        }
        else{
            char space = ' ';
            strcat(str, &space);
        }
    }
}

void str_proc_info(char *pinfo, int *coords){
    char tmp[sizeof(coords)], tmp1[sizeof(coords)];
    sprintf(tmp, "%d", coords[0]);
    sprintf(tmp1, "%d", coords[1]);
    strcat(pinfo, tmp);
    char space = ',';
    strcat(pinfo, &space);
    strcat(pinfo, tmp1);
    char dc = ':';
    strcat(pinfo, &dc);
    char end = '\0';
    strcat(pinfo, &end);
}

void get_big_matrix_coords(int ix, int iy, int I, int J, int nx, int ny, int *coords){
    coords[0] = nx * J + ix;
    coords[1] = ny * I + iy;
}

bool is_big_boundary(int ix, int iy, int I, int J, int nx, int ny, int NX, int NY){
    int coords[2];
    get_big_matrix_coords(ix, iy, I, J, nx, ny, &coords[0]);
    //int X = nx * J + ix;
    //int Y = ny * I + iy;
    if(coords[0] == 0 || coords[0] == NX - 1 || coords[1] == 0 || coords[1] == NY - 1)
    {
        return true;
    }
    return false;
}

void substitute_part_arr(double *target_arr, const double *sub, int x, int y, int n, enum layout typ)
{
    switch (typ) {
        case ROW:
            for(int i = 0; i < x; i++)
            {
                target_arr[n * x + i] = sub[i];
            }
            break;
        case COLUMN:
            for(int i = 0; i < y; i++)
            {
                target_arr[i * x + n] = sub[i];
            }
            break;
    }
}

void get_w(const double *arr, double *sum, int len)
{
    *sum = 0.0;
    for(int i = 0; i < len; i++)
    {
        *sum += fabs(arr[i]);
    }

}

bool idx_out_of_range(int idx, int nx, int ny){
    if(idx < 0 || idx > nx * ny - 1){
        return true;
    } else {
        return false;
    }
}

double get_neighbours(int ix, int iy, int nx, int ny, const double *v, const double *recvbuf){
    int idx_top = nx * (iy - 1) + ix;
    int idx_bottom = nx * (iy + 1) + ix;
    int idx_left = nx * iy + ix - 1;
    int idx_right = nx * iy + ix + 1;
    double sum = 0.0;

    if(iy == 0)
    {
        sum += recvbuf[ix];
    } else{
        sum += v[idx_top];
    }

    if(iy == ny - 1)
    {
        sum += recvbuf[nx + ix];
    } else{
        sum += v[idx_bottom];
    }

    if(ix == 0)
    {
        sum += recvbuf[2 * nx + iy];
    } else{
        sum += v[idx_left];
    }

    if(ix == nx - 1)
    {
        sum += recvbuf[2 * nx + ny + iy];
    } else{
        sum += v[idx_right];
    }

    return sum;
}

void add_zeros(const int *coords, int block_num, double *v, int nx, int ny){
    if(coords[0] == 0 && coords[1] == 0)
    {
        v[0] = 0.0;
    } else if(coords[0] == 0 && coords[1] == block_num - 1){
        v[nx - 1] = 0.0;
    } else if(coords[0] == block_num - 1 && coords[1] == 0){
        v[(ny - 1) * nx] = 0.0;
    } else if(coords[0] == block_num - 1 && coords[1] == block_num - 1){
        v[nx * ny - 1] = 0.0;
    } else{
        return;
    }
}

void distribute_vector(int x_size, double * v, double ** v_local, MPI_Comm comm) {
    // rank in grid
    int rank, size;
    int coords[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    // Determines process coords in cartesian topology given rank in group, coords store i,j coord in grid
    MPI_Cart_coords(comm, rank, 2, coords);

    // get rank of root 0,0 matrix in grid, todo: del this
    int rank00;
    int coords00[2] = {0, 0};
    MPI_Cart_rank(comm, coords00, &rank00);

    // floor and ceil values of the segments after split
    int fnp = (int) floor(((double) x_size) / size);
    int cnp = (int) ceil(((double) x_size) / size);

    // allocate memory for rcv buffer
    int rcvsize;
    if (rank < (x_size % size)) {
        rcvsize = cnp;
    }
    else {
        rcvsize = fnp;
    }

    // recvbuf
    double *temp_vector = (double *) malloc(rcvsize * sizeof(double));
    *v_local = &temp_vector[0];

    // displs
    int *disp = (int *) malloc(size * sizeof(int));

    // sendcounts
    int *ncount = (int *) malloc(size * sizeof(int));

    for (int i = 0; i < size; i++)
    {
        if (i < (x_size % size))
        {
            ncount[i] = cnp;
        }
        else {
            ncount[i] = fnp;
        }

        if (i > 0)
        {
            disp[i] = disp[i - 1] + ncount[i - 1];
        }
        else {
            disp[i] = 0;
        }
    }

    // scatterv
    MPI_Scatterv(v,ncount,disp,
                 MPI_DOUBLE,temp_vector,rcvsize,
                 MPI_DOUBLE,
                 rank00, comm);

    printf("\n data in process[%d] \n", rank);
    print_arr(temp_vector, ncount[rank]);

    free(ncount);
    free(disp);
}