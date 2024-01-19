//
// Created by shiting on 2023-12-21.
//
#include <stdbool.h>

#ifndef POISSON_ARRAY_HELPERS_H
#define POISSON_ARRAY_HELPERS_H

enum layout{
    ROW,
    COLUMN
};

void print_arr(double * arr, int len);

void transform_arr_to_string(char *str, double *arr, int len);

void str_proc_info(char *pinfo, int *coords);

void get_big_matrix_coords(int ix, int iy, int I, int J, int nx, int ny, int *coords);

bool is_big_boundary(int ix, int iy, int I, int J, int nx, int ny, int NX, int NY);

void substitute_part_arr(double *target_arr, const double *sub, int x, int y, int n, enum layout typ);

void add_zeros(const int *coords, int block_num, double *v, int nx, int ny);

void get_w(const double *arr, double *sum, int len);

double get_neighbours(int ix, int iy, int nx, int ny, const double *v, const double *recvbuf);

#endif //POISSON_ARRAY_HELPERS_H
