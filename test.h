//
// Created by shiting on 2023-12-22.
//

#ifndef POISSON_TEST_H
#define POISSON_TEST_H

#include <stdbool.h>

bool check_matrix_init(const double *local_v, const double *local_f, int q, int NX, int NY, int I, int J);

void update_single_matrix(double *v, int nx, int ny, double *w);
bool test_sum(const double *arr, double sum, int len);

#endif //POISSON_TEST_H
