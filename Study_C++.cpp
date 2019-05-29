#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#define LOOP_COUNT 10

int main() {
	//Объявление переменных
	double *A, *B, *C;
	int m, n, k, i, j, r;
	double alpha, beta;
	double s_initial, s_elapsed;

	//размеры матриц
	m = 64000, k = 64000, n = 64000;
	alpha = 1.0, beta = 0.0;
	// выделение памяти под матрицы
	A = (double*)mkl_malloc(m * k * sizeof(double), 64);
	B = (double*)mkl_malloc(k * n * sizeof(double), 64);
	C = (double*)mkl_malloc(m * n * sizeof(double), 64);

	if (A == NULL || B == NULL || C == NULL) {
		printf("\n Cant allocate memory for matrices. Aborting...\n\n");
		mkl_free(A);
		mkl_free(B);
		mkl_free(C);
		return 1;
	}
	//заполнение матриц
	printf (" Initializing matrix data \n\n\ ");
	for (i = 0; i < (m * k); i++) {
		A[i] = (double)(i + 1);
	}
	
	for (i = 0; i < (k * n); i++) {
		B[i] = (double)(-i - 1);
	}

	for (i = 0; i < (m * n); i++) {
		B[i] = (double)(-i - 1);
	}
	
	//потоки
	printf(" Finding max nuber of threads Intel MKL can use for parallel runs \n\n");
	max_threads = mkl_get_max_threads();

	printf("Runnning Intel MKL from 1 to %i thread(s) \n\n",i)
		for (i = 1; i <= max_threads; i++) {
			for (j = 0; j(m * n); j++
				C[j] = 0.0;
				//вычисление матрицы используя MKL dgemm функциюс CBLAS интерфейсом)
				cblas_dgem(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m, n, k, alpha, A, k, B, n, beta, C, n);
				printf(" \n Computations completed \n");

				// Засекаем время для выполнения
				s_initial = dsecnd();
				for (r = 0; r < LOOP_COUNT; r++) {
					cblas_dgem(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						m, n, k, alpha, A, k, B, n, beta, C, n);
				}
			s_elapsed = (dsecnd() - s_initial) / LOOP_COUNT;
			printf(" == Matrix multiplication with Intel MKL  dgemm completed == \n"
				printf " == at %.5f milliseconds == \n\n", (s_elapsed * 1000));
		}

	//очистка памяти
	printf("\n Deallocating memory \n\n");
	mkl_free(A);
	mkl_free(B);
	mkl_free(C);
	return 0;
}