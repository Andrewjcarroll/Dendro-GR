#include "JTP6.h"

using namespace std;

void JTPDeriv6_dP(double *P, int n) {
    double alpha = 17.0 / 57.0;
    double beta = -1.0 / 114.0;

    // Initialize the matrix to zeros
    for (int i = 0; i < n * n; i++) {
        P[i] = 0.0;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                // Main diagonal
                P[i * n + j] = 1;
            } else if (i == j + 1 || i == j - 1) {
                // Adjacent diagonals
                P[i * n + j] = alpha;
            } else if (i == j + 2 || i == j - 2) {
                // Adjacent super-diagonals
                P[i * n + j] = beta;
            } else {
                P[i * n + j] = 0;
            }
        }
    }

    P[0 * n + 1] = 8.0;
    P[0 * n + 2] = 6.0;
    P[(n - 1) * n + (n - 3)] = 6.0;
    P[(n - 1) * n + (n - 2)] = 8.0;

    // Set diagonal values to 1
    for (int i = 0; i < n; i++) {
        P[i * n + i] = 1.0;
    }
}
void JTPDeriv6_dQ(double *Q, int n) {
    double hm1 = 1.0 / 1.0;

    // Initialize the matrix to zeros
    for (int i = 0; i < n * n; i++) {
        Q[i] = 0.0;
    }

    double a = -43.0 / 12.0 * hm1;
    double b = -20.0 / 3.0 * hm1;
    double c = 9.0 * hm1;
    double d = 4.0 / 3.0 * hm1;
    double e = -1.0 / 12.0 * hm1;

    double t1 = 30.0 / 19.0 * 0.5 * hm1;

    Q[(n - 1) * n + (n - 5)] = -e;
    Q[(n - 1) * n + (n - 4)] = -d;
    Q[(n - 1) * n + (n - 3)] = -c;
    Q[(n - 1) * n + (n - 2)] = -b;
    Q[(n - 1) * n + (n - 1)] = -a;

    Q[0 * n + 0] = a;
    Q[0 * n + 1] = b;
    Q[0 * n + 2] = c;
    Q[0 * n + 3] = d;
    Q[0 * n + 4] = e;

    for (int i = 1; i < n - 1; i++) {
        Q[i * n + (i - 1)] = -t1;
        Q[i * n + (i + 1)] = t1;
    }
}
bool initJTPDeriv6(double *R, const unsigned int n) {
    double *P = new double[n * n];
    double *Q = new double[n * n];
    JTPDeriv6_dP(P, n);  // define the matrix P using the function provided

    // Define the matrix Q
    JTPDeriv6_dQ(Q, n);

    // Compute the LU decomposition of the matrix P
    int *ipiv = new int[n];
    int info;
    int nx = n;  // lapack needs fortran-compatible ints, not const unsigned
    dgetrf_(&nx, &nx, P, &nx, ipiv, &info);

    if (info != 0) {
        std::cerr << "LU factorization failed: " << info << std::endl;
        delete[] ipiv;
        return 1;
    }

    // Compute the inverse of the matrix P
    double *Pinv = new double[n * n];
    std::memcpy(Pinv, P, n * n * sizeof(double));
    int lwork = n * n;
    double *work = new double[lwork];
    dgetri_(&nx, Pinv, &nx, ipiv, work, &lwork, &info);

    if (info != 0) {
        std::cerr << "Matrix inversion failed: " << info << std::endl;
        delete[] ipiv;
        delete[] Pinv;
        delete[] work;
        return 1;
    }

    // Compute the product of the inverted matrix Pinv and matrix Q
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            R[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                R[i * n + j] += Pinv[i * n + k] * Q[k * n + j];
            }
        }
    }

    delete[] ipiv;
    delete[] Pinv;
    delete[] work;
    delete[] P;
    delete[] Q;

    return 0;
}
