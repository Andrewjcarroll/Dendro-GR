#include "cfd.h"

namespace dendro_cfd {

// initialize a "global" cfd object
CompactFiniteDiff cfd(0, 0);

CompactFiniteDiff::CompactFiniteDiff(const unsigned int num_dim,
                                     const unsigned int padding_size,
                                     const DerType deriv_type,
                                     const unsigned int filter_type) {
    if (deriv_type != CFD_NONE && deriv_type != CFD_P1_O4 &&
        deriv_type != CFD_P1_O6 && deriv_type != CFD_Q1_O6_ETA1 &&
        deriv_type != CFD_KIM_O4 && deriv_type != CFD_HAMR_O4 &&
        deriv_type != CFD_JT_O6) {
        throw std::invalid_argument(
            "Couldn't initialize CFD object, deriv type was not a valid 'base' "
            "type: deriv_type = " +
            std::to_string(deriv_type));
    }

    m_deriv_type = deriv_type;
    m_filter_type = filter_type;
    m_curr_dim_size = num_dim;
    m_padding_size = padding_size;

    initialize_cfd_storage();

    if (num_dim == 0) {
        return;
    }

    if (deriv_type == CFD_NONE) {
        return;
    }

    initialize_cfd_matrix();
    initialize_cfd_filter();
}

CompactFiniteDiff::~CompactFiniteDiff() {
    // make sure we delete the cfd matrix to avoid memory leaks
    delete_cfd_matrices();
}

void CompactFiniteDiff::change_dim_size(const unsigned int dim_size) {
    if (m_curr_dim_size == dim_size) {
        return;
    } else {
        delete_cfd_matrices();

        m_curr_dim_size = dim_size;

        initialize_cfd_storage();

        // if deriv type is none, for some reason, just exit
        if (m_deriv_type == CFD_NONE) {
            return;
        }

        initialize_cfd_matrix();
        initialize_cfd_filter();
    }
}

void CompactFiniteDiff::initialize_cfd_storage() {
    // NOTE: 0 indicates that it's initialized with all elements set to 0
    m_RF = new double[m_curr_dim_size * m_curr_dim_size]();

    m_R = new double[m_curr_dim_size * m_curr_dim_size]();
    m_R_left = new double[m_curr_dim_size * m_curr_dim_size]();
    m_R_right = new double[m_curr_dim_size * m_curr_dim_size]();
    m_r_leftright = new double[m_curr_dim_size * m_curr_dim_size]();

    // NOTE: the () syntax only works with C++ 11 or greater, may need to
    // use std::fill_n(array, n, 0); to 0 set the data or use std::memset(array,
    // 0, sizeof *array * size)

    m_u1d = new double[m_curr_dim_size];
    m_du1d = new double[m_curr_dim_size];
}

void CompactFiniteDiff::initialize_cfd_matrix() {
    // temporary P and Q storage used in calculations
    double *P = new double[m_curr_dim_size * m_curr_dim_size]();
    double *Q = new double[m_curr_dim_size * m_curr_dim_size]();

    // TODO: need to build up the other three combinations (the last one is
    // likely never going to happen)
    buildPandQMatrices(P, Q, m_padding_size, m_curr_dim_size, m_deriv_type,
                       false, false);

    std::cout << "\nP MATRIX" << std::endl;
    print_square_mat(P, m_curr_dim_size);

    std::cout << "\nQ MATRIX" << std::endl;
    print_square_mat(Q, m_curr_dim_size);

    calculateDerivMatrix(m_R, P, Q, m_curr_dim_size);

    std::cout << "\nDERIV MATRIX" << std::endl;
    print_square_mat(m_R, m_curr_dim_size);

    delete[] P;
    delete[] Q;

    // // Print the R matrix
    // MPI_Comm comm = MPI_COMM_WORLD;

    // int rank, npes;
    // MPI_Comm_rank(comm, &rank);
    // MPI_Comm_size(comm, &npes);
    //     int rank = 0;

    //     if (rank == 0) {
    //         printf("R matrix:\n");
    //         for (unsigned int k = 0; k < m_curr_dim_size; k++) {
    //             for (unsigned int m = 0; m < m_curr_dim_size; m++) {
    //                 printf("%f ", m_R[k * m_curr_dim_size + m]);
    //             }
    //             printf("\n");
    //         }
    //     }
}

void CompactFiniteDiff::initialize_cfd_filter() {
    switch (m_filter_type) {
        case 0:
            // do nothing
            break;
        case 1:
            if (initKim_Filter_Deriv4(m_RF, m_curr_dim_size)) {
                // failed to initialize again
                std::cerr << RED
                          << "ERROR: could not construct the Kim Deriv4 Filter "
                             "matrix!"
                          << std::endl;
                exit(0);
            }

        default:
            break;
    }

    // // Print the RF matrix
    // MPI_Comm comm = MPI_COMM_WORLD;

    // int rank, npes;
    // MPI_Comm_rank(comm, &rank);
    // MPI_Comm_size(comm, &npes);

    // if (rank == 0) {
    //     printf("RF matrix:\n");
    //     for (unsigned int k = 0; k < m_curr_dim_size; k++) {
    //         for (unsigned int m = 0; m < m_curr_dim_size; m++) {
    //             printf("%f ", m_RF[k * m_curr_dim_size + m]);
    //         }
    //         printf("\n");
    //     }
    // }
}

void CompactFiniteDiff::delete_cfd_matrices() {
    delete[] m_RF;
    delete[] m_R;
    delete[] m_u1d;
    delete[] m_du1d;
}

void CompactFiniteDiff::cfd_x(double *const Dxu, const double *const u,
                              const double dx, const unsigned int *sz,
                              unsigned bflag) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    std::cout << "Nx, ny, nz: " << nx << " " << ny << " " << nz << std::endl;

    char TRANSA = 'N';
    char TRANSB = 'N';
    int M = nx;
    int N = 1;
    int K = nx;
    double alpha = 1.0 / dx;
    double beta = 0.0;

    // left boundary to watch for
    // if (bflag & (1u << OCT_DIR_LEFT)) ;

    // // right boundary to watch for
    // if (bflag & (1u << OCT_DIR_RIGHT)) ;

    for (unsigned int k = 0; k < nz; k++) {
        for (unsigned int j = 0; j < ny; j++) {
            for (unsigned int i = 0; i < nx; i++) {
                m_u1d[i] = u[INDEX_3D(i, j, k)];
            }

            // TODO: call just a pointer
            // // optimization for X, since memory is laid out
            // // z -> y -> x (as indicated by IDX/INDEX_3D)
            // // we can just calculate the pointer address to use
            // double *u_ptr = Dxu + nx * (j + ny * k);
            // std::cout << "\nDxu and u_ptr " << Dxu << " " << u_ptr << " the
            // junk:"  << nx * (j + ny * k) << std::endl;

            // std::cout << u_ptr - Dxu << std::endl;

            // // std::cout << std::endl;
            // for (unsigned int i = 0; i < nx; i++) {
            //     std::cout << m_u1d[i] << " " << u_ptr[i] << std::endl;
            // }

            // for (unsigned int i = 0; i < nx; i++) {
            //     m_du1d[i] = 0.0;
            //     for (unsigned int m = 0; m < nx; m++) {
            //         m_du1d[i] += m_R[i * nx + m] * m_u1d[m];
            //     }
            // }

            dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, m_R, &M, m_u1d, &K,
                   &beta, m_du1d, &M);

            for (unsigned int i = 0; i < nx; i++) {
                Dxu[INDEX_3D(i, j, k)] = m_du1d[i];
            }
        }
    }
}

void CompactFiniteDiff::cfd_y(double *const Dyu, const double *const u,
                              const double dy, const unsigned int *sz,
                              unsigned bflag) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    char TRANSA = 'N';
    char TRANSB = 'N';
    int M = ny;
    int N = 1;
    int K = ny;
    double alpha = 1.0 / dy;
    double beta = 0.0;

    // if (bflag & (1u << OCT_DIR_DOWN));

    // if (bflag & (1u << OCT_DIR_UP));

    for (unsigned int k = 0; k < nz; k++) {
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int j = 0; j < ny; j++) {
                m_u1d[j] = u[INDEX_3D(i, j, k)];
            }

            // for (unsigned int j = 0; j < ny; j++) {
            //     m_du1d[j] = 0.0;
            //     for (unsigned int m = 0; m < ny; m++) {
            //         m_du1d[j] += m_R[j * ny + m] * m_u1d[m];
            //         // m_du1d[j] += m_R[j * ny + m] * u[INDEX_3D(i, m, k)];
            //     }
            // }

            dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, m_R, &M, m_u1d, &K,
                   &beta, m_du1d, &M);

            for (unsigned int j = 0; j < ny; j++) {
                Dyu[INDEX_3D(i, j, k)] = m_du1d[j];
            }
        }
    }
}

void CompactFiniteDiff::cfd_z(double *const Dzu, const double *const u,
                              const double dz, const unsigned int *sz,
                              unsigned bflag) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    char TRANSA = 'N';
    char TRANSB = 'N';
    int M = nz;
    int N = 1;
    int K = nz;
    double alpha = 1.0 / dz;
    double beta = 0.0;

    // if (bflag & (1u << OCT_DIR_BACK)) ;

    // if (bflag & (1u << OCT_DIR_FRONT)) ;

    for (unsigned int j = 0; j < ny; j++) {
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                m_u1d[k] = u[INDEX_3D(i, j, k)];
            }

            // for (unsigned int k = 0; k < nz; k++) {
            //     m_du1d[k] = 0.0;
            //     for (unsigned int m = 0; m < nz; m++) {
            //         m_du1d[k] += m_R[k * nz + m] * m_u1d[m];
            //         // m_du1d[k] += m_R[k * nz + m] * u[INDEX_3D(i, j, m)];
            //     }
            // }

            dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, m_R, &M, m_u1d, &K,
                   &beta, m_du1d, &M);

            for (unsigned int k = 0; k < nz; k++) {
                Dzu[INDEX_3D(i, j, k)] = m_du1d[k];
            }
        }
    }
}

void CompactFiniteDiff::filter_cfd_x(double *const u, const double dx,
                                     const unsigned int *sz, unsigned bflag) {
    if (m_filter_type == 0) {
        return;
    }

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    for (unsigned int k = 0; k < nz; k++) {
        for (unsigned int j = 0; j < ny; j++) {
            for (unsigned int i = 0; i < nx; i++) {
                m_u1d[i] = u[INDEX_3D(i, j, k)];
            }

            for (unsigned int i = 0; i < nx; i++) {
                m_du1d[i] = 0.0;
                for (unsigned int m = 0; m < nx; m++) {
                    m_du1d[i] += m_RF[i * nx + m] * m_u1d[m];
                }
            }

            for (unsigned int i = 0; i < nx; i++) {
                u[INDEX_3D(i, j, k)] +=
                    m_du1d[i];  // do we need to include a dx here?
            }
        }
    }
}

void CompactFiniteDiff::filter_cfd_y(double *const u, const double dx,
                                     const unsigned int *sz, unsigned bflag) {
    if (m_filter_type == 0) {
        return;
    }

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];
    for (unsigned int k = 0; k < nz; k++) {
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int j = 0; j < ny; j++) {
                m_u1d[j] = u[INDEX_3D(i, j, k)];
            }
            for (unsigned int j = 0; j < ny; j++) {
                m_du1d[j] = 0.0;
                for (unsigned int m = 0; m < ny; m++) {
                    m_du1d[j] += m_RF[j * ny + m] * m_u1d[m];
                }
            }
            for (unsigned int j = 0; j < ny; j++) {
                u[INDEX_3D(i, j, k)] +=
                    m_du1d[j];  // do we need to include a dy here?
            }
        }
    }
}

void CompactFiniteDiff::filter_cfd_z(double *const u, const double dz,
                                     const unsigned int *sz, unsigned bflag) {
    if (m_filter_type == 0) {
        return;
    }

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    for (unsigned int j = 0; j < ny; j++) {
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                m_u1d[k] = u[INDEX_3D(i, j, k)];
            }
            for (unsigned int k = 0; k < nz; k++) {
                m_du1d[k] = 0.0;
                for (unsigned int m = 0; m < nz; m++) {
                    m_du1d[k] += m_RF[k * nz + m] * m_u1d[m];
                }
            }
            for (unsigned int k = 0; k < nz; k++) {
                u[INDEX_3D(i, j, k)] +=
                    m_du1d[k];  // do we need to include a dz here?
            }
        }
    }
}

DerType getDerTypeForEdges(const DerType derivtype,
                           const BoundaryType boundary) {
    DerType doptions_CFD_P1_O4[4] = {CFD_P1_O4, CFD_DRCHLT_ORDER_4,
                                     CFD_P1_O4_CLOSE, CFD_P1_O4_L4_CLOSE};
    DerType doptions_CFD_P1_O6[4] = {CFD_P1_O6, CFD_DRCHLT_ORDER_6,
                                     CFD_P1_O6_CLOSE, CFD_P1_O6_L6_CLOSE};
    DerType doptions_CFD_Q1_O6_ETA1[4] = {CFD_Q1_O6_ETA1, CFD_DRCHLT_Q6,
                                          CFD_Q1_O6_ETA1_CLOSE,
                                          CFD_P1_O6_L6_CLOSE};

    // the doptions to use
    DerType *doptions;

    switch (derivtype) {
        case CFD_P1_O4:
            doptions = doptions_CFD_P1_O4;
            break;
        case CFD_P1_O6:
            doptions = doptions_CFD_P1_O6;
            break;
        case CFD_Q1_O6_ETA1:
            doptions = doptions_CFD_Q1_O6_ETA1;
            break;

        default:
            throw std::invalid_argument(
                "Invalid type of CFD derivative called! derivtype=" +
                std::to_string(derivtype));
            break;
    }

    switch (boundary) {
        case BLOCK_CFD_DIRICHLET:
        case BLOCK_PHYS_BOUNDARY:
            return doptions[1];
        case BLOCK_CFD_CLOSURE:
            return doptions[2];
        case BLOCK_CFD_LOPSIDE_CLOSURE:
            return doptions[3];
        default:
            return doptions[1];
    }
}

void buildPandQMatrices(double *P, double *Q, const uint32_t padding,
                        const uint32_t n, const DerType derivtype,
                        const bool is_left_edge, const bool is_right_edge) {
    // NOTE: we're pretending that all of the "mpi" or "block" boundaries
    // are treated equally. We only need to account for physical "left" and
    // "right" edges

    // NOTE: (2) we're also assuming that P and Q are initialized to **zero**.
    // There are no guarantees in this function if they are not.
    std::cout << derivtype << " is the deriv type" << std::endl;

    uint32_t curr_n = n;
    uint32_t i_start = 0;
    uint32_t i_end = n;
    uint32_t j_start = 0;
    uint32_t j_end = n;

    if (is_left_edge) {
        // initialize the "diagonal" in the padding to 1
        for (uint32_t ii = 0; ii < padding; ii++) {
            P[INDEX_2D(ii, ii)] = 1.0;
            Q[INDEX_2D(ii, ii)] = 1.0;
        }
        i_start += padding;
        j_start += padding;
        curr_n -= padding;
    }

    if (is_right_edge) {
        // initialize bottom "diagonal" in padding to 1 as well
        for (uint32_t ii = n; ii >= n - padding; ii--) {
            P[INDEX_2D(ii, ii)] = 1.0;
            Q[INDEX_2D(ii, ii)] = 1.0;
        }
        i_end -= padding;
        j_end -= padding;
        curr_n -= padding;
    }

    std::cout << "i : " << i_start << " " << i_end << std::endl;
    std::cout << "j : " << j_start << " " << j_end << std::endl;

    // NOTE: when at the "edges", we need a temporary array that can be copied
    // over
    double *tempP = nullptr;
    double *tempQ = nullptr;

    if (is_left_edge or is_right_edge) {
        // initialize tempP to be a "smaller" square matrix for use
        tempP = new double[curr_n * curr_n]();
        tempQ = new double[curr_n * curr_n]();
    } else {
        // just use the same pointer value, then no need to adjust later even
        tempP = P;
        tempQ = Q;
    }

    if (derivtype == CFD_P1_O4 || derivtype == CFD_P1_O6 ||
        derivtype == CFD_Q1_O6_ETA1) {
        // NOTE: this is only for the NONISOTROPIC matrices!!!

        // now build up the method object that will be used to calculate the
        // in-between values
        CFDMethod method(derivtype);

        int ibgn = 0;
        int iend = 0;

        DerType leftEdgeDtype;
        DerType rightEdgeDtype;

        if (is_left_edge) {
            leftEdgeDtype = getDerTypeForEdges(
                derivtype, BoundaryType::BLOCK_PHYS_BOUNDARY);
        } else {
            // TODO: update the boundary type based on what we want to build in
            leftEdgeDtype = getDerTypeForEdges(
                derivtype, BoundaryType::BLOCK_CFD_DIRICHLET);
        }

        if (is_right_edge) {
            rightEdgeDtype = getDerTypeForEdges(
                derivtype, BoundaryType::BLOCK_PHYS_BOUNDARY);
        } else {
            // TODO: update the boundary type based on what we want to build in
            rightEdgeDtype = getDerTypeForEdges(
                derivtype, BoundaryType::BLOCK_CFD_DIRICHLET);
        }

        std::cout << "Left edge is " << leftEdgeDtype << std::endl;

        buildMatrixLeft(tempP, tempQ, &ibgn, leftEdgeDtype, padding, curr_n);
        buildMatrixRight(tempP, tempQ, &iend, rightEdgeDtype, padding, curr_n);

        for (int i = ibgn; i <= iend; i++) {
            for (int k = -method.Ld; k <= method.Rd; k++) {
                if (!(i > -1) && !(i < curr_n)) {
                    if (is_left_edge or is_right_edge) {
                        delete[] tempP;
                        delete[] tempQ;
                    }
                    throw std::out_of_range(
                        "I is either less than zero or greater than curr_n! "
                        "i=" +
                        std::to_string(i) +
                        " curr_n=" + std::to_string(curr_n));
                }
                if (!((i + k) > -1) && !((i + k) < curr_n)) {
                    if (is_left_edge or is_right_edge) {
                        delete[] tempP;
                        delete[] tempQ;
                    }
                    throw std::out_of_range(
                        "i + k is either less than 1 or greater than curr_n! "
                        "i=" +
                        std::to_string(i + k) + " k=" + std::to_string(k) +
                        " curr_n=" + std::to_string(curr_n));
                }

                tempP[INDEX_N2D(i, i + k, curr_n)] =
                    method.alpha[k + method.Ld];
            }
            for (int k = -method.Lf; k <= method.Rf; k++) {
                if (!(i > -1) && !(i < curr_n)) {
                    throw std::out_of_range(
                        "(i is either less than zero or greater than curr_n! "
                        "i=" +
                        std::to_string(i) +
                        " curr_n=" + std::to_string(curr_n));
                }
                if (!((i + k) > -1) && !((i + k) < curr_n)) {
                    throw std::out_of_range(
                        "i + k is either less than 1 or greater than curr_n! "
                        "i=" +
                        std::to_string(i + k) + " k=" + std::to_string(k) +
                        " curr_n=" + std::to_string(curr_n));
                }

                tempQ[INDEX_N2D(i, i + k, curr_n)] = method.a[k + method.Lf];
            }
        }
    } else if (derivtype == CFD_KIM_O4) {
        // build Kim4 P
        KimDeriv4_dP(tempP, curr_n);

        // then build Q
        KimDeriv4_dQ(tempQ, curr_n);
    } else if (derivtype == CFD_HAMR_O4) {
        // build HAMR 4 P
        HAMRDeriv4_dP(tempP, curr_n);

        // then build Q
        HAMRDeriv4_dQ(tempQ, curr_n);
    } else if (derivtype == CFD_JT_O6) {
        // build JTP Deriv P
        JTPDeriv6_dP(tempP, curr_n);

        // then build Q
        JTPDeriv6_dQ(tempQ, curr_n);
    } else if (derivtype == CFD_NONE) {
        // just.... do nothing... keep them at zeros
        if (is_left_edge or is_right_edge) {
            delete[] tempP;
            delete[] tempQ;
        }
        throw std::invalid_argument(
            "dendro_cfd::buildPandQMatrices should never be called with a "
            "CFD_NONE deriv type!");
    } else {
        if (is_left_edge or is_right_edge) {
            delete[] tempP;
            delete[] tempQ;
        }
        throw std::invalid_argument(
            "The CFD deriv type was not one of the valid options. derivtype=" +
            std::to_string(derivtype));
    }

    // copy the values back in
    // NOTE: the use of j and i assumes ROW-MAJOR order, but it will just copy a
    // square matrix in no matter what, so it's not a big issue
    if (is_left_edge or is_right_edge) {
        // then memcopy the "chunks" to where they go inside the matrix
        uint32_t temp_arr_i = 0;
        // iterate over the rows
        for (uint32_t jj = j_start; jj < j_end; jj++) {
            // ii will only go from empty rows we actually need to fill...
            // j will start at "j_start" and go until "j_end" where we need to
            // fill memory start index of our main array

            uint32_t temp_start = INDEX_N2D(0, temp_arr_i, curr_n);
            // uint32_t temp_end = INDEX_N2D(curr_n - 1, temp_arr_i, curr_n);

            std::copy_n(&tempP[temp_start], curr_n, &P[INDEX_2D(i_start, jj)]);
            std::copy_n(&tempQ[temp_start], curr_n, &Q[INDEX_2D(i_start, jj)]);

            // increment temp_arr "row" value
            temp_arr_i++;
        }
        // clear up our temporary arrays we don't need
        delete[] tempP;
        delete[] tempQ;
    }
    // NOTE: tempP doesn't need to be deleted if it was not initialized,
    // so we don't need to delete it unless we're dealing with left/right edges
}

void calculateDerivMatrix(double *D, double *P, double *Q, const int n) {
    int *ipiv = new int[n];

    int info;
    int nx = n;

    dgetrf_(&nx, &nx, P, &nx, ipiv, &info);

    if (info != 0) {
        delete[] ipiv;
        throw std::runtime_error("LU factorization failed: info=" +
                                 std::to_string(info));
    }

    double *Pinv = new double[n * n];
    for (int i = 0; i < n * n; i++) {
        Pinv[i] = P[i];
    }

    int lwork = n * n;

    double *work = new double[lwork];

    dgetri_(&nx, Pinv, &nx, ipiv, work, &lwork, &info);

    if (info != 0) {
        delete[] ipiv;
        delete[] Pinv;
        delete[] work;
        throw std::runtime_error("Matrix inversion failed: info=" +
                                 std::to_string(info));
    }

    std::cout << "P INVERSE" << std::endl;
    print_square_mat(Pinv, n);

    mulMM(D, Pinv, Q, n, n);

    delete[] ipiv;
    delete[] Pinv;
    delete[] work;
}

void mulMM(double *C, double *A, double *B, int na, int nb) {
    /*  M = number of rows of A and C
        N = number of columns of B and C
        K = number of columns of A and rows of B
    */

    char TA[4], TB[4];
    double ALPHA = 1.0;
    double BETA = 0.0;
    sprintf(TA, "N");
    sprintf(TB, "N");
    int M = na;
    int N = nb;
    int K = na;
    int LDA = na;
    int LDB = na;
    int LDC = na;

    dgemm_(TA, TB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}

void buildMatrixLeft(double *P, double *Q, int *xib, const DerType dtype,
                     const int nghosts, const int n) {
    int ib = 0;

    std::cout << dtype << std::endl;

    switch (dtype) {
        case CFD_DRCHLT_ORDER_4: {
            P[INDEX_2D(0, 0)] = 1.0;
            P[INDEX_2D(0, 1)] = 3.0;

            Q[INDEX_2D(0, 0)] = -17.0 / 6.0;
            Q[INDEX_2D(0, 1)] = 3.0 / 2.0;
            Q[INDEX_2D(0, 2)] = 3.0 / 2.0;
            Q[INDEX_2D(0, 3)] = -1.0 / 6.0;
            ib = 1;
        } break;

        case CFD_DRCHLT_ORDER_6: {
            P[INDEX_2D(0, 0)] = 1.0;
            P[INDEX_2D(0, 1)] = 5.0;

            P[INDEX_2D(1, 0)] = 2.0 / 11.0;
            P[INDEX_2D(1, 1)] = 1.0;
            P[INDEX_2D(1, 2)] = 2.0 / 11.0;

            Q[INDEX_2D(0, 0)] = -197.0 / 60.0;
            Q[INDEX_2D(0, 1)] = -5.0 / 12.0;
            Q[INDEX_2D(0, 2)] = 5.0;
            Q[INDEX_2D(0, 3)] = -5.0 / 3.0;
            Q[INDEX_2D(0, 4)] = 5.0 / 12.0;
            Q[INDEX_2D(0, 5)] = -1.0 / 20.0;

            Q[INDEX_2D(1, 0)] = -20.0 / 33.0;
            Q[INDEX_2D(1, 1)] = -35.0 / 132.0;
            Q[INDEX_2D(1, 2)] = 34.0 / 33.0;
            Q[INDEX_2D(1, 3)] = -7.0 / 33.0;
            Q[INDEX_2D(1, 4)] = 2.0 / 33.0;
            Q[INDEX_2D(1, 5)] = -1.0 / 132.0;
            ib = 2;
        } break;

        case CFD_DRCHLT_Q6: {
            P[INDEX_2D(0, 0)] = 1.0;
            P[INDEX_2D(0, 1)] = 5.0;

            P[INDEX_2D(1, 0)] = 2.0 / 11.0;
            P[INDEX_2D(1, 1)] = 1.0;
            P[INDEX_2D(1, 2)] = 2.0 / 11.0;

            P[INDEX_2D(2, 1)] = 1.0 / 3.0;
            P[INDEX_2D(2, 2)] = 1.0;
            P[INDEX_2D(2, 3)] = 1.0 / 3.0;

            Q[INDEX_2D(0, 0)] = -197.0 / 60.0;
            Q[INDEX_2D(0, 1)] = -5.0 / 12.0;
            Q[INDEX_2D(0, 2)] = 5.0;
            Q[INDEX_2D(0, 3)] = -5.0 / 3.0;
            Q[INDEX_2D(0, 4)] = 5.0 / 12.0;
            Q[INDEX_2D(0, 5)] = -1.0 / 20.0;

            Q[INDEX_2D(1, 0)] = -20.0 / 33.0;
            Q[INDEX_2D(1, 1)] = -35.0 / 132.0;
            Q[INDEX_2D(1, 2)] = 34.0 / 33.0;
            Q[INDEX_2D(1, 3)] = -7.0 / 33.0;
            Q[INDEX_2D(1, 4)] = 2.0 / 33.0;
            Q[INDEX_2D(1, 5)] = -1.0 / 132.0;

            Q[INDEX_2D(2, 0)] = -1.0 / 36.0;
            Q[INDEX_2D(2, 1)] = -14.0 / 18.0;
            Q[INDEX_2D(3, 2)] = 0.0;
            Q[INDEX_2D(2, 3)] = 14.0 / 18.0;
            Q[INDEX_2D(2, 4)] = 1.0 / 36.0;

            ib = 3;
        } break;

        case CFD_P1_O4_CLOSE: {
            if (nghosts < 3) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "3! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = 0; i < nghosts; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ib = nghosts;
            P[INDEX_2D(ib, ib)] = 1.0;

            const double t1 = 1.0 / 72.0;
            Q[INDEX_2D(ib, ib - 3)] = -t1;
            Q[INDEX_2D(ib, ib - 2)] = 10.0 * t1;
            Q[INDEX_2D(ib, ib - 1)] = -53.0 * t1;
            Q[INDEX_2D(ib, ib)] = 0.0;
            Q[INDEX_2D(ib, ib + 1)] = 53.0 * t1;
            Q[INDEX_2D(ib, ib + 2)] = -10.0 * t1;
            Q[INDEX_2D(ib, ib + 3)] = t1;
            ib += 1;
        } break;

        case CFD_P1_O6_CLOSE: {
            if (nghosts < 4) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "4! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = 0; i < nghosts; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ib = nghosts;
            P[INDEX_2D(ib, ib)] = 1.0;

            const double t2 = 1.0 / 300.0;
            Q[INDEX_2D(ib, ib - 4)] = t2;
            Q[INDEX_2D(ib, ib - 3)] = -11.0 * t2;
            Q[INDEX_2D(ib, ib - 2)] = 59.0 * t2;
            Q[INDEX_2D(ib, ib - 1)] = -239.0 * t2;
            Q[INDEX_2D(ib, ib)] = 0.0;
            Q[INDEX_2D(ib, ib + 1)] = 239.0 * t2;
            Q[INDEX_2D(ib, ib + 2)] = -59.0 * t2;
            Q[INDEX_2D(ib, ib + 3)] = 11.0 * t2;
            Q[INDEX_2D(ib, ib + 4)] = -t2;
            ib += 1;
        } break;

        case CFD_P1_O4_L4_CLOSE: {
            if (nghosts < 1) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "1! nghosts = " +
                    std::to_string(nghosts));
            }

            // set ghost region to identity...don't use these derivs!
            for (int i = 0; i < nghosts; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ib = nghosts;
            P[INDEX_2D(ib, ib)] = 1.0;

            const double t3 = 1.0 / 12.0;
            Q[INDEX_2D(ib, ib - 1)] = -3.0 * t3;
            Q[INDEX_2D(ib, ib)] = -10.0 * t3;
            Q[INDEX_2D(ib, ib + 1)] = 18.0 * t3;
            Q[INDEX_2D(ib, ib + 2)] = -6.0 * t3;
            Q[INDEX_2D(ib, ib + 3)] = t3;

            ib += 1;
        } break;

        case CFD_P1_O6_L6_CLOSE: {
            if (nghosts < 2) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "2! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = 0; i < nghosts; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ib = nghosts;
            const double t4 = 1.0 / 60.0;
            P[INDEX_2D(ib, ib)] = 1.0;

            Q[INDEX_2D(ib, ib - 2)] = 2.0 * t4;
            Q[INDEX_2D(ib, ib - 1)] = -24.0 * t4;
            Q[INDEX_2D(ib, ib)] = -35.0 * t4;
            Q[INDEX_2D(ib, ib + 1)] = 80.0 * t4;
            Q[INDEX_2D(ib, ib + 2)] = -30.0 * t4;
            Q[INDEX_2D(ib, ib + 3)] = 8.0 * t4;
            Q[INDEX_2D(ib, ib + 4)] = -1.0 * t4;

            ib += 1;
        } break;

        case CFD_Q1_O6_ETA1_CLOSE: {
            if (nghosts < 4) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at "
                    "least "
                    "4! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = 0; i < nghosts; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ib = nghosts;
            P[INDEX_2D(ib, ib)] = 1.0;

            Q[INDEX_2D(ib, ib - 4)] = 0.0035978349;
            Q[INDEX_2D(ib, ib - 3)] = -0.038253676;
            Q[INDEX_2D(ib, ib - 2)] = 0.20036969;
            Q[INDEX_2D(ib, ib - 1)] = -0.80036969;
            Q[INDEX_2D(ib, ib)] = 0.0;
            Q[INDEX_2D(ib, ib + 1)] = 0.80036969;
            Q[INDEX_2D(ib, ib + 2)] = -0.20036969;
            Q[INDEX_2D(ib, ib + 3)] = 0.038253676;
            Q[INDEX_2D(ib, ib + 4)] = -0.0035978349;
            ib += 1;
        } break;

            // NOTE: in original initcfd.c file from David Neilsen, this was
            // repeated in the if statement, but in an elif, so it's unreachable
            // anyway since this value is handled in the same way above case
            // CFD_P1_O4_L4_CLOSE: ...

        default:
            throw std::invalid_argument(
                "Unknown derivative type for initializing CFD matrices! "
                "dtype=" +
                std::to_string(dtype));
            break;
    }
    // update xib
    *xib = ib;
}

void buildMatrixRight(double *P, double *Q, int *xie, const DerType dtype,
                      const int nghosts, const int n) {
    int ie = n - 1;

    switch (dtype) {
        case CFD_DRCHLT_ORDER_4: {
            P[INDEX_2D(n - 1, n - 1)] = 1.0;
            P[INDEX_2D(n - 1, n - 2)] = 3.0;

            Q[INDEX_2D(n - 1, n - 1)] = 17.0 / 6.0;
            Q[INDEX_2D(n - 1, n - 2)] = -3.0 / 2.0;
            Q[INDEX_2D(n - 1, n - 3)] = -3.0 / 2.0;
            Q[INDEX_2D(n - 1, n - 4)] = 1.0 / 6.0;
            ie = n - 2;
        } break;

        case CFD_DRCHLT_ORDER_6: {
            P[INDEX_2D(n - 1, n - 1)] = 1.0;
            P[INDEX_2D(n - 1, n - 2)] = 5.0;

            P[INDEX_2D(n - 2, n - 1)] = 2.0 / 11.0;
            P[INDEX_2D(n - 2, n - 2)] = 1.0;
            P[INDEX_2D(n - 2, n - 3)] = 2.0 / 11.0;

            Q[INDEX_2D(n - 1, n - 1)] = 197.0 / 60.0;
            Q[INDEX_2D(n - 1, n - 2)] = 5.0 / 12.0;
            Q[INDEX_2D(n - 1, n - 3)] = -5.0;
            Q[INDEX_2D(n - 1, n - 4)] = 5.0 / 3.0;
            Q[INDEX_2D(n - 1, n - 5)] = -5.0 / 12.0;
            Q[INDEX_2D(n - 1, n - 6)] = 1.0 / 20.0;

            Q[INDEX_2D(n - 2, n - 1)] = 20.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 2)] = 35.0 / 132.0;
            Q[INDEX_2D(n - 2, n - 3)] = -34.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 4)] = 7.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 5)] = -2.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 6)] = 1.0 / 132.0;
            ie = n - 3;
        } break;

        case CFD_DRCHLT_Q6: {
            P[INDEX_2D(n - 1, n - 1)] = 1.0;
            P[INDEX_2D(n - 1, n - 2)] = 5.0;

            P[INDEX_2D(n - 2, n - 1)] = 2.0 / 11.0;
            P[INDEX_2D(n - 2, n - 2)] = 1.0;
            P[INDEX_2D(n - 2, n - 3)] = 2.0 / 11.0;

            P[INDEX_2D(n - 3, n - 2)] = 1.0 / 3.0;
            P[INDEX_2D(n - 3, n - 3)] = 1.0;
            P[INDEX_2D(n - 3, n - 4)] = 1.0 / 3.0;

            Q[INDEX_2D(n - 1, n - 1)] = 197.0 / 60.0;
            Q[INDEX_2D(n - 1, n - 2)] = 5.0 / 12.0;
            Q[INDEX_2D(n - 1, n - 3)] = -5.0;
            Q[INDEX_2D(n - 1, n - 4)] = 5.0 / 3.0;
            Q[INDEX_2D(n - 1, n - 5)] = -5.0 / 12.0;
            Q[INDEX_2D(n - 1, n - 6)] = 1.0 / 20.0;

            Q[INDEX_2D(n - 2, n - 1)] = 20.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 2)] = 35.0 / 132.0;
            Q[INDEX_2D(n - 2, n - 3)] = -34.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 4)] = 7.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 5)] = -2.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 6)] = 1.0 / 132.0;

            Q[INDEX_2D(n - 3, n - 1)] = 1.0 / 36.0;
            Q[INDEX_2D(n - 3, n - 2)] = 14.0 / 18.0;
            Q[INDEX_2D(n - 3, n - 3)] = 0.0;
            Q[INDEX_2D(n - 3, n - 4)] = -14.0 / 18.0;
            Q[INDEX_2D(n - 3, n - 5)] = -1.0 / 36.0;

            ie = n - 4;
        } break;

        case CFD_P1_O4_CLOSE: {
            if (nghosts < 3) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "3! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = n - nghosts; i < n; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ie = n - nghosts - 1;
            P[INDEX_2D(ie, ie)] = 1.0;

            const double t1 = 1.0 / 72.0;
            Q[INDEX_2D(ie, ie - 3)] = -t1;
            Q[INDEX_2D(ie, ie - 2)] = 10.0 * t1;
            Q[INDEX_2D(ie, ie - 1)] = -53.0 * t1;
            Q[INDEX_2D(ie, ie)] = 0.0;
            Q[INDEX_2D(ie, ie + 1)] = 53.0 * t1;
            Q[INDEX_2D(ie, ie + 2)] = -10.0 * t1;
            Q[INDEX_2D(ie, ie + 3)] = t1;
            ie -= 1;
        } break;

        case CFD_P1_O6_CLOSE: {
            if (nghosts < 4) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "4! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = n - nghosts; i < n; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ie = n - nghosts - 1;
            P[INDEX_2D(ie, ie)] = 1.0;

            const double t2 = 1.0 / 300.0;
            Q[INDEX_2D(ie, ie - 4)] = t2;
            Q[INDEX_2D(ie, ie - 3)] = -11.0 * t2;
            Q[INDEX_2D(ie, ie - 2)] = 59.0 * t2;
            Q[INDEX_2D(ie, ie - 1)] = -239.0 * t2;
            Q[INDEX_2D(ie, ie)] = 0.0;
            Q[INDEX_2D(ie, ie + 1)] = 239.0 * t2;
            Q[INDEX_2D(ie, ie + 2)] = -59.0 * t2;
            Q[INDEX_2D(ie, ie + 3)] = 11.0 * t2;
            Q[INDEX_2D(ie, ie + 4)] = -t2;
            ie -= 1;
        } break;

        case CFD_P1_O4_L4_CLOSE: {
            if (nghosts < 1) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "1! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = n - nghosts; i < n; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ie = n - nghosts - 1;
            P[INDEX_2D(ie, ie)] = 1.0;

            const double t3 = 1.0 / 12.0;
            Q[INDEX_2D(ie, ie + 1)] = 3.0 * t3;
            Q[INDEX_2D(ie, ie)] = 10.0 * t3;
            Q[INDEX_2D(ie, ie - 1)] = -18.0 * t3;
            Q[INDEX_2D(ie, ie - 2)] = 6.0 * t3;
            Q[INDEX_2D(ie, ie - 3)] = -t3;

            ie -= 1;
        } break;

        case CFD_P1_O6_L6_CLOSE: {
            if (nghosts < 2) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "2! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = n - nghosts; i < n; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ie = n - nghosts - 1;
            const double t4 = 1.0 / 60.0;
            P[INDEX_2D(ie, ie)] = 1.0;

            Q[INDEX_2D(ie, ie + 2)] = -2.0 * t4;
            Q[INDEX_2D(ie, ie + 1)] = 24.0 * t4;
            Q[INDEX_2D(ie, ie)] = 35.0 * t4;
            Q[INDEX_2D(ie, ie - 1)] = -80.0 * t4;
            Q[INDEX_2D(ie, ie - 2)] = 30.0 * t4;
            Q[INDEX_2D(ie, ie - 3)] = -8.0 * t4;
            Q[INDEX_2D(ie, ie - 4)] = 1.0 * t4;

            ie -= 1;
        } break;

        case CFD_Q1_O6_ETA1_CLOSE: {
            if (nghosts < 4) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at "
                    "least "
                    "4! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = n - nghosts; i < n; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ie = n - nghosts - 1;
            P[INDEX_2D(ie, ie)] = 1.0;

            Q[INDEX_2D(ie, ie - 4)] = 0.0035978349;
            Q[INDEX_2D(ie, ie - 3)] = -0.038253676;
            Q[INDEX_2D(ie, ie - 2)] = 0.20036969;
            Q[INDEX_2D(ie, ie - 1)] = -0.80036969;
            Q[INDEX_2D(ie, ie)] = 0.0;
            Q[INDEX_2D(ie, ie + 1)] = 0.80036969;
            Q[INDEX_2D(ie, ie + 2)] = -0.20036969;
            Q[INDEX_2D(ie, ie + 3)] = 0.038253676;
            Q[INDEX_2D(ie, ie + 4)] = -0.0035978349;
            ie -= 1;
        } break;

        default:
            break;
    }
    // update xib
    *xie = ie;
}

void print_square_mat(double *m, const uint32_t n) {
    // assumes "col" order in memory
    // J is the row!
    for (uint16_t i = 0; i < n; i++) {
        printf("%3d : ", i);
        // I is the column!
        for (uint16_t j = 0; j < n; j++) {
            printf("%8.3f ", m[INDEX_2D(i, j)]);
        }
        printf("\n");
    }
}

}  // namespace dendro_cfd