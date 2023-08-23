#include "cfd.h"

namespace dendro_cfd {

// initialize a "global" cfd object
CompactFiniteDiff cfd(0);

CompactFiniteDiff::CompactFiniteDiff(const unsigned int num_dim,
                                     const unsigned int deriv_type,
                                     const unsigned int filter_type) {
    m_deriv_type = deriv_type;
    m_filter_type = filter_type;
    m_curr_dim_size = num_dim;

    initialize_cfd_storage();

    if (num_dim == 0) {
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

        initialize_cfd_matrix();
        initialize_cfd_filter();
    }
}

void CompactFiniteDiff::initialize_cfd_storage() {
    m_RF = new double[m_curr_dim_size * m_curr_dim_size];
    m_R = new double[m_curr_dim_size * m_curr_dim_size];
    m_u1d = new double[m_curr_dim_size];
    m_du1d = new double[m_curr_dim_size];
}

void CompactFiniteDiff::initialize_cfd_matrix() {
    switch (m_deriv_type) {
        case 0:
            break;
        case 1:
            if (initKimDeriv4(m_R, m_curr_dim_size)) {
                // failed to initialize...
                std::cerr << RED
                          << "ERROR: could not construct the Kim Deriv4 matrix!"
                          << std::endl;
                exit(0);
            }
            break;
        case 2:
            if (initHAMRDeriv4(m_R, m_curr_dim_size)) {
                // failed to initialize
                std::cerr
                    << RED
                    << "ERROR: could not construct the HAMR Deriv4 matrix!"
                    << std::endl;
                exit(0);
            }
            break;
        case 3:
            if (initJTPDeriv6(m_R, m_curr_dim_size)) {
                // failed to initialize again
                std::cerr << RED
                          << "ERROR: could not construct the JTP Deriv6 matrix!"
                          << std::endl;
                exit(0);
            }

        default:
            // do exit
            std::cerr << RED << "ERROR: Unknown CFD derivative type of "
                      << m_deriv_type << ". Exiting..." << std::endl;
            exit(0);
            break;
    }

    // // Print the R matrix
    // MPI_Comm comm = MPI_COMM_WORLD;

    // int rank, npes;
    // MPI_Comm_rank(comm, &rank);
    // MPI_Comm_size(comm, &npes);

    // if (rank == 0) {
    //     printf("R matrix:\n");
    //     for (unsigned int k = 0; k < m_curr_dim_size; k++) {
    //         for (unsigned int m = 0; m < m_curr_dim_size; m++) {
    //             printf("%f ", m_R[k * m_curr_dim_size + m]);
    //         }
    //         printf("\n");
    //     }
    // }
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

    // left boundary to watch for
    // if (bflag & (1u << OCT_DIR_LEFT)) ;

    // // right boundary to watch for
    // if (bflag & (1u << OCT_DIR_RIGHT)) ;

    for (unsigned int k = 0; k < nz; k++) {
        for (unsigned int j = 0; j < ny; j++) {
            for (unsigned int i = 0; i < nx; i++) {
                m_u1d[i] = u[IDX(i, j, k)];
            }

            for (unsigned int i = 0; i < nx; i++) {
                m_du1d[i] = 0.0;
                for (unsigned int m = 0; m < nx; m++) {
                    m_du1d[i] += m_R[i * nx + m] * m_u1d[m];
                }
            }

            for (unsigned int i = 0; i < nx; i++) {
                Dxu[IDX(i, j, k)] = m_du1d[i] / dx;
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

    // if (bflag & (1u << OCT_DIR_DOWN));

    // if (bflag & (1u << OCT_DIR_UP));

    for (unsigned int k = 0; k < nz; k++) {
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int j = 0; j < ny; j++) {
                m_u1d[j] = u[IDX(i, j, k)];
            }
            for (unsigned int j = 0; j < ny; j++) {
                m_du1d[j] = 0.0;
                for (unsigned int m = 0; m < ny; m++) {
                    m_du1d[j] += m_R[j * ny + m] * m_u1d[m];
                }
            }
            for (unsigned int j = 0; j < ny; j++) {
                Dyu[IDX(i, j, k)] = m_du1d[j] / dy;
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

    // if (bflag & (1u << OCT_DIR_BACK)) ;

    // if (bflag & (1u << OCT_DIR_FRONT)) ;

    for (unsigned int j = 0; j < ny; j++) {
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                m_u1d[k] = u[IDX(i, j, k)];
            }
            for (unsigned int k = 0; k < nz; k++) {
                m_du1d[k] = 0.0;
                for (unsigned int m = 0; m < nz; m++) {
                    m_du1d[k] += m_R[k * nz + m] * m_u1d[m];
                }
            }
            for (unsigned int k = 0; k < nz; k++) {
                Dzu[IDX(i, j, k)] = m_du1d[k] / dz;
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
                m_u1d[i] = u[IDX(i, j, k)];
            }

            for (unsigned int i = 0; i < nx; i++) {
                m_du1d[i] = 0.0;
                for (unsigned int m = 0; m < nx; m++) {
                    m_du1d[i] += m_RF[i * nx + m] * m_u1d[m];
                }
            }

            for (unsigned int i = 0; i < nx; i++) {
                u[IDX(i, j, k)] +=
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
                m_u1d[j] = u[IDX(i, j, k)];
            }
            for (unsigned int j = 0; j < ny; j++) {
                m_du1d[j] = 0.0;
                for (unsigned int m = 0; m < ny; m++) {
                    m_du1d[j] += m_RF[j * ny + m] * m_u1d[m];
                }
            }
            for (unsigned int j = 0; j < ny; j++) {
                u[IDX(i, j, k)] +=
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
                m_u1d[k] = u[IDX(i, j, k)];
            }
            for (unsigned int k = 0; k < nz; k++) {
                m_du1d[k] = 0.0;
                for (unsigned int m = 0; m < nz; m++) {
                    m_du1d[k] += m_RF[k * nz + m] * m_u1d[m];
                }
            }
            for (unsigned int k = 0; k < nz; k++) {
                u[IDX(i, j, k)] +=
                    m_du1d[k];  // do we need to include a dz here?
            }
        }
    }
}

}  // namespace dendro_cfd