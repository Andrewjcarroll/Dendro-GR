#include <iostream>
#include <tuple>
#include <vector>

#include "cfd.h"
#include "derivs.h"

#define UNIFORM_RAND_0_TO_X(X) ((double_t)rand() / (double_t)RAND_MAX * X)

namespace helpers {
uint32_t padding;
}

void sine_init(double_t *u_var, const uint32_t *sz, const double_t *deltas) {
    const double_t x_start = 0.0;
    const double_t y_start = 0.0;
    const double_t z_start = 0.0;
    const double_t dx = deltas[0];
    const double_t dy = deltas[1];
    const double_t dz = deltas[2];

    const double_t amplitude = 0.01;

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    for (uint16_t k = 0; k < nz; k++) {
        double_t z = z_start + k * dz;
        for (uint16_t j = 0; j < ny; j++) {
            double_t y = y_start + j * dy;
            for (uint16_t i = 0; i < nx; i++) {
                double x = x_start + i * dx;
                u_var[IDX(i, j, k)] = 1.0 * sin(2 * x + 0.1) +
                                      2.0 * sin(3 * y - 0.1) +
                                      0.5 * sin(0.5 * z);
            }
        }
    }
}

void random_init(double_t *u_var, const uint32_t *sz) {
    const double_t amplitude = 0.001;
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    for (uint16_t k = 0; k < nz; k++) {
        for (uint16_t j = 0; j < ny; j++) {
            for (uint16_t i = 0; i < nx; i++) {
                u_var[IDX(i, j, k)] =
                    amplitude * (UNIFORM_RAND_0_TO_X(2) - 1.0);
            }
        }
    }
}

void zero_init(double_t *u_var, const uint32_t *sz) {
    const double_t amplitude = 0.001;
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    for (uint16_t k = 0; k < nz; k++) {
        for (uint16_t j = 0; j < ny; j++) {
            for (uint16_t i = 0; i < nx; i++) {
                u_var[IDX(i, j, k)] = 0.0;
            }
        }
    }
}

void init_data(const uint32_t init_type, double_t *u_var, const uint32_t *sz,
               const double *deltas) {
    switch (init_type) {
        case 0:
            zero_init(u_var, sz);
            break;

        case 1:
            random_init(u_var, sz);
            break;

        case 2:
            sine_init(u_var, sz, deltas);
            break;

        default:
            std::cout << "UNRECOGNIZED INITIAL DATA FUNCTION... EXITING"
                      << std::endl;
            exit(0);
            break;
    }
}

void print_3d_mat(double_t *u_var, const uint32_t *sz) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];
    for (uint16_t k = 0; k < nz; k++) {
        for (uint16_t j = 0; j < ny; j++) {
            for (uint16_t i = 0; i < nx; i++) {
                printf("%f ", u_var[IDX(i, j, k)]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
}

// void print_square_mat(double *m, const uint32_t n) {
//     // assumes "col" order in memory
//     // J is the row!
//     for (uint16_t i = 0; i < n; i++) {
//         printf("%3d : ", i);
//         // I is the column!
//         for (uint16_t j = 0; j < n; j++) {
//             printf("%8.3f ", m[INDEX_2D(i, j)]);
//         }
//         printf("\n");
//     }
// }

void print_square_mat_flat(double *m, const uint32_t n) {
    uint16_t j_count = 0;
    for (uint16_t i = 0; i < n * n; i++) {
        if (i % n == 0) {
            j_count++;
            printf("\n");
        }
        printf("%8.3f ", m[i]);
    }
}

std::tuple<double_t, double_t, double_t> calculate_mse(
    double_t *const x, double_t *const y, const uint32_t *sz,
    bool skip_pading = true) {
    // required for IDX function...
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    double_t max_err = 0.0;
    double_t min_err = __DBL_MAX__;
    double_t mse = 0.0;

    const uint32_t i_start = skip_pading ? helpers::padding : 0;
    const uint32_t j_start = skip_pading ? helpers::padding : 0;
    const uint32_t k_start = skip_pading ? helpers::padding : 0;

    const uint32_t i_end = skip_pading ? sz[0] - helpers::padding : sz[0];
    const uint32_t j_end = skip_pading ? sz[1] - helpers::padding : sz[1];
    const uint32_t k_end = skip_pading ? sz[2] - helpers::padding : sz[2];

    std::cout << i_start << " " << i_end << std::endl;

    const uint32_t total_points =
        (i_end - i_start) * (j_end - j_start) * (k_end - k_start);

    std::cout << total_points << std::endl;

    for (uint16_t k = k_start; k < k_end; k++) {
        for (uint16_t j = j_start; j < j_end; j++) {
            for (uint16_t i = i_start; i < i_end; i++) {
                double_t temp = (x[IDX(i, j, k)] - y[IDX(i, j, k)]) *
                                (x[IDX(i, j, k)] - y[IDX(i, j, k)]);

                if (temp > max_err) {
                    max_err = temp;
                }
                if (temp < min_err) {
                    min_err = temp;
                }

                mse += temp;
            }
        }
    }

    mse /= (total_points);

    return std::make_tuple(mse, min_err, max_err);
}

void test_cfd_with_original_stencil(double_t *const u_var, const uint32_t *sz,
                                    const double *deltas,
                                    dendro_cfd::CompactFiniteDiff *cfd) {
    // allocate a double block of memory
    const uint32_t totalSize = sz[0] * sz[1] * sz[2];
    double_t *deriv_workspace = new double_t[totalSize * 3 * 2];

    double_t *const derivx_stencil = deriv_workspace + 0 * totalSize;
    double_t *const derivy_stencil = deriv_workspace + 1 * totalSize;
    double_t *const derivz_stencil = deriv_workspace + 2 * totalSize;

    double_t *const derivx_cfd = deriv_workspace + 3 * totalSize;
    double_t *const derivy_cfd = deriv_workspace + 4 * totalSize;
    double_t *const derivz_cfd = deriv_workspace + 5 * totalSize;

    // then compute!
    deriv_x(derivx_stencil, u_var, deltas[0], sz, 0);
    deriv_y(derivy_stencil, u_var, deltas[1], sz, 0);
    deriv_z(derivz_stencil, u_var, deltas[2], sz, 0);

    cfd->cfd_x(derivx_cfd, u_var, deltas[0], sz, 0);
    cfd->cfd_y(derivy_cfd, u_var, deltas[1], sz, 0);
    cfd->cfd_z(derivz_cfd, u_var, deltas[2], sz, 0);

    // then compute the "error" difference between the two
    double_t min_x, max_x, mse_x;
    std::tie(mse_x, min_x, max_x) =
        calculate_mse(derivx_stencil, derivx_cfd, sz);

    double_t min_y, max_y, mse_y;
    std::tie(mse_y, min_y, max_y) =
        calculate_mse(derivy_stencil, derivy_cfd, sz);

    double_t min_z, max_z, mse_z;
    std::tie(mse_z, min_z, max_z) =
        calculate_mse(derivz_stencil, derivz_cfd, sz);

    std::cout << GRN << "===COMPARING CFD TO STENCIL TEST RESULTS===" << NRM
              << std::endl;
    std::cout << "   deriv_x : mse = \t" << mse_x << "\tmin = \t" << min_x
              << "\tmax = \t" << max_x << std::endl;
    std::cout << "   deriv_y : mse = \t" << mse_y << "\tmin = \t" << min_y
              << "\tmax = \t" << max_y << std::endl;
    std::cout << "   deriv_z : mse = \t" << mse_z << "\tmin = \t" << min_z
              << "\tmax = \t" << max_z << std::endl;

    delete[] deriv_workspace;
}

int main(int argc, char **argv) {
    uint32_t eleorder = 8;
    uint32_t deriv_type = 1;
    uint32_t filter_type = 0;
    uint32_t num_tests = 1000;
    uint32_t data_init = 1;

    if (argc == 1) {
        std::cout << "Using default parameters." << std::endl;

        std::cout << "If you wish to change the default parameters pass them "
                     "as command line arguments:"
                  << std::endl;
        std::cout
            << "<eleorder> <deriv_type> <filter_type> <num_tests> <data_init>"
            << std::endl;
    }

    if (argc > 1) {
        // read in the element order
        eleorder = atoi(argv[1]);
    }
    if (argc > 2) {
        // read in the deriv_type we want to use
        // if it's set to 0, we'll do the default derivatives
        deriv_type = atoi(argv[2]);
    }
    if (argc > 3) {
        filter_type = atoi(argv[3]);
    }
    if (argc > 4) {
        num_tests = atoi(argv[4]);
    }
    if (argc > 5) {
        data_init = atoi(argv[5]);
    }
    helpers::padding = eleorder >> 1u;

    std::cout << YLW
              << "Will run with the following user parameters:" << std::endl;
    std::cout << "    eleorder    -> " << eleorder << std::endl;
    std::cout << "    deriv_type  -> " << deriv_type << std::endl;
    std::cout << "    filter_type -> " << filter_type << std::endl;
    std::cout << "    num_tests   -> " << num_tests << std::endl;
    std::cout << "    data_init   -> " << data_init << std::endl;
    std::cout << "    INFO: padding is " << helpers::padding << NRM
              << std::endl;

    // the size in each dimension
    uint32_t fullwidth = 2 * eleorder + 1;
    uint32_t sz[3] = {fullwidth, fullwidth, fullwidth};

    // now we can actually build up our test block

    double_t *u_var = new double_t[sz[0] * sz[1] * sz[2]];

    double_t deltas[3] = {0.001, 0.001, 0.001};

    init_data(data_init, u_var, sz, deltas);

    // print_3d_mat(u_var, fullwidth, fullwidth, fullwidth);

    // build up the cfd object
    dendro_cfd::CompactFiniteDiff cfd(fullwidth, helpers::padding, deriv_type,
                                      filter_type);

    // run a short test to see what the errors are
    test_cfd_with_original_stencil((double_t *const)u_var, sz, deltas, &cfd);

    double *P = new double[fullwidth * fullwidth]();
    double *Q = new double[fullwidth * fullwidth]();

    dendro_cfd::buildPandQMatrices(P, Q, helpers::padding, fullwidth,
                              dendro_cfd::CFD_Q1_O6_ETA1, true, true);

    std::cout << "P matrix: " << std::endl;
    dendro_cfd::print_square_mat(P, fullwidth);

    // print_square_mat_flat(P, fullwidth);

    std::cout << std::endl << "Q matrix: " << std::endl;
    dendro_cfd::print_square_mat(Q, fullwidth);


    double *Dmat = new double[fullwidth * fullwidth]();

    dendro_cfd::calculateDerivMatrix(Dmat, P, Q, fullwidth);

    std::cout << std::endl << "Dmat matrix: " << std::endl;
    dendro_cfd::print_square_mat(Dmat, fullwidth);

    delete[] P;
    delete[] Q;
    delete[] Dmat;
    // var cleanup
    delete[] u_var;
}
