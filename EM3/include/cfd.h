#pragma once

#include <cmath>

#include "HAMR.h"
#include "JTP6.h"
#include "dendro.h"
#include "kim.h"
#include "kim_filter.h"

#define IDX(i, j, k) ((i) + nx * ((j) + ny * (k)))

namespace dendro_cfd {

class CompactFiniteDiff {
   private:
    // STORAGE VARIABLES USED FOR THE DIFFERENT DIMENSIONS
    double *m_RF = nullptr;
    double *m_R = nullptr;
    double *m_u1d = nullptr;
    double *m_du1d = nullptr;

    // to check for initialization (not used)
    bool m_initialized_matrices = false;

    // storing the derivative and filter types internally
    // could just be the parameter types
    unsigned int m_deriv_type = 0;
    unsigned int m_filter_type = 0;
    unsigned int m_curr_dim_size = 0;

   public:
    CompactFiniteDiff(const unsigned int dim_size,
                        const unsigned int deriv_type = 0,
                        const unsigned int filter_type = 0);
    ~CompactFiniteDiff();

    void change_dim_size(const unsigned int dim_size);

    void initialize_cfd_storage();
    void initialize_cfd_matrix();
    void initialize_cfd_filter();
    void delete_cfd_matrices();

    void set_filter_type(const unsigned int filter_type) {
        m_filter_type = filter_type;
    }

    void set_deriv_type(const unsigned int deriv_type) {
        m_deriv_type = deriv_type;
    }

    // the actual derivative computation side of things
    void cfd_x(double *const Dxu, const double *const u, const double dx,
               const unsigned int *sz, unsigned bflag);
    void cfd_y(double *const Dyu, const double *const u, const double dy,
               const unsigned int *sz, unsigned bflag);
    void cfd_z(double *const Dzu, const double *const u, const double dz,
               const unsigned int *sz, unsigned bflag);

    // then the actual filters
    void filter_cfd_x(double *const u, const double dx, const unsigned int *sz,
                      unsigned bflag);
    void filter_cfd_y(double *const u, const double dx, const unsigned int *sz,
                      unsigned bflag);
    void filter_cfd_z(double *const u, const double dz, const unsigned int *sz,
                      unsigned bflag);
};

extern CompactFiniteDiff cfd;

}  // namespace dendro_cfd