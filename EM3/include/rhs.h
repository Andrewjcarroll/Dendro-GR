#ifndef RHS_H
#define RHS_H

#include <time.h>

#include <cmath>
#include <iostream>

#include "compact_derivs.h"
#include "derivs.h"
#include "mathUtils.h"
#include "parameters.h"
#include "profile_params.h"

#ifdef EM3_ENABLE_CUDA
#include "params_cu.h"
#include "profile_gpu.h"
#include "rhs_cuda.cuh"
#endif

/**
 * @brief Computes the right-hand side (RHS) of the evolution equations.
 *
 * This function calculates the RHS of the evolution equations based on the
 * given zipped variables, block information, and the number of blocks.
 *
 * @param[out] uzipVarsRHS Array of pointers to store the computed RHS.
 * @param[in] uZipVars Array of pointers to unzipped variables.
 * @param[in] blkList Array of Block objects for computing.
 * @param[in] numBlocks Number of computational blocks.
 */
void em3rhs(double **uzipVarsRHS, const double **uZipVars,
            const ot::Block *blkList, unsigned int numBlocks);

/**
 * @brief Computes the right-hand side (RHS) of the evolution equations.
 *
 * This function calculates the RHS of the evolution equations based on the
 * given zipped variables, offset, bounding box information, grid size, and
 * boundary flag.
 *
 * @param[out] uzipVarsRHS Array of pointers to store the computed RHS.
 * @param[in] uZipVars Array of pointers to unzipped variables.
 * @param[in] offset Offset for accessing arrays.
 * @param[in] ptmin Array representing the minimum bounding box coordinates.
 * @param[in] ptmax Array representing the maximum bounding box coordinates.
 * @param[in] sz Array representing the grid size in each dimension.
 * @param[in] bflag Boundary flag indicating which boundaries are included.
 */
void em3rhs(double **uzipVarsRHS, const double **uZipVars,
            const unsigned int &offset, const double *ptmin,
            const double *ptmax, const unsigned int *sz,
            const unsigned int &bflag);

/**
 * @brief Applies boundary conditions to the electromagnetic (EM) field.
 *
 * This function applies boundary conditions to the EM vars based on the current
 * data, grid derivatives, bounding box, falloff parameter, asymptotic
 * parameter, grid size, and boundary flag.
 *
 * @param[out] f_rhs Array to store the updated RHS.
 * @param[in] f Array representing the variable as it comes in.
 * @param[in] dxf Array representing the derivative in the x-direction.
 * @param[in] dyf Array representing the derivative in the y-direction.
 * @param[in] dzf Array representing the derivative in the z-direction.
 * @param[in] pmin Array representing the minimum bounding box coordinates.
 * @param[in] pmax Array representing the maximum bounding box coordinates.
 * @param[in] f_falloff Falloff parameter for boundary conditions.
 * @param[in] f_asymptotic Asymptotic value for boundary conditions.
 * @param[in] sz Array representing the grid size in each dimension.
 * @param[in] bflag Boundary flag indicating which boundaries are included.
 */
void em3_bcs(double *f_rhs, const double *f, const double *dxf,
             const double *dyf, const double *dzf, const double *pmin,
             const double *pmax, const double f_falloff,
             const double f_asymptotic, const unsigned int *sz,
             const unsigned int &bflag);

/**
 * @brief Computes the right-hand side (RHS) of the evolution equations.
 *
 * This function calculates the RHS of the evolution equations based on the
 * given zipped variables, offset, bounding box information, grid size, and
 * boundary flag. However, this time it includes the logic for Compact Finite
 * Differences.
 *
 * @param[out] uzipVarsRHS Array of pointers to store the computed RHS.
 * @param[in] uZipVars Array of pointers to unzipped variables.
 * @param[in] offset Offset for accessing arrays.
 * @param[in] ptmin Array representing the minimum bounding box coordinates.
 * @param[in] ptmax Array representing the maximum bounding box coordinates.
 * @param[in] sz Array representing the grid size in each dimension.
 * @param[in] bflag Boundary flag indicating which boundaries are included.
 */
void em3rhs_CFD(double **unzipVarsRHS, double **uZipVars,
                const unsigned int &offset, const double *pmin,
                const double *pmax, const unsigned int *sz,
                const unsigned int &bflag);
/**
 * @brief Applies filters to the field variables.
 *
 * This function applies filters to the field variables based on the given
 * zipped variables, offset, bounding box information, grid size, and boundary
 * flag.
 *
 * @param[in,out] uZipVars Array of pointers to zipped variables (input and
 * output).
 * @param[in] offset Offset for accessing arrays.
 * @param[in] pmin Array representing the minimum bounding box coordinates.
 * @param[in] pmax Array representing the maximum bounding box coordinates.
 * @param[in] sz Array representing the grid size in each dimension.
 * @param[in] bflag Boundary flag indicating which boundaries are included.
 */
void apply_filters(double **uZipVars, const unsigned int &offset,
                   const double *pmin, const double *pmax,
                   const unsigned int *sz, const unsigned int &bflag);

#endif
