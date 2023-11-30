//
// Created by milinda on 12/1/17.
/**
 *@author Milinda Fernando
 *School of Computing, University of Utah
 *@brief rk4 solver for em3 equations.
 */
//

#ifndef SFCSORTBENCH_RK4EM3_H
#define SFCSORTBENCH_RK4EM3_H

#include <iostream>
#include <string>

#include "checkPoint.h"
#include "compact_derivs.h"
#include "em3.h"
#include "em3Utils.h"
#include "fdCoefficient.h"
#include "mathMeshUtils.h"
#include "mesh.h"
#include "meshTestUtils.h"
#include "oct2vtk.h"
#include "parameters.h"
#include "physcon.h"
#include "rhs.h"
#include "rk.h"

static const double RK4_C[] = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};

static const double RK4_T[] = {0, 1.0 / 2.0, 1.0 / 2.0, 1.0};
static const double RK4_U[] = {0.0, 1.0 / 2.0, 1.0 / 2.0, 1.0};

namespace ode {
namespace solver {

class RK4_EM3 : public RK {
   private:
    // variables for EM3 formulation.

    /**@brief: list of pointers to the variable set indexed by enum VAR */
    double **m_uiVar;

    /**@brief: previous time step solution*/
    double **m_uiPrevVar;

    /**@brief: intermediate variable for RK*/
    double **m_uiVarIm;

    /**@brief list of pointers to unzip version of the variables **/
    double **m_uiUnzipVar;

    /**@brief unzip rhs for each variable.*/
    double **m_uiUnzipVarRHS;

    /** stage - value vector of RK45 method*/
    double ***m_uiStage;

    /** @brief zipped version physical constraint equations*/
    double **m_uiConstraintVars;

    /** @brief unzip physical constrint vars*/
    double **m_uiUnzipConstraintVars;

    /**Send node buffers for async (send) communication*/
    double **m_uiSendNodeBuf;

    /**recv node buffers for async (receive) communciation*/
    double **m_uiRecvNodeBuf;

    /**@brief mpi send reqs for evolution vars*/
    MPI_Request **m_uiSendReqs;

    /**@brief mpi recv reqs for evolution vars*/
    MPI_Request **m_uiRecvReqs;

    /**@brief mpi send status to sync on sends*/
    MPI_Status **m_uiSendSts;

    /**@brief mpi recv status to sync on recv*/
    MPI_Status **m_uiRecvSts;

   public:
    /**
     * @brief default constructor
     * @param[in] pMesh : pointer to the mesh.
     * @param[in] pTBegin: RK45 time begin
     * @param[in] pTEnd: RK45 time end
     * @param[in] pTh: times step size.
     */
    RK4_EM3(ot::Mesh *pMesh, double pTBegin, double pTEnd, double pTh);

    /**@brief default destructor*/
    ~RK4_EM3();

    /**
     * @brief: read parameters related to EM3 simulation and store them in
     * static variables defined in parameters.h
     */
    void readConfigFile(const char *fName);

    /**
     * @brief: starts the rk-45 solver.
     */
    void rkSolve();

    /**
     * @brief: restore rk45 solver from a given checkpoint. This will overwrite
     * the parameters given in the original constructor
     * @param[in]fNamePrefix: checkpoint file pre-fix name.
     * @param[in]step: step number which needs to be restored.
     * @param[in]comm: MPI communicator.
     */
    void restoreCheckPoint(const char *fNamePrefix, MPI_Comm comm);

   private:
    /**
     * @brief Applies initial conditions.
     * @param[in] zipIn Array of pointers to variables.
     */
    void applyInitialConditions(double **zipIn);

    /**
     * @brief Performs initial grid convergence until the mesh converges to
     * initial data.
     */
    void initialGridConverge();

    /**
     * @brief Reallocates MPI resources if the mesh is changed (needs to be
     * called during mesh refinement).
     */
    void reallocateMPIResources();

    /**
     * @brief Performs ghost exchange for all variables.
     * @param[in] zipIn Array of pointers to variables.
     */
    void performGhostExchangeVars(double **zipIn);

    /**
     * @brief Performs the intergrid transfer.
     * @param[in,out] zipIn Array of pointers to variables.
     * @param[in] pnewMesh Pointer to the new mesh.
     */
    void intergridTransferVars(double **&zipIn, const ot::Mesh *pnewMesh);

    /**
     * @brief Unzips all the variables specified in VARS.
     * @param[in] zipIn Array of pointers to zipped variables.
     * @param[out] uzipOut Array of pointers to unzipped variables.
     */
    void unzipVars(double **zipIn, double **uzipOut);

    /**
     * @brief Asynchronously unzips all the variables specified in VARS.
     * @param[in] zipIn Array of pointers to zipped variables.
     * @param[out] uzipOut Array of pointers to unzipped variables.
     */
    void unzipVars_async(double **zipIn, double **uzipOut);

    /**
     * @brief Zips all the variables specified in VARS.
     * @param[in] uzipIn Array of pointers to unzipped variables.
     * @param[out] zipOut Array of pointers to zipped variables.
     */
    void zipVars(double **uzipIn, double **zipOut);

    /**
     * @brief Writes the solution to a VTU file.
     * @param[in] evolZipVarIn Array of pointers to zipped evolved variables.
     * @param[in] constrZipVarIn Array of pointers to zipped constrained vars.
     * @param[in] numEvolVars Number of evolved variables.
     * @param[in] numConstVars Number of constrained variables.
     * @param[in] evolVarIndices Array of indices for evolved variables.
     * @param[in] constVarIndices Array of indices for constrained variables.
     * @param[in] zslice Flag indicating whether to only use z-slice..
     */
    void writeToVTU(double **evolZipVarIn, double **constrZipVarIn,
                    unsigned int numEvolVars, unsigned int numConstVars,
                    const unsigned int *evolVarIndices,
                    const unsigned int *constVarIndices, bool zslice);

    /**
     * @brief Implementation of the base class time step function.
     */
    void performSingleIteration();

    /**
     * @brief Implementation of the base class function to apply boundary
     * conditions.
     */
    void applyBoundaryConditions();

    /** @brief: Applys a filter to the evolution variables.
     */
    void applyFilter();

    /**
     * @brief Stores variables required to restore the RK45 solver at a given
     * stage.
     * @param[in] fNamePrefix Checkpoint file prefix name.
     */
    void storeCheckPoint(const char *fNamePrefix);
};

}  // end of namespace solver
}  // end of namespace ode

#endif  // SFCSORTBENCH_RK4EM3_H
