#ifndef SIMULATION_N_BODY_CUDA_AOS_CUH
#define SIMULATION_N_BODY_CUDA_AOS_CUH

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyCudaAoS : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyCudaAoS(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyCudaAoS() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_CUDA_AOS_CUH */
