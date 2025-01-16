#ifndef SIMULATION_N_BODY_CUDA_SOA_CUH
#define SIMULATION_N_BODY_CUDA_SOA_CUH

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyCudaSoA : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyCudaSoA(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyCudaSoA() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_CUDA_SOA_CUH */
