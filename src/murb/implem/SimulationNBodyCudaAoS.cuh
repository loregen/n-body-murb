#ifndef SIMULATION_N_BODY_CUDA_AOS_CUH
#define SIMULATION_N_BODY_CUDA_AOS_CUH

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyCudaAoS : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

    // vector to store the reduced AoS to be copied to the device
    float *h_AoS_4; 
    // device pointers
    float *d_AoS;  // float4, to be casted
    float *d_acc;  // float4 for alignment, to be casted

    unsigned numBlocks;

  public:
    SimulationNBodyCudaAoS(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyCudaAoS();
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_CUDA_AOS_CUH */
