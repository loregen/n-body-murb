#ifndef SIMULATION_N_BODY_HETERO_CUH
#define SIMULATION_N_BODY_HETERO_CUH

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyHetero : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

    std::vector<float> mTimesG;
    // vector to store the reduced AoS to be copied to the device
    float *h_AoS_4; 
    // device pointers
    float *d_AoS;  // float4, to be casted
    float *d_acc;  // float3, to be casted

    unsigned numBlocks;
    unsigned nBodiesGpu;

  public:
    SimulationNBodyHetero(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyHetero();
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
    void computeCpuBlock();
};

#endif /* SIMULATION_N_BODY_HETERO_CUH */
