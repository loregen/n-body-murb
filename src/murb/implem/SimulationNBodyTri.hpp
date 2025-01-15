#ifndef SIMULATION_N_BODY_TRI_HPP_
#define SIMULATION_N_BODY_TRI_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyTri : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyTri(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyTri() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_TRI_HPP_ */
