#ifndef SIMULATION_N_BODY_MIPP_HPP_
#define SIMULATION_N_BODY_MIPP_HPP_

#include <string>
#include "mipp.h"

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyMipp : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

    const mipp::Reg<float> softSquared; /*!< Softening factor squared. */
    std::vector<float> mTimesG; /*!< Array of G * m. */


  public:
    SimulationNBodyMipp(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyMipp() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_MIPP_HPP_ */
