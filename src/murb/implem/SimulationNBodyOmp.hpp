#ifndef SIMULATION_N_BODY_OMP_HPP_
#define SIMULATION_N_BODY_OMP_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyOmp : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

    float softSquared; /*!< Softening factor squared. */
    std::vector<float> mTimesG; /*!< Array of G * m. */

  public:
    SimulationNBodyOmp(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyOmp() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_OPTIM_HPP_ */
