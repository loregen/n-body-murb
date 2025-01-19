#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyOmp.hpp"

SimulationNBodyOmp::SimulationNBodyOmp(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit), softSquared(soft * soft), mTimesG(nBodies)
{
    this->flopsPerIte = 18.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());

    const std::vector<float> &masses = this->getBodies().getDataSoA().m;
    for (unsigned iBody = 0; iBody < nBodies; iBody++) {
        this->mTimesG[iBody] = this->G * masses[iBody];
    }
}

void SimulationNBodyOmp::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyOmp::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

    const unsigned N = this->getBodies().getN();

    #pragma omp parallel for
    for (unsigned long iBody = 0; iBody < N; iBody++) {

        float acc_x = 0.0f;
        float acc_y = 0.0f;
        float acc_z = 0.0f;

        #pragma omp simd
        for (unsigned long jBody = 0; jBody < N; jBody++) {
            const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
            const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
            const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop


            float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
            rijSquared += softSquared; // 1 flop
            const float ai = this->mTimesG[jBody] / (rijSquared * std::sqrt(rijSquared)); // 3 flops

            acc_x += ai * rijx; // 2 flops
            acc_y += ai * rijy; // 2 flops
            acc_z += ai * rijz; // 2 flops
        }

        this->accelerations[iBody].ax = acc_x;
        this->accelerations[iBody].ay = acc_y;
        this->accelerations[iBody].az = acc_z;
    }
}

void SimulationNBodyOmp::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);

}
