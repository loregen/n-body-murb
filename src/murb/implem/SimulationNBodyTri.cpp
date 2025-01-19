#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyTri.hpp"

SimulationNBodyTri::SimulationNBodyTri(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit), softSquared(soft * soft), mTimesG(nBodies)
{
    this->flopsPerIte = 25.0f * nBodies * (nBodies - 1) / 2;
    this->accelerations.resize(this->getBodies().getN());

    const std::vector<float> &masses = this->getBodies().getDataSoA().m;
    for (unsigned iBody = 0; iBody < nBodies; iBody++) {
        this->mTimesG[iBody] = this->G * masses[iBody];
    }
}

void SimulationNBodyTri::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyTri::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        for (unsigned long jBody = iBody + 1; jBody < this->getBodies().getN(); jBody++) {
            const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
            const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
            const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop

            const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops

            const float denominator = (rijSquared + softSquared) * std::sqrt(rijSquared + softSquared); // 3 flops
            const float ai = this->mTimesG[jBody] / denominator; // 1 flops
            const float aj = this->mTimesG[iBody] / denominator; // 1 flops

            this->accelerations[iBody].ax += ai * rijx; // 2 flops
            this->accelerations[iBody].ay += ai * rijy; // 2 flops
            this->accelerations[iBody].az += ai * rijz; // 2 flops

            this->accelerations[jBody].ax -= aj * rijx; // 2 flops
            this->accelerations[jBody].ay -= aj * rijy; // 2 flops
            this->accelerations[jBody].az -= aj * rijz; // 2 flops
        }
    }
}

void SimulationNBodyTri::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
