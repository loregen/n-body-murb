#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyOptim.hpp"

SimulationNBodyOptim::SimulationNBodyOptim(const unsigned long nBodies, const std::string &scheme, const float soft,
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

void SimulationNBodyOptim::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyOptim::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

    const unsigned N = this->getBodies().getN();

    // flops = n² * 20
    for (unsigned long iBody = 0; iBody < N; iBody++) {
        // flops = n * 20

        //registers to accumulate acceleration and avoid setting to zero before each iteration
        float acc_x = 0.f;
        float acc_y = 0.f;
        float acc_z = 0.f;

        for (unsigned long jBody = 0; jBody < N; jBody++) {
            const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
            const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
            const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop

            // compute the || rij ||² distance between body i and body j
            const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
            // compute e²
            // const float softSquared = this->soft * this->soft; // 1 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            const float ai = this->mTimesG[jBody] / ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared)); // 5 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij

            acc_x += ai * rijx; // 2 flop
            acc_y += ai * rijy; // 2 flop
            acc_z += ai * rijz; // 2 flop
        }

        this->accelerations[iBody].ax = acc_x;
        this->accelerations[iBody].ay = acc_y;
        this->accelerations[iBody].az = acc_z;

    }
}

void SimulationNBodyOptim::computeOneIteration()
{
    //this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
