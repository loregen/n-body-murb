#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyLoop.hpp"

constexpr unsigned UNROLL_FACTOR = 16; 

SimulationNBodyLoop::SimulationNBodyLoop(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyLoop::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}


template <unsigned UNROLL_FACTOR, unsigned CURRENT = 0>
void unrollCompute(
    unsigned long iBody, unsigned long jBody,
    const std::vector<dataAoS_t<float>>& d,
    float softSquared, float G,
    std::vector<accAoS_t<float>>& accelerations)
{
    if constexpr (CURRENT < UNROLL_FACTOR) {
        const float rijx = d[jBody + CURRENT].qx - d[iBody].qx;
        const float rijy = d[jBody + CURRENT].qy - d[iBody].qy;
        const float rijz = d[jBody + CURRENT].qz - d[iBody].qz;

        const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
        const float ai = G * d[jBody + CURRENT].m /
                         ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared));

        accelerations[iBody].ax += ai * rijx;
        accelerations[iBody].ay += ai * rijy;
        accelerations[iBody].az += ai * rijz;

        // Recur for the next unrolled iteration
        unrollCompute<UNROLL_FACTOR, CURRENT + 1>(iBody, jBody, d, softSquared, G, accelerations);
    }
}

void SimulationNBodyLoop::computeBodiesAcceleration()
{
    constexpr unsigned UNROLL_FACTOR = 50; // Fixed unroll factor

    const std::vector<dataAoS_t<float>>& d = this->getBodies().getDataAoS();
    const float softSquared = this->soft * this->soft; // 1 flop
    const unsigned N = this->getBodies().getN();

    for (unsigned long iBody = 0; iBody < N; iBody++) {
        unsigned long jBody = 0;

        // Unrolled loop
        for (; jBody + UNROLL_FACTOR - 1 < N; jBody += UNROLL_FACTOR) {
            unrollCompute<UNROLL_FACTOR>(iBody, jBody, d, softSquared, this->G, this->accelerations);
        }

        // Remainder loop
        for (; jBody < N; jBody++) {
            const float rijx = d[jBody].qx - d[iBody].qx;
            const float rijy = d[jBody].qy - d[iBody].qy;
            const float rijz = d[jBody].qz - d[iBody].qz;

            const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
            const float ai = this->G * d[jBody].m /
                             ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared));

            this->accelerations[iBody].ax += ai * rijx;
            this->accelerations[iBody].ay += ai * rijy;
            this->accelerations[iBody].az += ai * rijz;
        }
    }
}


void SimulationNBodyLoop::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
