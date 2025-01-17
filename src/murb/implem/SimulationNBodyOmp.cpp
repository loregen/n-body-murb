#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyOmp.hpp"

SimulationNBodyOmp::SimulationNBodyOmp(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
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

    const float softSquared = this->soft * this->soft; // 1 flops

    const unsigned N = this->getBodies().getN();

    // flops = n² * 20
    #pragma omp parallel for
    for (unsigned long iBody = 0; iBody < N; iBody++) {
        // flops = n * 20

        float acc_x = 0.0f;
        float acc_y = 0.0f;
        float acc_z = 0.0f;

        #pragma omp simd
        for (unsigned long jBody = 0; jBody < N; jBody++) {
            const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
            const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
            const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop

            // compute the || rij ||² distance between body i and body j
            const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
            // compute e²
            // const float softSquared = this->soft * this->soft; // 1 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            const float ai = this->G * d[jBody].m / ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared)); // 5 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
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
