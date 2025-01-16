#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyLoop.hpp"

constexpr unsigned UNROLL_FACTOR = 2; 

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

void SimulationNBodyLoop::computeBodiesAcceleration()
{
    // Runtime validation for UNROLL_FACTOR
    assert(UNROLL_FACTOR <= 2 && "UNROLL_FACTOR must be <= 2");

    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    const float softSquared = this->soft * this->soft; // 1 flop
    const unsigned N = this->getBodies().getN();

    for (unsigned long iBody = 0; iBody < N; iBody++) {
        unsigned long jBody = 0;

        // Unrolled loop
        for (; jBody + UNROLL_FACTOR - 1 < N; jBody += UNROLL_FACTOR) {
            // Unroll manually based on the defined factor
                float rijx = d[jBody].qx - d[iBody].qx;
                float rijy = d[jBody].qy - d[iBody].qy;
                float rijz = d[jBody].qz - d[iBody].qz;
                float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
                float ai = this->G * d[jBody].m /
                                 ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared));
                this->accelerations[iBody].ax += ai * rijx;
                this->accelerations[iBody].ay += ai * rijy;
                this->accelerations[iBody].az += ai * rijz;
            
                rijx = d[jBody + 1].qx - d[iBody].qx;
                rijy = d[jBody + 1].qy - d[iBody].qy;
                rijz = d[jBody + 1].qz - d[iBody].qz;
                rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
                ai = this->G * d[jBody + 1].m /
                                 ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared));
                this->accelerations[iBody].ax += ai * rijx;
                this->accelerations[iBody].ay += ai * rijy;
                this->accelerations[iBody].az += ai * rijz;
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


// void SimulationNBodyLoop::computeBodiesAcceleration()
// {
//     const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

//     const float softSquared = this->soft * this->soft; // 1 flops

//     const unsigned N = this->getBodies().getN();

//     // flops = n² * 20
//     for (unsigned long iBody = 0; iBody < N; iBody++) {
//         // flops = n * 20
//         #pragma GCC unroll 10
//         for (unsigned long jBody = 0; jBody < N; jBody++) {
//             const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
//             const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
//             const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop

//             // compute the || rij ||² distance between body i and body j
//             const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
//             // compute e²
//             // const float softSquared = this->soft * this->soft; // 1 flops
//             // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
//             const float ai = this->G * d[jBody].m / ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared)); // 5 flops

//             // add the acceleration value into the acceleration vector: ai += || ai ||.rij
//             this->accelerations[iBody].ax += ai * rijx; // 2 flops
//             this->accelerations[iBody].ay += ai * rijy; // 2 flops
//             this->accelerations[iBody].az += ai * rijz; // 2 flops
//         }
//     }
// }

void SimulationNBodyLoop::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
