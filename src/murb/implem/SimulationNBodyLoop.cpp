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

// void SimulationNBodyLoop::computeBodiesAcceleration()
// {
//     // Runtime validation for UNROLL_FACTOR
//     assert(UNROLL_FACTOR <= 4 && "UNROLL_FACTOR must be <= 4");

//     const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
//     const float softSquared = this->soft * this->soft; // 1 flop
//     const unsigned N = this->getBodies().getN();

//     for (unsigned long iBody = 0; iBody < N; iBody++) {
//         unsigned long jBody = 0;

//         // Unrolled loop
//         for (; jBody + UNROLL_FACTOR - 1 < N; jBody += UNROLL_FACTOR) {
//             // Unroll manually based on the defined factor
//                 float rijx = d[jBody].qx - d[iBody].qx;
//                 float rijy = d[jBody].qy - d[iBody].qy;
//                 float rijz = d[jBody].qz - d[iBody].qz;
//                 float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
//                 float ai = this->G * d[jBody].m /
//                                  ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared));
//                 this->accelerations[iBody].ax += ai * rijx;
//                 this->accelerations[iBody].ay += ai * rijy;
//                 this->accelerations[iBody].az += ai * rijz;
            
//                 rijx = d[jBody + 1].qx - d[iBody].qx;
//                 rijy = d[jBody + 1].qy - d[iBody].qy;
//                 rijz = d[jBody + 1].qz - d[iBody].qz;
//                 rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
//                 ai = this->G * d[jBody + 1].m /
//                                  ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared));
//                 this->accelerations[iBody].ax += ai * rijx;
//                 this->accelerations[iBody].ay += ai * rijy;
//                 this->accelerations[iBody].az += ai * rijz;

//                 rijx = d[jBody + 2].qx - d[iBody].qx;
//                 rijy = d[jBody + 2].qy - d[iBody].qy;
//                 rijz = d[jBody + 2].qz - d[iBody].qz;
//                 rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
//                 ai = this->G * d[jBody + 2].m /
//                                  ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared));
//                 this->accelerations[iBody].ax += ai * rijx;
//                 this->accelerations[iBody].ay += ai * rijy;
//                 this->accelerations[iBody].az += ai * rijz;

//                 rijx = d[jBody + 3].qx - d[iBody].qx;
//                 rijy = d[jBody + 3].qy - d[iBody].qy;
//                 rijz = d[jBody + 3].qz - d[iBody].qz;
//                 rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
//                 ai = this->G * d[jBody + 3].m /
//                                  ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared));
//                 this->accelerations[iBody].ax += ai * rijx;
//                 this->accelerations[iBody].ay += ai * rijy;
//                 this->accelerations[iBody].az += ai * rijz;
                
//                 // rijx = d[jBody + 4].qx - d[iBody].qx;
//                 // rijy = d[jBody + 4].qy - d[iBody].qy;
//                 // rijz = d[jBody + 4].qz - d[iBody].qz;
//                 // rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
//                 // ai = this->G * d[jBody + 4].m /
//                 //                  ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared));
//                 // this->accelerations[iBody].ax += ai * rijx;
//                 // this->accelerations[iBody].ay += ai * rijy;
//                 // this->accelerations[iBody].az += ai * rijz;

//                 // rijx = d[jBody + 5].qx - d[iBody].qx;
//                 // rijy = d[jBody + 5].qy - d[iBody].qy;
//                 // rijz = d[jBody + 5].qz - d[iBody].qz;
//                 // rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
//                 // ai = this->G * d[jBody + 5].m /
//                 //                  ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared));
//                 // this->accelerations[iBody].ax += ai * rijx;
//                 // this->accelerations[iBody].ay += ai * rijy;
//                 // this->accelerations[iBody].az += ai * rijz;

//                 // rijx = d[jBody + 6].qx - d[iBody].qx;
//                 // rijy = d[jBody + 6].qy - d[iBody].qy;
//                 // rijz = d[jBody + 6].qz - d[iBody].qz;
//                 // rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
//                 // ai = this->G * d[jBody + 6].m /
//                 //                  ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared));
//                 // this->accelerations[iBody].ax += ai * rijx;
//                 // this->accelerations[iBody].ay += ai * rijy;
//                 // this->accelerations[iBody].az += ai * rijz;

//                 // rijx = d[jBody + 7].qx - d[iBody].qx;
//                 // rijy = d[jBody + 7].qy - d[iBody].qy;
//                 // rijz = d[jBody + 7].qz - d[iBody].qz;
//                 // rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
//                 // ai = this->G * d[jBody + 7].m /
//                 //                  ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared));
//                 // this->accelerations[iBody].ax += ai * rijx;
//                 // this->accelerations[iBody].ay += ai * rijy;
//                 // this->accelerations[iBody].az += ai * rijz;

//                 // rijx = d[jBody + 8].qx - d[iBody].qx;
//                 // rijy = d[jBody + 8].qy - d[iBody].qy;
//                 // rijz = d[jBody + 8].qz - d[iBody].qz;
//                 // rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
//                 // ai = this->G * d[jBody + 8].m /
//                 //                  ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared));
//                 // this->accelerations[iBody].ax += ai * rijx;
//                 // this->accelerations[iBody].ay += ai * rijy;
//                 // this->accelerations[iBody].az += ai * rijz;

//                 // rijx = d[jBody + 9].qx - d[iBody].qx;
//                 // rijy = d[jBody + 9].qy - d[iBody].qy;
//                 // rijz = d[jBody + 9].qz - d[iBody].qz;
//                 // rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
//                 // ai = this->G * d[jBody + 9].m /
//                 //                  ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared));
//                 // this->accelerations[iBody].ax += ai * rijx;
//                 // this->accelerations[iBody].ay += ai * rijy;
//                 // this->accelerations[iBody].az += ai * rijz;
//             }

//         // Remainder loop
//         for (; jBody < N; jBody++) {
//             const float rijx = d[jBody].qx - d[iBody].qx;
//             const float rijy = d[jBody].qy - d[iBody].qy;
//             const float rijz = d[jBody].qz - d[iBody].qz;
//             const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
//             const float ai = this->G * d[jBody].m /
//                              ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared));
//             this->accelerations[iBody].ax += ai * rijx;
//             this->accelerations[iBody].ay += ai * rijy;
//             this->accelerations[iBody].az += ai * rijz;
//         }
//     }
//}

template <unsigned UNROLL_FACTOR, unsigned CURRENT = 0>
void unrollCompute(
    unsigned long iBody, unsigned long jBody,
    const std::vector<dataAoS_t<float>>& d,
    float softSquared, float G,
    //std::vector<acceleration_t<float>>& accelerations)
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
