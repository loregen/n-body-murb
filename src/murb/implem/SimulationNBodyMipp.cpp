#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "mipp.h"

#include "SimulationNBodyMipp.hpp"

SimulationNBodyMipp::SimulationNBodyMipp(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyMipp::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyMipp::computeBodiesAcceleration()
{
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();

    //const float softSquared = this->soft * this->soft; // 1 flops
    const mipp::Reg<float> softSquared = mipp::Reg<float>(this->soft * this->soft);

    const unsigned N = this->getBodies().getN();

    #pragma omp parallel for
    for (unsigned long iBody = 0; iBody < N; iBody++) {

        // mipp::Reg<float> iqx_reg = mipp::Reg<float>(d.qx[iBody]);
        // mipp::Reg<float> iqy_reg = mipp::Reg<float>(d.qy[iBody]);
        // mipp::Reg<float> iqz_reg = mipp::Reg<float>(d.qz[iBody]);

        for (unsigned long jBody = 0; jBody < N; jBody += mipp::N<float>()) {
            // const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
            // const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
            // const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop
            mipp::Reg<float> rijx = mipp::Reg<float>(&d.qx[jBody]) - d.qx[iBody];

            // // compute the || rij ||² distance between body i and body j
            // const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
            // // compute e²
            // // const float softSquared = this->soft * this->soft; // 1 flops
            // // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            // const float ai = this->G * d[jBody].m / ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared)); // 5 flops

            // // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            // this->accelerations[iBody].ax += ai * rijx; // 2 flops
            // this->accelerations[iBody].ay += ai * rijy; // 2 flops
            // this->accelerations[iBody].az += ai * rijz; // 2 flops
        }
    }
}

void SimulationNBodyMipp::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
