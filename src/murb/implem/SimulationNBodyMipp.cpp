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
    const mipp::Reg<float> softSquared(this->soft * this->soft);
    const mipp::Reg<float> G(this->G);

    const unsigned N = this->getBodies().getN();

    #pragma omp parallel for
    for (unsigned long iBody = 0; iBody < N; iBody++) {

        mipp::Reg<float> iqx(d.qx[iBody]);
        mipp::Reg<float> iqy(d.qy[iBody]);
        mipp::Reg<float> iqz(d.qz[iBody]);

        mipp::Reg<float> axi(0.0);
        mipp::Reg<float> ayi(0.0);
        mipp::Reg<float> azi(0.0);

        for (unsigned long jBody = 0; jBody < N; jBody += mipp::N<float>()) {
            // const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
            // const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
            // const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop
            mipp::Reg<float> rijx = mipp::Reg<float>(&d.qx[jBody]) - iqx;
            mipp::Reg<float> rijy = mipp::Reg<float>(&d.qy[jBody]) - iqy;
            mipp::Reg<float> rijz = mipp::Reg<float>(&d.qz[jBody]) - iqz;

            // // compute the || rij ||² distance between body i and body j
            // const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops

            mipp::Reg<float> rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;

            // // compute e²
            // // const float softSquared = this->soft * this->soft; // 1 flops
            // // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            // const float ai = this->G * d[jBody].m / ((rijSquared + softSquared) * std::sqrt(rijSquared + softSquared)); // 5 flops

            mipp::Reg<float> m(&d.m[jBody]);
            mipp::Reg<float> ai = G * m / ((rijSquared + softSquared) * mipp::sqrt(rijSquared + softSquared));

            // // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            // this->accelerations[iBody].ax += ai * rijx; // 2 flops
            // this->accelerations[iBody].ay += ai * rijy; // 2 flops
            // this->accelerations[iBody].az += ai * rijz; // 2 flops

            axi += ai * rijx;
            ayi += ai * rijy;
            azi += ai * rijz;
        }
        // store the acceleration value into the acceleration vector
        this->accelerations[iBody].ax = mipp::sum<float>(axi);
        this->accelerations[iBody].ay = mipp::sum<float>(ayi);
        this->accelerations[iBody].az = mipp::sum<float>(azi);
    }
}

void SimulationNBodyMipp::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
