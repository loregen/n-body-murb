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
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit), softSquared(soft * soft), mTimesG(nBodies)
{
    this->flopsPerIte = 18.0f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());

    const std::vector<float> &masses = this->getBodies().getDataSoA().m;
    for (unsigned iBody = 0; iBody < nBodies; iBody++) {
        this->mTimesG[iBody] = this->G * masses[iBody];
    }
}

void SimulationNBodyMipp::initIteration()
{
    #pragma omp parallel for
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyMipp::computeBodiesAcceleration()
{
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();
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
            mipp::Reg<float> rijx(mipp::Reg<float>(&d.qx[jBody]) - iqx); // 1 flop
            mipp::Reg<float> rijy(mipp::Reg<float>(&d.qy[jBody]) - iqy); // 1 flop
            mipp::Reg<float> rijz(mipp::Reg<float>(&d.qz[jBody]) - iqz); // 1 flop

            mipp::Reg<float> rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
            //mipp::Reg<float> rijSquared = mipp::fmadd(rijx, rijx, mipp::fmadd(rijy, rijy, rijz * rijz));

            mipp::Reg<float> mTimesG_reg(&mTimesG[jBody]);
            rijSquared += softSquared; // 1 flop
            mipp::Reg<float> ai = mTimesG_reg / (rijSquared * mipp::sqrt(rijSquared)); // 3 flops

            axi = mipp::fmadd(ai, rijx, axi); // 2 flops
            ayi = mipp::fmadd(ai, rijy, ayi); // 2 flops
            azi = mipp::fmadd(ai, rijz, azi); // 2 flops
            // axi += ai * rijx;
            // ayi += ai * rijy;
            // azi += ai * rijz;
        }
        // store the acceleration value into the acceleration vector
        this->accelerations[iBody].ax = mipp::sum<float>(axi);
        this->accelerations[iBody].ay = mipp::sum<float>(ayi);
        this->accelerations[iBody].az = mipp::sum<float>(azi);
    }
}

void SimulationNBodyMipp::computeOneIteration()
{
    //this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
