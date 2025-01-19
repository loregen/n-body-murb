#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "mipp.h"

#include "SimulationNBodyMippOmpInternal.hpp"

SimulationNBodyMippOmpInternal::SimulationNBodyMippOmpInternal(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyMippOmpInternal::initIteration()
{
    #pragma omp parallel for
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyMippOmpInternal::computeBodiesAcceleration()
{
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();

    //const float softSquared = this->soft * this->soft; // 1 flops
    const mipp::Reg<float> softSquared(this->soft * this->soft);
    const mipp::Reg<float> G(this->G);

    const unsigned N = this->getBodies().getN();

    int stride = mipp::N<float>();

    for (unsigned long iBody = 0; iBody < N; iBody++) {


        #pragma omp parallel 
        {
            mipp::Reg<float> iqx(d.qx[iBody]);
            mipp::Reg<float> iqy(d.qy[iBody]);
            mipp::Reg<float> iqz(d.qz[iBody]);

            mipp::Reg<float> axi(0.0);
            mipp::Reg<float> ayi(0.0);
            mipp::Reg<float> azi(0.0);

            #pragma omp for
            for (unsigned long jBody = 0; jBody < N; jBody += stride) {

                mipp::Reg<float> rijx(mipp::Reg<float>(&d.qx[jBody]) - iqx);
                mipp::Reg<float> rijy(mipp::Reg<float>(&d.qy[jBody]) - iqy);
                mipp::Reg<float> rijz(mipp::Reg<float>(&d.qz[jBody]) - iqz);

                mipp::Reg<float> rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
                mipp::Reg<float> m(&d.m[jBody]);
                rijSquared = rijSquared + softSquared; 
                mipp::Reg<float> ai = G * m / ((rijSquared) * mipp::sqrt(rijSquared));

                axi += ai * rijx;
                ayi += ai * rijy;
                azi += ai * rijz;
            }
            // store the acceleration value into the acceleration vector
            #pragma omp critical
            {
                this->accelerations[iBody].ax += mipp::sum<float>(axi);
                this->accelerations[iBody].ay += mipp::sum<float>(ayi);
                this->accelerations[iBody].az += mipp::sum<float>(azi);
            }
        }
    }
}

void SimulationNBodyMippOmpInternal::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
