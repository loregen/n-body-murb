#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <omp.h>

#include "mipp.h"

#include "SimulationNBodyBlocked.hpp"

SimulationNBodyBlocked::SimulationNBodyBlocked(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyBlocked::initIteration()
{
    #pragma omp parallel for
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

#define TILE_SIZE 1000

void SimulationNBodyBlocked::computeBodiesAcceleration()
{
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();

    //const float softSquared = this->soft * this->soft; // 1 flops
    const mipp::Reg<float> softSquared(this->soft * this->soft);
    const mipp::Reg<float> G(this->G);

    const unsigned N = this->getBodies().getN(); 
    #pragma omp parallel
    {
        unsigned nthreads = omp_get_num_threads(); 
        unsigned thread_id = omp_get_thread_num();

        unsigned iStart = (N / nthreads) * thread_id;
        unsigned iEnd = (thread_id == nthreads - 1) ? N : (N / nthreads) * (thread_id + 1);

        unsigned nFullTiles = N / TILE_SIZE;

        for(unsigned tile = 0; tile < nFullTiles; tile++)
        {
            for(unsigned iBody = iStart; iBody < iEnd; iBody++)
            {
                mipp::Reg<float> iqx(d.qx[iBody]);
                mipp::Reg<float> iqy(d.qy[iBody]);
                mipp::Reg<float> iqz(d.qz[iBody]);

                mipp::Reg<float> axi(0.0);
                mipp::Reg<float> ayi(0.0);
                mipp::Reg<float> azi(0.0);

                for(unsigned jBody = tile * TILE_SIZE; jBody < (tile + 1) * TILE_SIZE; jBody += mipp::N<float>())
                {
                    mipp::Reg<float> rijx(mipp::Reg<float>(&d.qx[jBody]) - iqx);
                    mipp::Reg<float> rijy(mipp::Reg<float>(&d.qy[jBody]) - iqy);
                    mipp::Reg<float> rijz(mipp::Reg<float>(&d.qz[jBody]) - iqz);

                    mipp::Reg<float> rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;

                    mipp::Reg<float> m(&d.m[jBody]);
                    rijSquared += softSquared;
                    mipp::Reg<float> ai = G * m / (rijSquared * mipp::sqrt(rijSquared));

                    axi = mipp::fmadd(ai, rijx, axi);
                    ayi = mipp::fmadd(ai, rijy, ayi);
                    azi = mipp::fmadd(ai, rijz, azi);
                }

                //store the acceleration value into the acceleration vector
                this->accelerations[iBody].ax += mipp::sum<float>(axi);
                this->accelerations[iBody].ay += mipp::sum<float>(ayi);
                this->accelerations[iBody].az += mipp::sum<float>(azi);
            }
        }

        for(unsigned iBody = iStart; iBody < iEnd; iBody++)
        {
            mipp::Reg<float> iqx(d.qx[iBody]);
            mipp::Reg<float> iqy(d.qy[iBody]);
            mipp::Reg<float> iqz(d.qz[iBody]);

            mipp::Reg<float> axi(0.0);
            mipp::Reg<float> ayi(0.0);
            mipp::Reg<float> azi(0.0);

            for(unsigned jBody = nFullTiles * TILE_SIZE; jBody < N; jBody += mipp::N<float>())
            {
                mipp::Reg<float> rijx(mipp::Reg<float>(&d.qx[jBody]) - iqx);
                mipp::Reg<float> rijy(mipp::Reg<float>(&d.qy[jBody]) - iqy);
                mipp::Reg<float> rijz(mipp::Reg<float>(&d.qz[jBody]) - iqz);

                mipp::Reg<float> rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;

                mipp::Reg<float> m(&d.m[jBody]);
                rijSquared += softSquared;
                mipp::Reg<float> ai = G * m / (rijSquared * mipp::sqrt(rijSquared));

                axi = mipp::fmadd(ai, rijx, axi);
                ayi = mipp::fmadd(ai, rijy, ayi);
                azi = mipp::fmadd(ai, rijz, azi);
            }

            // store the acceleration value into the acceleration vector
            this->accelerations[iBody].ax += mipp::sum<float>(axi);
            this->accelerations[iBody].ay += mipp::sum<float>(ayi);
            this->accelerations[iBody].az += mipp::sum<float>(azi);
        }
    }
}

void SimulationNBodyBlocked::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
