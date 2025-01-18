#include "SimulationNBodyHetero.cuh"
#include "mipp.h"

void SimulationNBodyHetero::computeEpilogueMipp()
{
  unsigned iStart = nBodiesGpu;
  unsigned nBodies = this->getBodies().getN();

  const dataSoA_t<float> &h_SoA = this->getBodies().getDataSoA();

  //std::cout << "istart: " << iStart << std::endl;

  // compute epilogue on the CPU
  const mipp::Reg<float> softSquared(this->soft * this->soft);
  const mipp::Reg<float> G(this->G);

  #pragma omp parallel for schedule(static)
  for (unsigned iBody = iStart; iBody < nBodies; iBody++) {

      mipp::Reg<float> iqx(h_SoA.qx[iBody]);
      mipp::Reg<float> iqy(h_SoA.qy[iBody]);
      mipp::Reg<float> iqz(h_SoA.qz[iBody]);

      mipp::Reg<float> axi_reg(0.0);
      mipp::Reg<float> ayi_reg(0.0);
      mipp::Reg<float> azi_reg(0.0);

      for (unsigned jBody = 0; jBody < nBodies; jBody += mipp::N<float>()) {

          mipp::Reg<float> rijx(mipp::Reg<float>(&h_SoA.qx[jBody]) - iqx);
          mipp::Reg<float> rijy(mipp::Reg<float>(&h_SoA.qy[jBody]) - iqy);
          mipp::Reg<float> rijz(mipp::Reg<float>(&h_SoA.qz[jBody]) - iqz);

          mipp::Reg<float> rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
          rijSquared = rijSquared + softSquared;
          // // compute e²
          // mipp::Reg<float> m(&h_SoA.m[jBody]);
          // mipp::Reg<float> ai = m / (rijSquared * mipp::sqrt(rijSquared));
          mipp::Reg<float> mTimesG_reg(&mTimesG[jBody]);
          mipp::Reg<float> ai = mTimesG_reg / (rijSquared * mipp::sqrt(rijSquared));

          // // add the acceleration value into the acceleration vector: ai += || ai ||.rij
          axi_reg = mipp::fmadd(ai, rijx, axi_reg);
          ayi_reg = mipp::fmadd(ai, rijy, ayi_reg);
          azi_reg = mipp::fmadd(ai, rijz, azi_reg);
      }
      // store the acceleration value into the acceleration vector
      accelerations[iBody].ax = mipp::sum<float>(axi_reg);
      accelerations[iBody].ay = mipp::sum<float>(ayi_reg);
      accelerations[iBody].az = mipp::sum<float>(azi_reg);
  }
}