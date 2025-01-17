#include "mippEpilogue.hpp"
#include "mipp.h"

void compute_epilogue_mipp(const unsigned numBodies, const dataSoA_t<float> &h_SoA, std::vector<accAoS_t<float>> &accelerations, const float G_param, const float softSquared_param, unsigned jStart)
{
    const unsigned remainder = (numBodies - jStart) % mipp::N<float>();
    const unsigned jEnd = numBodies - remainder;

    // compute epilogue on the CPU
    const mipp::Reg<float> softSquared(softSquared_param);
    const mipp::Reg<float> G(G_param);

    //#pragma omp parallel for
    for (unsigned long iBody = 0; iBody < numBodies; iBody++) {

        mipp::Reg<float> iqx(h_SoA.qx[iBody]);
        mipp::Reg<float> iqy(h_SoA.qy[iBody]);
        mipp::Reg<float> iqz(h_SoA.qz[iBody]);


        mipp::Reg<float> axi_reg(0.0);
        mipp::Reg<float> ayi_reg(0.0);
        mipp::Reg<float> azi_reg(0.0);

        for (unsigned long jBody = jStart; jBody < jEnd; jBody += mipp::N<float>()) {

            mipp::Reg<float> rijx(mipp::Reg<float>(&h_SoA.qx[jBody]) - iqx);
            mipp::Reg<float> rijy(mipp::Reg<float>(&h_SoA.qy[jBody]) - iqy);
            mipp::Reg<float> rijz(mipp::Reg<float>(&h_SoA.qz[jBody]) - iqz);

            mipp::Reg<float> rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;

            // // compute e²
            mipp::Reg<float> m(&h_SoA.m[jBody]);
            mipp::Reg<float> ai = G * m / ((rijSquared + softSquared) * mipp::sqrt(rijSquared + softSquared));

            // // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            axi_reg += ai * rijx;
            ayi_reg += ai * rijy;
            azi_reg += ai * rijz;
        }
        // store the acceleration value into the acceleration vector
        accelerations[iBody].ax = mipp::sum<float>(axi_reg);
        accelerations[iBody].ay = mipp::sum<float>(ayi_reg);
        accelerations[iBody].az = mipp::sum<float>(azi_reg);

        // Final epilogue 
        float axi = 0.0;
        float ayi = 0.0;
        float azi = 0.0;

        for (unsigned long jBody = jEnd; jBody < numBodies; jBody++) {

            auto rijx = h_SoA.qx[jBody] - h_SoA.qx[iBody];
            auto rijy = h_SoA.qy[jBody] - h_SoA.qy[iBody];
            auto rijz = h_SoA.qz[jBody] - h_SoA.qz[iBody];

            auto rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;

            // // compute e²
            auto m = h_SoA.m[jBody];
            auto ai = G_param * m / ((rijSquared + softSquared_param) * sqrt(rijSquared + softSquared_param));


            // // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            axi += ai * rijx;
            ayi += ai * rijy;
            azi += ai * rijz;
        }
        // store the acceleration value into the acceleration vector
        accelerations[iBody].ax += axi;
        accelerations[iBody].ay += ayi;
        accelerations[iBody].az += azi;
    }
}

// void compute_epilogue_mipp(const unsigned numBodies, const dataSoA_t<float> &h_SoA, std::vector<accAoS_t<float>> &accelerations, const float G_param, const float softSquared_param, unsigned jStart)
// {

//     // compute epilogue on the CPU

//     #pragma omp parallel for
//     for (unsigned long iBody = 0; iBody < numBodies; iBody++) {

//         // mipp::Reg<float> iqx(h_SoA.qx[iBody]);
//         // mipp::Reg<float> iqy(h_SoA.qy[iBody]);
//         // mipp::Reg<float> iqz(h_SoA.qz[iBody]);
//         auto &iqx = h_SoA.qx[iBody];
//         auto &iqy = h_SoA.qy[iBody];
//         auto &iqz = h_SoA.qz[iBody];


//         // mipp::Reg<float> axi(0.0);
//         // mipp::Reg<float> ayi(0.0);
//         // mipp::Reg<float> azi(0.0);
//         float axi = 0.0;
//         float ayi = 0.0;
//         float azi = 0.0;

//         // for (unsigned long jBody = jStart; jBody < numBodies; jBody += mipp::N<float>()) {
//         for (unsigned long jBody = jStart; jBody < numBodies; jBody++) {

//             // mipp::Reg<float> rijx(mipp::Reg<float>(&h_SoA.qx[jBody]) - iqx);
//             // mipp::Reg<float> rijy(mipp::Reg<float>(&h_SoA.qy[jBody]) - iqy);
//             // mipp::Reg<float> rijz(mipp::Reg<float>(&h_SoA.qz[jBody]) - iqz);
//             auto rijx = h_SoA.qx[jBody] - iqx;
//             auto rijy = h_SoA.qy[jBody] - iqy;
//             auto rijz = h_SoA.qz[jBody] - iqz;

//             // mipp::Reg<float> rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
//             auto rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;

//             // // compute e²
//             // mipp::Reg<float> m(&h_SoA.m[jBody]);
//             // mipp::Reg<float> ai = G * m / ((rijSquared + softSquared) * mipp::sqrt(rijSquared + softSquared));
//             auto m = h_SoA.m[jBody];
//             auto ai = G_param * m / ((rijSquared + softSquared_param) * sqrt(rijSquared + softSquared_param));


//             // // add the acceleration value into the acceleration vector: ai += || ai ||.rij
//             axi += ai * rijx;
//             ayi += ai * rijy;
//             azi += ai * rijz;
//         }
//         // store the acceleration value into the acceleration vector
//         // accelerations[iBody].ax = mipp::sum<float>(axi);
//         // accelerations[iBody].ay = mipp::sum<float>(ayi);
//         // accelerations[iBody].az = mipp::sum<float>(azi);
//         accelerations[iBody].ax = axi;
//         accelerations[iBody].ay = ayi;
//         accelerations[iBody].az = azi;
//     }
// }