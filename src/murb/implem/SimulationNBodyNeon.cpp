#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <algorithm>

#include "SimulationNBodyNeon.hpp"

#if defined __ARM_NEON || defined _ARM_NEON_
#define NEON_VF_LEN (4)
#include <arm_neon.h>
#define BLOCK_SIZE (512)
SimulationNBodyNeon::SimulationNBodyNeon(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyNeon::initIteration()
{
    #pragma omp parallel for
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyNeon::computeBodiesAcceleration()
{

        // Pre-broadcast constants
    const float soft2 = this->soft * this->soft;  // e²
    const float32x4_t vSoft2 = vdupq_n_f32(soft2);
    const float32x4_t vG     = vdupq_n_f32(this->G);
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();

    unsigned long nBodies = this->getBodies().getN();

        #pragma omp parallel for 
    for (unsigned long iBody = 0; iBody < nBodies; iBody++)
    {
        // Broadcast iBody's position
        const float32x4_t vi_x = vdupq_n_f32(d.qx[iBody]);
        const float32x4_t vi_y = vdupq_n_f32(d.qy[iBody]);
        const float32x4_t vi_z = vdupq_n_f32(d.qz[iBody]);

        // We'll accumulate the NEON sums here:
        float32x4_t partial_sum_x = vdupq_n_f32(0.0f);
        float32x4_t partial_sum_y = vdupq_n_f32(0.0f);
        float32x4_t partial_sum_z = vdupq_n_f32(0.0f);

        unsigned long vecloop_size = ((nBodies)/NEON_VF_LEN)*NEON_VF_LEN;
        // Vectorized loop
        for (unsigned long jBody = 0; jBody  <vecloop_size; jBody += NEON_VF_LEN)
        {
            // Load jBody positions
            float32x4_t rij_x = vsubq_f32(vld1q_f32(&d.qx[jBody]), vi_x);
            float32x4_t rij_y = vsubq_f32(vld1q_f32(&d.qy[jBody]), vi_y);
            float32x4_t rij_z = vsubq_f32(vld1q_f32(&d.qz[jBody]), vi_z);

            // Compute r² = (x² + y² + z²)
            float32x4_t r2_e2 = vmlaq_f32(
                                    vmlaq_f32(
                                        vmulq_f32(rij_x, rij_x), 
                                        rij_y, rij_y
                                        )
                                        , rij_z, rij_z
                                    ); 
            

            // Add softening e²
            r2_e2 = vaddq_f32(r2_e2, vSoft2);


            float32x4_t inv_r3    = vmulq_f32(r2_e2, vsqrtq_f32(r2_e2));      // (r² + e²) * sqrt(r² + e²)
            inv_r3 = vdivq_f32(vmulq_f32(vG, vld1q_f32(&d.m[jBody])), inv_r3);

            // a_i += inv_r3 * r_ij
            partial_sum_x = vmlaq_f32(partial_sum_x, inv_r3, rij_x);
            partial_sum_y = vmlaq_f32(partial_sum_y, inv_r3, rij_y);
            partial_sum_z = vmlaq_f32(partial_sum_z, inv_r3, rij_z);
        }

        float32x2_t tmp = vadd_f32(vget_low_f32(partial_sum_x),
                                    vget_high_f32(partial_sum_x));
        this->accelerations[iBody].ax += vget_lane_f32(tmp, 0)+vget_lane_f32(tmp, 1);
        tmp = vadd_f32(vget_low_f32(partial_sum_y),
                                    vget_high_f32(partial_sum_y));
        this->accelerations[iBody].ay += vget_lane_f32(tmp, 0)+vget_lane_f32(tmp, 1);
        tmp = vadd_f32(vget_low_f32(partial_sum_z),
                                    vget_high_f32(partial_sum_z));
        this->accelerations[iBody].az += vget_lane_f32(tmp, 0)+vget_lane_f32(tmp, 1);


        // Handle any leftover bodies if nBodies is not multiple of NEON_VF_LEN
        for (unsigned int jBody=vecloop_size; jBody < nBodies; jBody++)
        {
            // Scalar fallback for leftover
            float rx = d.qx[jBody] - d.qx[iBody];
            float ry = d.qy[jBody] - d.qy[iBody];
            float rz = d.qz[jBody] - d.qz[iBody];
            float r2 = rx*rx + ry*ry + rz*rz + soft2; // include e²
            float inv_r3 = (this->G * d.m[jBody]) / (r2 * std::sqrt(r2));
            this->accelerations[iBody].ax += inv_r3 * rx;
            this->accelerations[iBody].ay += inv_r3 * ry;
            this->accelerations[iBody].az += inv_r3 * rz;
        }
    }
}

void SimulationNBodyNeon::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

#else
#pragma message("Compiler does not support Neon, neon implementation will be defaulted to Naive")
SimulationNBodyNeon::SimulationNBodyNeon(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyNeon::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyNeon::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

    // flops = n² * 20
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        // flops = n * 20
        for (unsigned long jBody = 0; jBody < this->getBodies().getN(); jBody++) {
            const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
            const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
            const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop

            // compute the || rij ||² distance between body i and body j
            const float rijSquared = std::pow(rijx, 2) + std::pow(rijy, 2) + std::pow(rijz, 2); // 5 flops
            // compute e²
            const float softSquared = std::pow(this->soft, 2); // 1 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            const float ai = this->G * d[jBody].m / std::pow(rijSquared + softSquared, 3.f / 2.f); // 5 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            this->accelerations[iBody].ax += ai * rijx; // 2 flops
            this->accelerations[iBody].ay += ai * rijy; // 2 flops
            this->accelerations[iBody].az += ai * rijz; // 2 flops
        }
    }
}

void SimulationNBodyNeon::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
#endif