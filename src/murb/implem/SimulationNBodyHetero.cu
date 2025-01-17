#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyHetero.cuh"
#include "mippEpilogue.hpp"

#define MAX_SHARED_PER_BLOCK 48000


namespace cuda
{
  __global__ void computeBodiesAccellHetero_k(float4 *d_AoS, float3 *d_acc, const unsigned long nBodies, const float softSquared, const float G)
  {
    __shared__ float4 shared_mem[THREADS_PER_BLK];

    const unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;
    float4 myBody;
    if(iBody < nBodies)
    {
      myBody = d_AoS[iBody];
    }
    float3 acc = {0.f, 0.f, 0.f};

    unsigned tileIdx;
    unsigned tile; 
    for(tile = 0; tile < nBodies / THREADS_PER_BLK; tile++)
    {
      tileIdx = tile * THREADS_PER_BLK + threadIdx.x;

      shared_mem[threadIdx.x] = d_AoS[tileIdx];
      __syncthreads();
      // #pragma unroll 2
      for(unsigned jBody = 0; jBody < THREADS_PER_BLK; jBody++)
      {
        float4 otherBody = shared_mem[jBody];
        float3 rij = {otherBody.x - myBody.x, otherBody.y - myBody.y, otherBody.z - myBody.z};
        float rijSquared = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z + softSquared;

        float ai = G * otherBody.w / (rijSquared * sqrtf(rijSquared));

        acc.x += ai * rij.x;
        acc.y += ai * rij.y;
        acc.z += ai * rij.z;
      }
      __syncthreads();
    }
    // tileIdx = tile * THREADS_PER_BLK + threadIdx.x;


    // //load the last tile
    // shared_mem[threadIdx.x] = (tileIdx < nBodies) ? d_AoS[tileIdx] : make_float4(0.f, 0.f, 0.f, 0.f);
    // __syncthreads();

    // for(unsigned jBody = 0; jBody < nBodies % THREADS_PER_BLK; jBody++)
    // {
    //   float4 otherBody = shared_mem[jBody];
    //   float3 rij = {otherBody.x - myBody.x, otherBody.y - myBody.y, otherBody.z - myBody.z};
    //   float rijSquared = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z + softSquared;

    //   float ai = G * otherBody.w / (rijSquared * sqrtf(rijSquared));

    //   acc.x += ai * rij.x;
    //   acc.y += ai * rij.y;
    //   acc.z += ai * rij.z;
    // }

    if(iBody < nBodies)
    {
      d_acc[iBody] = acc;
    }

  }

  __global__ void sumAccell_k(float3 *d_acc, float3 *d_acc_cpu, const unsigned long nBodies)
  {
    const unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;
    if(iBody < nBodies)
    {
      d_acc_cpu[iBody].x += d_acc[iBody].x;
      d_acc_cpu[iBody].y += d_acc[iBody].y;
      d_acc_cpu[iBody].z += d_acc[iBody].z;
    }
  }

}


SimulationNBodyHetero::SimulationNBodyHetero(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());

}

void SimulationNBodyHetero::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyHetero::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &h_AoS = this->getBodies().getDataAoS();
    const dataSoA_t<float> &h_SoA = this->getBodies().getDataSoA();
    const unsigned long numBodies = this->getBodies().getN();

    const unsigned numFullTiles = numBodies / THREADS_PER_BLK;
    //const unsigned remainder = numBodies % THREADS_PER_BLK;

    std::vector<float4> d_new(numBodies);
    #pragma omp parallel for
    for(unsigned i = 0; i < numBodies; i++)
    {
        d_new[i] = make_float4(h_AoS[i].qx, h_AoS[i].qy, h_AoS[i].qz, h_AoS[i].m);
    }

    // device pointers
    float4 *d_AoS;
    float3 *d_acc;

    // allocate memory on the device
    cudaMalloc(&d_AoS, numBodies * sizeof(float4));
    cudaMalloc(&d_acc, numBodies * sizeof(float3));

    //copy body data on device
    cudaMemcpy(d_AoS, d_new.data(), numBodies * sizeof(float4), cudaMemcpyHostToDevice);

    // launch the kernel (no epilogue)
    int numBlocks = (numBodies + THREADS_PER_BLK - 1) / THREADS_PER_BLK;
    cuda::computeBodiesAccellHetero_k<<<numBlocks, THREADS_PER_BLK>>>(d_AoS, d_acc, numBodies, this->soft * this->soft, this->G);

    compute_epilogue_mipp(numBodies, h_SoA, this->accelerations, this->G, this->soft * this->soft, numFullTiles * THREADS_PER_BLK);
    //not needed ?
    //cudaDeviceSynchronize();

    // copy the cpu acc to the gpu
    float3 *d_acc_cpu = (float3*)d_AoS;
    cudaMemcpy(d_acc_cpu, this->accelerations.data(), numBodies * sizeof(float3), cudaMemcpyHostToDevice);

    // sum the cpu and gpu acc
    cuda::sumAccell_k<<<numBlocks, THREADS_PER_BLK>>>(d_acc, d_acc_cpu, numBodies);

    // copy back the result
    cudaMemcpy(this->accelerations.data(), d_acc_cpu, numBodies * sizeof(float3), cudaMemcpyDeviceToHost);

    //free memory
    cudaFree(d_AoS);
    cudaFree(d_acc);
}

void SimulationNBodyHetero::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
