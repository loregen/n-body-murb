#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyCudaSoA.cuh"

#define MAX_SHARED_PER_BLOCK 48000
#define THREADS_PER_BLK 512

__constant__ float d_G;
__constant__ float d_softSquared;

namespace cuda
{
  __global__ void computeBodiesAccellSoA_k(float *d_data, float3 *d_acc, const unsigned long nBodies, const float softSquared, const float G)
  {

    float *d_x = d_data;
    float *d_y = d_x + nBodies;
    float *d_z = d_y + nBodies;
    float *d_mass = d_z + nBodies;
    __shared__ float4 shared_mem[THREADS_PER_BLK];

    const unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;
    float4 myBody;
    if(iBody < nBodies)
    {
      myBody = make_float4(d_x[iBody], d_y[iBody], d_z[iBody], d_mass[iBody]);
    }
    float3 acc = make_float3(0.f, 0.f, 0.f);

    unsigned tileIdx;
    unsigned tile; 
    for(tile = 0; tile < nBodies / THREADS_PER_BLK; tile++)
    {
      tileIdx = tile * THREADS_PER_BLK + threadIdx.x;

      shared_mem[threadIdx.x] = make_float4(d_x[tileIdx], d_y[tileIdx], d_z[tileIdx], d_mass[tileIdx]);
      __syncthreads();
      //#pragma unroll 8
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

    tileIdx = tile * THREADS_PER_BLK + threadIdx.x;

    //load the last tile
    //shared_mem[threadIdx.x] = (tileIdx < nBodies) ? make_float4(d_x[tileIdx], d_y[tileIdx], d_z[tileIdx], d_mass[tileIdx]) : make_float4(0.f, 0.f, 0.f, 0.f);
    if(tileIdx < nBodies)
    {
      shared_mem[threadIdx.x].x = d_x[tileIdx];
      shared_mem[threadIdx.x].y = d_y[tileIdx];
      shared_mem[threadIdx.x].z = d_z[tileIdx];
      shared_mem[threadIdx.x].w = d_mass[tileIdx];
    }
    else
    {
      shared_mem[threadIdx.x] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();

    for(unsigned jBody = 0; jBody < nBodies % THREADS_PER_BLK; jBody++)
    {
      float4 otherBody = shared_mem[jBody];
      float3 rij = {otherBody.x - myBody.x, otherBody.y - myBody.y, otherBody.z - myBody.z};
      float rijSquared = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z + softSquared;

      float ai = G * otherBody.w / (rijSquared * sqrtf(rijSquared));

      acc.x += ai * rij.x;
      acc.y += ai * rij.y;
      acc.z += ai * rij.z;
    }

    if(iBody < nBodies)
    {
      d_acc[iBody] = acc;
    }

  }

}


SimulationNBodyCudaSoA::SimulationNBodyCudaSoA(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyCudaSoA::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyCudaSoA::computeBodiesAcceleration()
{
    //const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();
    const unsigned long n = this->getBodies().getN();

    //device pointers
    float *d_data;

    // allocate memory on the device
    cudaMalloc(&d_data, 4 * n * sizeof(float) + n * sizeof(float3));
    float3 *d_acc = (float3 *)(d_data + 4 * n);

    //copy body data on device
    float *d_pos = d_data, *d_mass = d_data + 3 * n;
    cudaMemcpy(d_pos, d.qx.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos + n, d.qy.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos + 2 * n, d.qz.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, d.m.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (n + THREADS_PER_BLK - 1) / THREADS_PER_BLK;

    cuda::computeBodiesAccellSoA_k<<<numBlocks, THREADS_PER_BLK>>>(d_data, d_acc, n, this->soft * this->soft, this->G);

    //copy back the result
    cudaMemcpy(this->accelerations.data(), d_acc, n * sizeof(float3), cudaMemcpyDeviceToHost);

    //free memory
    cudaFree(d_data);
}

void SimulationNBodyCudaSoA::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
