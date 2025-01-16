#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyCuda.cuh"

#define MAX_SHARED_PER_BLOCK 48000
#define THREADS_PER_BLK 1024

namespace cuda
{
  void printGPUInfo()
  {
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    printf("Number of devices: %d\n", nDevices);

    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (MHz): %d\n",
               prop.memoryClockRate / 1024);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  Total global memory (Gbytes) %.1f\n", (float)(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
        printf("  Shared memory per block (Kbytes) %.1f\n", (float)(prop.sharedMemPerBlock) / 1024.0);
        printf("  minor-major: %d-%d\n", prop.minor, prop.major);
        printf("  Warp-size: %d\n", prop.warpSize);
        printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("  Concurrent computation/communication: %s\n\n", prop.deviceOverlap ? "yes" : "no");
    }
  }

  __global__ void computeBodiesAccell_k(void *d_AoS, void *d_acc, const unsigned long nBodies, const float softSquared, const float G)
  {
    __shared__ float4 shared_mem[THREADS_PER_BLK];
    float4 *global_AoS = (float4 *)d_AoS;
    float3 *global_acc = (float3 *)d_acc;
    
    const unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;
    
    float4 myBody;
    if(iBody < nBodies)
    {
      myBody = global_AoS[iBody];
    }
    float3 acc = {0.f, 0.f, 0.f};

    unsigned tileIdx;
    for(unsigned tile = 0; tile < nBodies / THREADS_PER_BLK; tile++)
    {
      tileIdx = tile * THREADS_PER_BLK + threadIdx.x;

      shared_mem[threadIdx.x] = global_AoS[tileIdx];
      __syncthreads();
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

    tileIdx += THREADS_PER_BLK;
    // epilogue
    if(tileIdx < nBodies)
    {
      //load the last tile
      shared_mem[threadIdx.x] = global_AoS[tileIdx];
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
    }

    if(iBody < nBodies)
    {
      global_acc[iBody] = acc;
    }

  }

}

SimulationNBodyCuda::SimulationNBodyCuda(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());

    //print CUDA device properties of the current device
    //cuda::printGPUInfo();

}

void SimulationNBodyCuda::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyCuda::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    const unsigned long n = this->getBodies().getN();

    // device pointers
    void *d_AoS;
    void *d_acc;

    // allocate memory on the device
    cudaMalloc(&d_AoS, 4 * n * sizeof(float));
    cudaMalloc(&d_acc, 3 * n * sizeof(float));

    //copy body data on device
    cudaMemcpy(d_AoS, d.data(), 4 * n * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (n + THREADS_PER_BLK - 1) / THREADS_PER_BLK;

    cuda::computeBodiesAccell_k<<<numBlocks, THREADS_PER_BLK>>>(d_AoS, d_acc, n, this->soft * this->soft, this->G);

    //copy back the result
    cudaMemcpy(this->accelerations.data(), d_acc, 3 * n * sizeof(float), cudaMemcpyDeviceToHost);

    //free memory
    cudaFree(d_AoS);
    cudaFree(d_acc);
}

void SimulationNBodyCuda::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
