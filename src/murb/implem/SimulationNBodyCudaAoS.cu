#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <algorithm>

#include "SimulationNBodyCudaAoS.cuh"

#define MAX_SHARED_PER_BLOCK 48000
#define THREADS_PER_BLK 512

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

  __global__ void computeBodiesAccellAoS_k(float4 *d_AoS, float3 *d_acc, const unsigned long nBodies, const float softSquared, const float G)
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
      #pragma unroll 4
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

    // compute epilogue
    tileIdx = tile * THREADS_PER_BLK + threadIdx.x;
    //load the last tile
    shared_mem[threadIdx.x] = (tileIdx < nBodies) ? d_AoS[tileIdx] : make_float4(0.f, 0.f, 0.f, 0.f);
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

    // store the result in global memory
    if(iBody < nBodies)
    {
      d_acc[iBody] = acc;
    }

  }

}


SimulationNBodyCudaAoS::SimulationNBodyCudaAoS(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : numBlocks((nBodies + THREADS_PER_BLK - 1) / THREADS_PER_BLK),
      SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());

    //print CUDA device properties of the current device
    cuda::printGPUInfo();

    cudaHostAlloc(&h_AoS_4, nBodies * sizeof(float4), cudaHostAllocDefault);

    cudaMalloc(&d_AoS, nBodies * sizeof(float4));
    cudaMalloc(&d_acc, nBodies * sizeof(float3));

}

void SimulationNBodyCudaAoS::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyCudaAoS::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &h_AoS_8 = this->getBodies().getDataAoS();
    const unsigned long n = this->getBodies().getN();

    #pragma omp parallel for 
    for(unsigned long i = 0; i < n; i++)
    {
      ((float4*)h_AoS_4)[i] = make_float4(h_AoS_8[i].qx, h_AoS_8[i].qy, h_AoS_8[i].qz, h_AoS_8[i].m);
    }

    //copy body data on device
    cudaMemcpy(d_AoS, h_AoS_4, n * sizeof(float4), cudaMemcpyHostToDevice);

    cuda::computeBodiesAccellAoS_k<<<numBlocks, THREADS_PER_BLK>>>((float4*)d_AoS, (float3*)d_acc, n, this->soft * this->soft, this->G);

    //copy back the result
    cudaMemcpy(this->accelerations.data(), d_acc, n * sizeof(float3), cudaMemcpyDeviceToHost);

}

void SimulationNBodyCudaAoS::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

SimulationNBodyCudaAoS::~SimulationNBodyCudaAoS()
{
    //free memory
    cudaFree(d_AoS);
    cudaFree(d_acc);

    cudaFreeHost(h_AoS_4);
}
