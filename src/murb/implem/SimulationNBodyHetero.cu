#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <chrono>

#include "SimulationNBodyHetero.cuh"

//CUDA parameters, tuned on -n 30000
#define THREADS_PER_BLK 512
#define MAX_SHARED_PER_BLOCK 48000
#define NUM_BLOCKS_CPU 7

namespace cuda
{
  __constant__ float d_G;
  __constant__ float d_softSquared;

  __global__ void computeBodiesAccellHetero_k(float4 *d_AoS, float3 *d_acc, const unsigned nBodies)
  {
    __shared__ float4 shared_mem[THREADS_PER_BLK];
    const unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;

    //read the body data from the global memory
    float4 myBody = d_AoS[iBody];

    //initialize the acceleration
    float3 acc = {0.f, 0.f, 0.f};

    unsigned tileIdx;
    unsigned tile; 
    for(tile = 0; tile < nBodies / THREADS_PER_BLK; tile++)
    {
      tileIdx = tile * THREADS_PER_BLK + threadIdx.x;

      shared_mem[threadIdx.x] = d_AoS[tileIdx];
      __syncthreads();
      #pragma unroll 16
      for(unsigned jBody = 0; jBody < THREADS_PER_BLK; jBody++)
      {
        float4 otherBody = shared_mem[jBody];
        float3 rij = {otherBody.x - myBody.x, otherBody.y - myBody.y, otherBody.z - myBody.z};
        float rijSquared = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z + d_softSquared;

        float ai = d_G * otherBody.w / (rijSquared * sqrtf(rijSquared));
        //molto strano che la prossima riga sia molto più lenta
        //float ai = otherBody.w / (rijSquared * sqrtf(rijSquared));

        acc.x += ai * rij.x;
        acc.y += ai * rij.y;
        acc.z += ai * rij.z;
      }
      __syncthreads();
    }

    tileIdx = tile * THREADS_PER_BLK + threadIdx.x;

    //load the last tile
    shared_mem[threadIdx.x] = (tileIdx < nBodies) ? d_AoS[tileIdx] : make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();

    for(unsigned jBody = 0; jBody < nBodies % THREADS_PER_BLK; jBody++)
    {
      float4 otherBody = shared_mem[jBody];
      float3 rij = {otherBody.x - myBody.x, otherBody.y - myBody.y, otherBody.z - myBody.z};
      float rijSquared = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z + d_softSquared;

      float ai = d_G * otherBody.w / (rijSquared * sqrtf(rijSquared));

      acc.x += ai * rij.x;
      acc.y += ai * rij.y;
      acc.z += ai * rij.z;
    }

    //store the result in the global memory
    d_acc[iBody] = acc;

  }

}

SimulationNBodyHetero::SimulationNBodyHetero(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit), mTimesG(nBodies)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());

    unsigned nFullBlocks = nBodies / THREADS_PER_BLK;
    if(nFullBlocks >= NUM_BLOCKS_CPU)
    {
      numBlocks = nFullBlocks - NUM_BLOCKS_CPU;
    }
    else
    {
      std::cout << "Running only epilogue on CPU" << std::endl;
      numBlocks = nFullBlocks;
    }
    nBodiesGpu = numBlocks * THREADS_PER_BLK;
    if(nBodiesGpu == 0)
    {
      std::cout << "Not enough bodies to run on GPU" << std::endl;
    }

    cudaHostAlloc(&h_AoS_4, nBodies * sizeof(float4), cudaHostAllocDefault);

    cudaMalloc(&d_AoS, nBodies * sizeof(float4));
    cudaMalloc(&d_acc, nBodiesGpu * sizeof(float3));

    //copy the constant memory
    cudaMemcpyToSymbol(cuda::d_G, &G, sizeof(float));
    const float softSquared = soft * soft;
    cudaMemcpyToSymbol(cuda::d_softSquared, &softSquared, sizeof(float));

    //initialize mTimesG auxiliar vector
    const std::vector<float> &masses = this->getBodies().getDataSoA().m;
    for (unsigned iBody = 0; iBody < nBodies; iBody++)
    {
        this->mTimesG[iBody] = masses[iBody] * G;
    }
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
    const std::vector<dataAoS_t<float>> &h_AoS_8 = this->getBodies().getDataAoS();
    const dataSoA_t<float> &h_SoA = this->getBodies().getDataSoA();
    const unsigned nBodies = this->getBodies().getN();

    //auto beforeAoScopy = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(static)
    for(unsigned i = 0; i < nBodies; i++)
    {
      ((float4*)h_AoS_4)[i] = make_float4(h_AoS_8[i].qx, h_AoS_8[i].qy, h_AoS_8[i].qz, h_AoS_8[i].m);
    }
    // auto afterAoScopy = std::chrono::high_resolution_clock::now();
    // std::cout << "AoS copy duration: " << std::chrono::duration<float, std::milli>(afterAoScopy - beforeAoScopy).count() << " ms\n";

    //copy body data on device
    cudaMemcpy(d_AoS, h_AoS_4, nBodies * sizeof(float4), cudaMemcpyHostToDevice);

    // Launch GPU kernel (non-blocking)
    // auto beforeKernel = std::chrono::high_resolution_clock::now();
    if(numBlocks > 0)
    {
      cuda::computeBodiesAccellHetero_k<<<numBlocks, THREADS_PER_BLK>>>((float4*)d_AoS, (float3*)d_acc, nBodies);
    }
    // Do CPU work
    computeEpilogueMipp();

    // auto afterCpuEpilogue = std::chrono::high_resolution_clock::now();

    // cudaError_t hasFinished = cudaStreamQuery(0);
    // if(hasFinished == cudaSuccess)
    // {
    //   std::cout << "GPU finished before CPU" << std::endl;
    // }
    // else
    // { 
    //   std::cout << "CPU is waiting for GPU" << std::endl;
    //   cudaDeviceSynchronize();
    // }
    // auto afterKernel = std::chrono::high_resolution_clock::now();

    // // Calculate durations
    // float kernelDuration = std::chrono::duration<float, std::milli>(afterKernel - beforeKernel).count();
    // float cpuDuration = std::chrono::duration<float, std::milli>(afterCpuEpilogue - beforeKernel).count();
    // float gpuWaitDuration = std::chrono::duration<float, std::milli>(afterKernel - afterCpuEpilogue).count();

    // std::cout << "GPU Kernel duration: " << kernelDuration << " ms\n";
    // std::cout << "CPU Work duration: " << cpuDuration << " ms\n";
    // std::cout << "CPU waited for: " << gpuWaitDuration << " ms\n";

    //copy 
    cudaMemcpy(this->accelerations.data(), d_acc, nBodiesGpu * sizeof(float3), cudaMemcpyDeviceToHost);
}

void SimulationNBodyHetero::computeOneIteration()
{
    //this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

SimulationNBodyHetero::~SimulationNBodyHetero()
{
    //free memory
    cudaFree(d_AoS);
    cudaFree(d_acc);

    cudaFreeHost(h_AoS_4);
}