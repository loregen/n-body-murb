#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyCuda.cuh"

#define MAX_SHARED_PER_BLOCK 48000
#define THREADS_PER_BLK 512

#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t error = call;                                            \
        if (error != cudaSuccess)                                            \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error));                              \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

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
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory (Gbytes): %.1f\n", (float)(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
        printf("  Shared memory per block (Kbytes): %.1f\n", (float)(prop.sharedMemPerBlock) / 1024.0);
        printf("  Total shared memory per multiprocessor (Kbytes): %.1f\n", (float)(prop.sharedMemPerMultiprocessor) / 1024.0);
        printf("  Registers per block: %d\n", prop.regsPerBlock);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max blocks per dimension: %d x %d x %d\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Max threads per dimension: %d x %d x %d\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Number of multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Memory clock rate (MHz): %d\n", prop.memoryClockRate / 1024);
        printf("  Memory bus width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak memory bandwidth (GB/s): %.1f\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("  Concurrent computation/communication: %s\n", prop.deviceOverlap ? "yes" : "no");
        printf("  Clock rate (GHz): %.2f\n", (float)(prop.clockRate) / 1.0e6);
        printf("  L2 cache size (Kbytes): %.1f\n", (float)(prop.l2CacheSize) / 1024.0);
        printf("  Compute mode: %d\n", prop.computeMode);
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
    unsigned tile; 
    for(tile = 0; tile < nBodies / THREADS_PER_BLK; tile++)
    {
      tileIdx = tile * THREADS_PER_BLK + threadIdx.x;

      shared_mem[threadIdx.x] = global_AoS[tileIdx];
      __syncthreads();
      for(unsigned jBody = 0; jBody < THREADS_PER_BLK; jBody++)
      {
        float4 otherBody = shared_mem[jBody];
        float3 rij = {otherBody.x - myBody.x, otherBody.y - myBody.y, otherBody.z - myBody.z};
        float rijSquared = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z + softSquared;

        float ai = G * otherBody.w / (rijSquared * sqrt(rijSquared));

        acc.x += ai * rij.x;
        acc.y += ai * rij.y;
        acc.z += ai * rij.z;
      }
      __syncthreads();
    }

    tileIdx = tile * THREADS_PER_BLK  + threadIdx.x;
    // epilogue
    if(tileIdx < nBodies)
    {
      //load the last tile
      shared_mem[threadIdx.x] = global_AoS[tileIdx];
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
    cuda::printGPUInfo();

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, cuda::computeBodiesAccell_k);
    std::cout << "Shared memory per block required by the kernel: " << attr.sharedSizeBytes << " bytes" << std::endl;

    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cuda::computeBodiesAccell_k, 0, 0);
    std::cout << "Max potential block size: " << blockSize << " threads" << std::endl;

}

void SimulationNBodyCuda::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

typedef struct myAos
{
    float x;
    float y;
    float z;
    float w;
} myAos;

void SimulationNBodyCuda::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    const unsigned long n = this->getBodies().getN();

    std::vector<myAos> d_new(n);
    #pragma omp parallel for 
    for(unsigned long i = 0; i < n; i++)
    {
        d_new[i].x = d[i].qx;
        d_new[i].y = d[i].qy;
        d_new[i].z = d[i].qz;
        d_new[i].w = d[i].m;
    }

    // device pointers
    void *d_AoS;
    void *d_acc;

    // allocate memory on the device
    CUDA_CHECK(cudaMalloc(&d_AoS, 4 * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_acc, 3 * n * sizeof(float)));

    //copy body data on device
    // cudaMemcpy(d_AoS, d.data(), 4 * n * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaMemcpy(d_AoS, d_new.data(), 4 * n * sizeof(float), cudaMemcpyHostToDevice));

    int numBlocks = (n + THREADS_PER_BLK - 1) / THREADS_PER_BLK;

    std::cout << "Launching kernel with " << numBlocks << " blocks and " << THREADS_PER_BLK << " threads per block" << std::endl;
    cuda::computeBodiesAccell_k<<<numBlocks, THREADS_PER_BLK>>>(d_AoS, d_acc, n, this->soft * this->soft, this->G);

    //check for errors
    CUDA_CHECK(cudaPeekAtLastError());

    //copy back the result
    CUDA_CHECK(cudaMemcpy(this->accelerations.data(), d_acc, 3 * n * sizeof(float), cudaMemcpyDeviceToHost));

    //free memory
    CUDA_CHECK(cudaFree(d_AoS));
    CUDA_CHECK(cudaFree(d_acc));
}

void SimulationNBodyCuda::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
