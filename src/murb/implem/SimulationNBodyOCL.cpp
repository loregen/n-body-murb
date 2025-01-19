#include "SimulationNBodyOCL.hpp"
#include <CL/cl.hpp>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

SimulationNBodyOCL::SimulationNBodyOCL(const unsigned long nBodies, const std::string &scheme, const float soft,
                                       const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
    /* OCL Device test */
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found!" << std::endl;
    }

    // Iterate through platforms and display information
    for (size_t i = 0; i < platforms.size(); ++i) {

        std::cout << "Platform " << i + 1 << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
        // Get devices for this platform
        std::vector<cl::Device> devices;
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        if (devices.empty()) {
            std::cout << "  No devices found for this platform." << std::endl;
            continue;
        }

        // Display device information
        for (size_t j = 0; j < devices.size(); ++j) {
            std::cout << "  Device " << j + 1 << ": " << devices[j].getInfo<CL_DEVICE_NAME>() << std::endl;
            std::cout << "    Vendor: " << devices[j].getInfo<CL_DEVICE_VENDOR>() << std::endl;
            std::cout << "    Version: " << devices[j].getInfo<CL_DEVICE_VERSION>() << std::endl;
            std::cout << "    Max Compute Units: " << devices[j].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            std::cout << "    Global Memory Size: " << devices[j].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024)
                      << " MB" << std::endl;
        }
    }
}

void SimulationNBodyOCL::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyOCL::computeBodiesAcceleration()
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

void SimulationNBodyOCL::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
