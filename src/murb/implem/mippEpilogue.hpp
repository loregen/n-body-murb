#include "core/Bodies.hpp"

//CUDA parameters
#define THREADS_PER_BLK 512

void compute_epilogue_mipp(const unsigned numBodies, const dataSoA_t<float> &h_SoA, std::vector<accAoS_t<float>> &accelerations, const float G_param, const float softSquared, unsigned jStart);