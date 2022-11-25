#pragma once

extern "C" {
    #include "parallel_pthread_impl.h"
}
#include "parallel_cuda_impl.cuh"

double profiling(unsigned int maxNumBits, unsigned int maxDifficultyBits, unsigned int maxNumThreads);
