#include "../inc/profiling.cuh"

double profiling(unsigned int maxNumBits, unsigned int maxDifficultyBits, unsigned int maxNumThreads)
{
    printf("Enter profiling.\n");
    unsigned int hashes = (1 << maxNumBits);
    GPUData gpuData;
    gpuData.m_gridDimX = 2;
    gpuData.m_gridDimY = 2;
    gpuData.m_blockDimX = hashes / gpuData.m_gridDimX / gpuData.m_gridDimY;
    gpuData.m_difficulty = maxDifficultyBits;
    
    CPUData cpuData;
    cpuData.m_maxNumBits = maxNumBits;
    cpuData.m_maxDifficultyBits = maxDifficultyBits;
    cpuData.m_maxNumThreads = maxNumThreads;
    cpuData.m_offset = 0;

    parallel_pthread_impl((void*)&cpuData);
    parallel_cuda_impl((void*)&gpuData);

    printf("Exit profiling.\n");

    return cpuData.m_time / gpuData.m_time;
}
