#include <cstdio>
#include <cstdlib>
#include <stdbool.h>
#include <stdint.h>

#include "../inc/parallel_cuda.cuh"


int main(int argc, char **argv) {
    if (argc < 4)
    {
        printf("Wrong argument. Sample correct format: ./gpu_miner grid.x(8192) grid.y(8192) block.x(64)\n");
        return -1;
    }

    const unsigned int gridX = atoi(argv[1]);
    const unsigned int gridY = atoi(argv[2]);
    const unsigned int blockX = atoi(argv[3]);
    
    GPUData gpuData;
    gpuData.m_gridDimX = gridX;
    gpuData.m_gridDimY = gridY;
    gpuData.m_blockDimX = blockX;
    parallel_cuda_impl((void*)&gpuData);

    return 0;
}
