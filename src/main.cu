#include "../inc/parallel_cuda_impl.cuh"

int main(int argc, char *argv[]) {
    GPUData gpuData;
    gpuData.m_gridDimX = GDIMX;
    gpuData.m_gridDimY = GDIMY;
    gpuData.m_blockDimX = BDIMX;
    parallel_cuda_impl((void*)&gpuData);
    return 0;
}
