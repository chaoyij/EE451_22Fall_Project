#include "../inc/profiling.cuh"
#include "cuPrintf.cu"

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        printf("Wrong argument. Sample correct format: ./parallel_pthread maxNumBits maxDifficultyBits maxNumThreads\n");
        return -1;
    }

    // Profiling PThread and CUDA
    const unsigned int maxNumBits = atoi(argv[1]);
    const unsigned int maxDifficultyBits = atoi(argv[2]);
    const unsigned int maxNumThreads = atoi(argv[3]);

    double ratio = profiling(maxNumBits, maxDifficultyBits, maxNumThreads);

    printf("ratio is %lf\n", ratio);

    // Allocate workload based on profiling
    // Based on ratio (r1:r2), if we search in the range from 0 to 2^k - 1,
    // the workload of GPU is rangeGPU = 2^k * r1 / (r1 + r2), and
    // the workload of CPU is rangeCPU = 2^k * r2 / (r1 + r2). Thus,
    // GPU processes the range from 0 to rangeGPU - 1, and
    // CPU processes the range from rangeGPU to 2^k - 1.
    const unsigned int hashes = (1 << maxNumBits);
    const unsigned int rangeGPU = hashes * ratio / (1 + ratio);
    printf("rangeGPU:%ld\n", rangeGPU);

    // Start two threads: one for parallel_cuda_impl and the other is for parallel_pthread_impl.
    pthread_t gpuThread;
    pthread_t cpuThread;

    const int DIM = 2;
    GPUData gpuData;
    gpuData.m_gridDimX = DIM;
    gpuData.m_gridDimY = DIM;
    int blockDim = rangeGPU / DIM / DIM;
    gpuData.m_blockDimX = (blockDim == 0 ? 1 : blockDim);
    gpuData.m_difficulty = maxDifficultyBits;
    printf("gridDimX:%ld, gridDimY:%ld, blockDim:%ld\n", gpuData.m_gridDimX, gpuData.m_gridDimY, gpuData.m_blockDimX);

    CPUData cpuData;
    cpuData.m_maxNumBits = maxNumBits;
    cpuData.m_maxDifficultyBits = maxDifficultyBits;
    cpuData.m_maxNumThreads = maxNumThreads;
    cpuData.m_offset = rangeGPU;

    int rc;

    struct timespec start;
    struct timespec stop; 
    double time;

    if(clock_gettime(CLOCK_REALTIME, &start) == -1)
    {
        perror("clock gettime");
    }

    if (rangeGPU > 0)
    {
        printf("Create gpuThread\n");
        int rc = pthread_create(&gpuThread, NULL, parallel_cuda_impl, (void*)&gpuData);
        if (rc)
        {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }
    printf("Create cpuThread\n");
    rc = pthread_create(&cpuThread, NULL, parallel_pthread_impl, (void*)&cpuData);
    if (rc)
    {
        printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
    }

    // Join GPU and CPU threads
    if (rangeGPU > 0)
    {
        rc = pthread_join(gpuThread, NULL);
        if (rc)
        {
            printf("joining error %d\n", rc);
            exit(-1);
        }
    }

    rc = pthread_join(cpuThread, NULL);
    if (rc)
    {
        printf("joining error %d\n", rc);
        exit(-1);
    }
    printf("Finish\n");

    if(clock_gettime(CLOCK_REALTIME, &stop) == -1)
    {
        perror("clock gettime");
    }

    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec) / 1e9;
    printf("[Heterogeneous] Execution time = %f seconds\n", time);

    return 0;
}
