#include "../inc/parallel_cuda.cuh"
extern "C" {
    #include "../inc/parallel_pthread.h"
}

int main(int argc, char* argv[])
{
    if (argc < 9)
    {
        printf("Wrong argument. Sample correct format: ./hetero_miner maxNumBits maxDifficultyBits maxNumThreads percentageCPU percentageYes grid.x(8192) grid.y(8192) block.x(64) \n");
        return -1;
    }

    const unsigned int maxNumBits = atoi(argv[1]);
    const unsigned int maxDifficultyBits = atoi(argv[2]);
    const unsigned int maxNumThreads = atoi(argv[3]);

    // Percentage on the job distribution
    const unsigned int percentageCPU = atoi(argv[4]);
    const unsigned int percentageYes = atoi(argv[5]);
    
    const unsigned int gridX = atoi(argv[6]);
    const unsigned int gridY = atoi(argv[7]);
    const unsigned int blockX = atoi(argv[8]);

    // Initialize Pthread Implementation
    unsigned int hashes = 0;
    if (percentageYes)
    {
        hashes = blockX * gridY * percentageCPU;
        // printf("hashes1:%ld\n", hashes);
        hashes = hashes * gridX / 100;
        // printf("hashes2:%ld\n", hashes);
    }

    // Start two threads: one for parallel_cuda_impl and the other is for parallel_pthread_impl.
    pthread_t gpuThread;
    pthread_t cpuThread;

    GPUData gpuData;
    gpuData.m_gridDimX = gridX;
    gpuData.m_gridDimY = gridY;
    gpuData.m_blockDimX = blockX;
    gpuData.m_difficulty = maxDifficultyBits;

    CPUData cpuData;
    cpuData.m_maxNumBits = maxNumBits;
    cpuData.m_maxDifficultyBits = maxDifficultyBits;
    cpuData.m_maxNumThreads = maxNumThreads;
    cpuData.m_hashes = hashes;

    struct timespec start;
    struct timespec stop; 
    double time;

    if(clock_gettime(CLOCK_REALTIME, &start) == -1)
    {
        perror("clock gettime");
    }

    int rc = pthread_create(&gpuThread, NULL, parallel_cuda_impl, (void*)&gpuData);
    if (rc)
    {
        printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
    }

    rc = pthread_create(&cpuThread, NULL, parallel_pthread_impl, (void*)&cpuData);
    if (rc)
    {
        printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
    }

    rc = pthread_join(gpuThread, NULL);
    if (rc)
    {
        printf("joining error %d\n", rc);
        exit(-1);
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
