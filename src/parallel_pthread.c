#include "../inc/parallel_pthread.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        printf("Wrong argument. Sample correct format: ./parallel_pthread maxNumBits maxDifficultyBits maxNumThreads\n");
        return -1;
    }

    const unsigned int maxNumBits = atoi(argv[1]);
    const unsigned int maxDifficultyBits = atoi(argv[2]);
    const unsigned int maxNumThreads = atoi(argv[3]);

    CPUData cpuData;
    cpuData.m_maxNumBits = maxNumBits;
    cpuData.m_maxDifficultyBits = maxDifficultyBits;
    cpuData.m_maxNumThreads = maxNumThreads;
    cpuData.m_offset = 0;

    parallel_pthread_impl((void*)&cpuData);

    return 0;
}
