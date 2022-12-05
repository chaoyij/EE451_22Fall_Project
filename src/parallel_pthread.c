#include "../inc/parallel_pthread.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        printf("Wrong argument. Sample correct format: ./parallel_pthread maxNumBits maxDifficulty maxNumThreads\n");
        return -1;
    }

    const unsigned int maxNumBits = atoi(argv[1]);
    const unsigned int maxDifficulty = atoi(argv[2]);
    const unsigned int maxNumThreads = atoi(argv[3]);

    int i, j;
    unsigned char* data = test_block;
    unsigned char hash[32];
    const unsigned int MaxThreads = maxNumThreads;
    const unsigned int hashes = (1 << maxNumBits);
    const unsigned int threadNum = hashes <= MaxThreads ? hashes : MaxThreads;
    const unsigned int elementsPerThread = hashes / threadNum;
    pthread_t threads[threadNum];
    ThreadData threadDataArray[threadNum];
    SHA256_CTX ctx;
    Nonce_result nr;
    struct timespec start;
    struct timespec stop; 
    double time;

    initialize_nonce_result(&nr);

    sha256_init(&ctx);
    sha256_update(&ctx, (unsigned char*) data, 80);    //ctx.state contains a-h
    sha256_pad(&ctx);

    customize_difficulty(ctx.difficulty, maxDifficulty);

    if(clock_gettime(CLOCK_REALTIME, &start) == -1)
    {
        perror("clock gettime");
    }

    for (j = 0; j < threadNum; j++)
    {
        threadDataArray[j].m_nonce = j * elementsPerThread;
        threadDataArray[j].m_ctx = &ctx;
        threadDataArray[j].m_nr = &nr;
        threadDataArray[j].m_length = elementsPerThread;
        int rc = pthread_create(&threads[j], NULL, KernelSHA256d, (void*)&threadDataArray[j]);
        if (rc)
        {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for (i = 0; i < threadNum; i++)
    {
        int rc = pthread_join(threads[i], NULL);
        if (rc)
        {
            printf("joining error %d\n", rc);
            exit(-1);
        }
    }

    if(clock_gettime(CLOCK_REALTIME, &stop) == -1)
    {
        perror("clock gettime");
    }

    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    printf("Total number of threads = %ld\n", threadNum);
    printf("Total number of testing hashes = %ld\n", hashes);
    printf("Execution time = %f s\n", time);
    printf("Hashrate = %f H/s\n", hashes / (time + 0.0));
    if (nr.nonce_found)
    {
        printf("Nonce found! %.8x\n", nr.nonce);
    }
    else
    {
        printf("Nonce not found :(\n");
    }
}
