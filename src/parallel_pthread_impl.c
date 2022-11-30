#include "../inc/parallel_pthread_impl.h"

void* KernelSHA256d(void* threadArg)
{
    ThreadData* threadDataPtr = (ThreadData*) threadArg;
    const SHA256_CTX* ctx = threadDataPtr->m_ctx;
    Nonce_result* nr = threadDataPtr->m_nr;
    unsigned int m[64];
    unsigned int hash[8];
    unsigned int a,b,c,d,e,f,g,h,t1,t2;
    int i, j, index;
    unsigned int nonce = threadDataPtr->m_nonce;
    unsigned int length = threadDataPtr->m_length;

    for (index = 0; index < length; index++)
    {
        if (nr->nonce_found)
        {
            break;
        }
        //Compute SHA-256 Message Schedule
        unsigned int* le_data = (unsigned int *) ctx->data;
        for(i = 0; i < 16; i++)
        {
            m[i] = le_data[i];
        }
        //Replace the nonce
        m[3] = nonce + index;
        for ( ; i < 64; ++i)
        {
            m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
        }

        //Copy Initial Values into registers
        a = ctx->state[0];
        b = ctx->state[1];
        c = ctx->state[2];
        d = ctx->state[3];
        e = ctx->state[4];
        f = ctx->state[5];
        g = ctx->state[6];
        h = ctx->state[7];

        //This is a large multiline macro for the SHA256 compression rounds
        SHA256_COMPRESS_8X

        //Prepare input for next SHA-256
        m[0] = a + ctx->state[0];
        m[1] = b + ctx->state[1];
        m[2] = c + ctx->state[2];
        m[3] = d + ctx->state[3];
        m[4] = e + ctx->state[4];
        m[5] = f + ctx->state[5];
        m[6] = g + ctx->state[6];
        m[7] = h + ctx->state[7];
        //Pad the input
        m[8] = 0x80000000;
        for(i = 9; i < 15; i++)
        {
            m[i] = 0x00;
        }
        m[15] = 0x00000100;    //Write out l=256
        for (i = 16 ; i < 64; ++i)
        {
            m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
        }

        //Initialize the SHA-256 registers
        a = 0x6a09e667;
        b = 0xbb67ae85;
        c = 0x3c6ef372;
        d = 0xa54ff53a;
        e = 0x510e527f;
        f = 0x9b05688c;
        g = 0x1f83d9ab;
        h = 0x5be0cd19;

        SHA256_COMPRESS_1X

        hash[0] = ENDIAN_SWAP_32(a + 0x6a09e667);
        hash[1] = ENDIAN_SWAP_32(b + 0xbb67ae85);
        hash[2] = ENDIAN_SWAP_32(c + 0x3c6ef372);
        hash[3] = ENDIAN_SWAP_32(d + 0xa54ff53a);
        hash[4] = ENDIAN_SWAP_32(e + 0x510e527f);
        hash[5] = ENDIAN_SWAP_32(f + 0x9b05688c);
        hash[6] = ENDIAN_SWAP_32(g + 0x1f83d9ab);
        hash[7] = ENDIAN_SWAP_32(h + 0x5be0cd19);

        unsigned char* hhh = (unsigned char*) hash;
        i = 0;
        while (hhh[i] == ctx->difficulty[i])
        {
            i++;
        }
        // printf("hhh: %d, difficulty: %d\n",hhh[i], ctx->difficulty[i]);
        if (hhh[i] < ctx->difficulty[i] && !nr->nonce_found)
        {
            // Acquire lock
            pthread_mutex_lock(&mutex);

            printf("Nonce found! %.8x\n", nonce);

            nr->nonce_found = true;
            //The nonce here has the correct endianess,
            //but it must be stored in the block in little endian order
            nr->nonce = nonce;

            printf("hash:");
            for (j = 0; j < 32; j++)
            {
                printf("%02x", hhh[j]);
            }
            printf("\n");

            // Release lock
            pthread_mutex_unlock(&mutex);
        }
    }
    return 0;
}

void* parallel_pthread_impl(void* threadArg)
{
    CPUData* cpuDataPtr = (CPUData*)threadArg;
    unsigned int maxNumBits = cpuDataPtr->m_maxNumBits;
    unsigned int maxDifficultyBits = cpuDataPtr->m_maxDifficultyBits;
    unsigned int maxNumThreads = cpuDataPtr->m_maxNumThreads;

    int i, j;
    unsigned char* data = test_block;
    unsigned char hash[32];
    const unsigned int MaxThreads = maxNumThreads;
    const unsigned int hashes = cpuDataPtr->m_hashes == 0 ? (1 << maxNumBits) : cpuDataPtr->m_hashes;
    const unsigned int threadNum = hashes <= MaxThreads ? hashes : MaxThreads;
    const unsigned int elementsPerThread = hashes / threadNum;

    printf("maxNumBits:%ld, maxDifficultyBits:%ld, maxNumThreads:%ld, hashes:%ld\n", maxNumBits, maxDifficultyBits, maxNumThreads, hashes);

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

    customize_difficulty(ctx.difficulty, maxDifficultyBits);

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
    printf("[Pthread] Total number of threads = %ld\n", threadNum);
    printf("[Pthread] Total number of testing hashes = %ld\n", hashes);
    printf("[Pthread] Execution time = %f seconds\n", time);
    printf("[Pthread] Hashrate = %f hashes/second\n", hashes / (time + 0.0));
    if (nr.nonce_found)
    {
        printf("[Pthread] Nonce found! %.8x\n", nr.nonce);
    }
    else
    {
        printf("[Pthread] Nonce not found :(\n");
    }

    cpuDataPtr->m_time = time;

    return 0;
}