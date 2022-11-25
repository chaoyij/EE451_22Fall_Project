#include "../inc/parallel_pthread_impl.h"
#include "../inc/sha256_unrolls.h"
#include "../inc/test.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void* KernelSHA256d(void* threadArg)
{
    ThreadData* threadDataPtr = (ThreadData*) threadArg;
    const SHA256_CTX* ctx = threadDataPtr->m_ctx;
    Nonce_result* nr = threadDataPtr->m_nr;
    unsigned int m[64];
    unsigned int hash[8];
    unsigned int arr[8];
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
        for (i = 0; i < 8; i++)
        {
            arr[i] = ctx->state[i];
        }

        //This is a large multiline macro for the SHA256 compression rounds
        // SHA256_COMPRESS_8X
        sha256_compress_8x(arr, cpu_k, m);

        //Prepare input for next SHA-256
        for (i = 0; i < 8; i++)
        {
            m[i] = arr[i] + ctx->state[i];
        }
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
        arr[0] = 0x6a09e667;
        arr[1] = 0xbb67ae85;
        arr[2] = 0x3c6ef372;
        arr[3] = 0xa54ff53a;
        arr[4] = 0x510e527f;
        arr[5] = 0x9b05688c;
        arr[6] = 0x1f83d9ab;
        arr[7] = 0x5be0cd19;

        // SHA256_COMPRESS_1X
        sha256_compress_1x(arr, cpu_k, m);

        hash[0] = ENDIAN_SWAP_32(arr[0] + 0x6a09e667);
        hash[1] = ENDIAN_SWAP_32(arr[1] + 0xbb67ae85);
        hash[2] = ENDIAN_SWAP_32(arr[2] + 0x3c6ef372);
        hash[3] = ENDIAN_SWAP_32(arr[3] + 0xa54ff53a);
        hash[4] = ENDIAN_SWAP_32(arr[4] + 0x510e527f);
        hash[5] = ENDIAN_SWAP_32(arr[5] + 0x9b05688c);
        hash[6] = ENDIAN_SWAP_32(arr[6] + 0x1f83d9ab);
        hash[7] = ENDIAN_SWAP_32(arr[7] + 0x5be0cd19);

        unsigned char* hhh = (unsigned char*) hash;
        i = 0;
        while (hhh[i] == ctx->difficulty[i])
        {
            i++;
        }

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
}

void* parallel_pthread_impl(void* threadArg)
{
    CPUData* cpuDataPtr = (CPUData*)threadArg;
    unsigned int maxNumBits = cpuDataPtr->m_maxNumBits;
    unsigned int maxDifficultyBits = cpuDataPtr->m_maxDifficultyBits;
    unsigned int maxNumThreads = cpuDataPtr->m_maxNumThreads;
    unsigned int offset = cpuDataPtr->m_offset;

    int i, j;
    unsigned char* data = test_block;
    unsigned char hash[32];
    const unsigned int MaxThreads = maxNumThreads;
    const unsigned int hashes = (1 << maxNumBits) - offset;
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

    customize_difficulty(ctx.difficulty, maxDifficultyBits);

    if(clock_gettime(CLOCK_REALTIME, &start) == -1)
    {
        perror("clock gettime");
    }

    for (j = 0; j < threadNum; j++)
    {
        threadDataArray[j].m_nonce = offset + j * elementsPerThread;
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

    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec) / 1e9;
    printf("[PThread] Total number of threads = %ld\n", threadNum);
    printf("[PThread] Total number of testing hashes = %ld\n", hashes);
    printf("[PThread] Execution time = %f seconds\n", time);
    printf("[PThread] Hashrate = %f hashes/second\n", hashes / (time + 0.0));

    if (nr.nonce_found)
    {
        printf("[PThread] Nonce found! %.8x\n", nr.nonce);
    }
    else
    {
        printf("[PThread] Nonce not found :(\n");
    }

    cpuDataPtr->m_time = time;
}

void sha256_compress_1x(unsigned int* arr, const unsigned int* k, unsigned int* m)
{
    unsigned int t1;
    unsigned int t2;
    int i, j;
    for (i = 0; i < 64; i++)
    {
        t1 = arr[7] + EP1(arr[4]) + CH(arr[4], arr[5], arr[6]) + k[i] + m[i];
        t2 = EP0(arr[0]) + MAJ(arr[0], arr[1], arr[2]);
        for (j = 7; j >= 0; j--)
        {
            if (j == 4)
            {
                arr[j] = arr[j - 1] + t1;
            }
            else if (j == 0)
            {
                arr[j] = t1 + t2;
            }
            else
            {
                arr[j] = arr[j - 1];
            }
        }
    }
}

void sha256_compress_8x(unsigned int* arr, const unsigned int* k, unsigned int* m)
{
    unsigned int t1;
    unsigned int t2;
    int i, j;
    for (i = 0; i < 64; i += 8)
    {
        t1 = arr[7] + EP1(arr[4]) + CH(arr[4], arr[5], arr[6]) + k[i] + m[i];
        t2 = EP0(arr[0]) + MAJ(arr[0], arr[1], arr[2]);
        for (j = 7; j >= 0; j--)
        {
            if (j == 4)
            {
                arr[j] = arr[j - 1] + t1;
            }
            else if (j == 0)
            {
                arr[j] = t1 + t2;
            }
            else
            {
                arr[j] = arr[j - 1];
            }
        }

        t1 = arr[7] + EP1(arr[4]) + CH(arr[4], arr[5], arr[6]) + k[i + 1] + m[i + 1];
        t2 = EP0(arr[0]) + MAJ(arr[0], arr[1], arr[2]);
        for (j = 7; j >= 0; j--)
        {
            if (j == 4)
            {
                arr[j] = arr[j - 1] + t1;
            }
            else if (j == 0)
            {
                arr[j] = t1 + t2;
            }
            else
            {
                arr[j] = arr[j - 1];
            }
        }

        t1 = arr[7] + EP1(arr[4]) + CH(arr[4], arr[5], arr[6]) + k[i + 2] + m[i + 2];
        t2 = EP0(arr[0]) + MAJ(arr[0], arr[1], arr[2]);
        for (j = 7; j >= 0; j--)
        {
            if (j == 4)
            {
                arr[j] = arr[j - 1] + t1;
            }
            else if (j == 0)
            {
                arr[j] = t1 + t2;
            }
            else
            {
                arr[j] = arr[j - 1];
            }
        }

        t1 = arr[7] + EP1(arr[4]) + CH(arr[4], arr[5], arr[6]) + k[i + 3] + m[i + 3];
        t2 = EP0(arr[0]) + MAJ(arr[0], arr[1], arr[2]);
        for (j = 7; j >= 0; j--)
        {
            if (j == 4)
            {
                arr[j] = arr[j - 1] + t1;
            }
            else if (j == 0)
            {
                arr[j] = t1 + t2;
            }
            else
            {
                arr[j] = arr[j - 1];
            }
        }

        t1 = arr[7] + EP1(arr[4]) + CH(arr[4], arr[5], arr[6]) + k[i + 4] + m[i + 4];
        t2 = EP0(arr[0]) + MAJ(arr[0], arr[1], arr[2]);
        for (j = 7; j >= 0; j--)
        {
            if (j == 4)
            {
                arr[j] = arr[j - 1] + t1;
            }
            else if (j == 0)
            {
                arr[j] = t1 + t2;
            }
            else
            {
                arr[j] = arr[j - 1];
            }
        }

        t1 = arr[7] + EP1(arr[4]) + CH(arr[4], arr[5], arr[6]) + k[i + 5] + m[i + 5];
        t2 = EP0(arr[0]) + MAJ(arr[0], arr[1], arr[2]);
        for (j = 7; j >= 0; j--)
        {
            if (j == 4)
            {
                arr[j] = arr[j - 1] + t1;
            }
            else if (j == 0)
            {
                arr[j] = t1 + t2;
            }
            else
            {
                arr[j] = arr[j - 1];
            }
        }

        t1 = arr[7] + EP1(arr[4]) + CH(arr[4], arr[5], arr[6]) + k[i + 6] + m[i + 6];
        t2 = EP0(arr[0]) + MAJ(arr[0], arr[1], arr[2]);
        for (j = 7; j >= 0; j--)
        {
            if (j == 4)
            {
                arr[j] = arr[j - 1] + t1;
            }
            else if (j == 0)
            {
                arr[j] = t1 + t2;
            }
            else
            {
                arr[j] = arr[j - 1];
            }
        }

        t1 = arr[7] + EP1(arr[4]) + CH(arr[4], arr[5], arr[6]) + k[i + 7] + m[i + 7];
        t2 = EP0(arr[0]) + MAJ(arr[0], arr[1], arr[2]);
        for (j = 7; j >= 0; j--)
        {
            if (j == 4)
            {
                arr[j] = arr[j - 1] + t1;
            }
            else if (j == 0)
            {
                arr[j] = t1 + t2;
            }
            else
            {
                arr[j] = arr[j - 1];
            }
        }
    }
}
