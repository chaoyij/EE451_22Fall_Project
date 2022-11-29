#include "sha256.h"
#include "sha256_unrolls.h"
#include "test.h"
#include "utils.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct 
{
    SHA256_CTX* m_ctx;
    Nonce_result* m_nr;
    unsigned int m_nonce;
    unsigned int m_length;
} ThreadData;

pthread_mutex_t mutex;

//Declare SHA-256 constants
const uint32_t k[64] = {                                              
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};


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

