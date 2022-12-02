extern "C" {
    #include "../inc/sha256.h"
    #include "../inc/utils.h"
}

#include "../inc/parallel_pthread.h"

#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>

__global__ void kernel_sha256d(SHA256_CTX* ctx, Nonce_result* nr, void* debug, unsigned int* hash_limit);

inline void gpuAssert(cudaError_t code, char* file, int line, bool abort)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            exit(code);
        }
    }
}

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }

int main(int argc, char* argv[])
{
    if (argc < 9)
    {
        printf("Wrong argument. Sample correct format: ./hetero_miner maxNumBits maxDifficulty maxNumThreads percentageCPU percentageYes grid.x(8192) grid.y(8192) block.x(64) \n");
        return -1;
    }

    const unsigned int maxNumBits = atoi(argv[1]);
    const unsigned int maxDifficulty = atoi(argv[2]);
    const unsigned int maxNumThreads = atoi(argv[3]);

    // Percentage on the job distribution
    const unsigned int percentageCPU = atoi(argv[4]);
    const unsigned int percentageYes = atoi(argv[5]);
    
    const unsigned int gridX = atoi(argv[6]);
    const unsigned int gridY = atoi(argv[7]);
    const unsigned int blockX = atoi(argv[8]);

    struct timespec e2eStart;
    struct timespec e2eStop;
    double e2eTime;

    int i, j;
    unsigned char* data = test_block;
    unsigned char hash[32];
    
    // Initialize Pthread Implementation
    const unsigned int MaxThreads = maxNumThreads;

    unsigned int hashes;
    if (!percentageYes)
    {
        hashes = (1 << maxNumBits);
    }
    else
    {
        hashes = blockX * percentageCPU;
        hashes = hashes * gridX * gridY / 100;
    }
    const unsigned int threadNum = hashes <= MaxThreads ? hashes : MaxThreads;  
    const unsigned int elementsPerThread = hashes / threadNum;
    pthread_t threads[threadNum];
    ThreadData threadDataArray[threadNum];
    
    // Initialize CPU CTX
    SHA256_CTX cpu_ctx;
    Nonce_result cpu_nr;
    struct timespec start;
    struct timespec stop; 
    double cpu_time;
    initialize_nonce_result(&cpu_nr);
    sha256_init(&cpu_ctx);
    sha256_update(&cpu_ctx, (unsigned char*) data, 80);    //ctx.state contains a-h
    sha256_pad(&cpu_ctx);
    customize_difficulty(cpu_ctx.difficulty, maxDifficulty);
   
    // Initialize CUDA stuff
    dim3 DimGrid(gridX, gridY);
    dim3 DimBlock(blockX, 1);
    float gpu_time;
    SHA256_CTX gpu_ctx;
    Nonce_result gpu_nr;
    initialize_nonce_result(&gpu_nr);
    sha256_init(&gpu_ctx);
    sha256_update(&gpu_ctx, (unsigned char*) data, 80);
    sha256_pad(&gpu_ctx);
    customize_difficulty(gpu_ctx.difficulty, maxDifficulty);

    // Data buffer for sending debug information to/from the GPU
    unsigned char debug[32];
    unsigned char* d_debug;
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_debug, 32 * sizeof(unsigned char)));
    CUDA_SAFE_CALL(cudaMemcpy(d_debug, (void *) &debug, 32 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    // Allocate Space on Global Memory 
    SHA256_CTX* d_ctx;
    Nonce_result* d_nr;
    unsigned int* d_hashes;
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_ctx, sizeof(SHA256_CTX)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_nr, sizeof(Nonce_result)));
    // Checkpoint
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_hashes, sizeof(unsigned int)));

    //Copy data to device
    CUDA_SAFE_CALL(cudaMemcpy(d_ctx, (void *) &gpu_ctx, sizeof(SHA256_CTX), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_nr, (void *) &gpu_nr, sizeof(Nonce_result), cudaMemcpyHostToDevice));
    // Checkpoint
    CUDA_SAFE_CALL(cudaMemcpy(d_hashes, (void*) &hashes, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Start GPU Timer
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start, 0);

    if(clock_gettime(CLOCK_REALTIME, &e2eStart) == -1)
    {
        perror("clock gettime");
    }

    // Start CPU Timer
    if(clock_gettime(CLOCK_REALTIME, &start) == -1)
    {
        perror("clock gettime");
    }

    // Launch Kernel
    kernel_sha256d<<<DimGrid, DimBlock>>>(d_ctx, d_nr, (void *) d_debug, d_hashes);
    
    // CPU Pthread execution
    for (j = 0; j < threadNum; j++)
    {
        threadDataArray[j].m_nonce = j * elementsPerThread;
        threadDataArray[j].m_ctx = &cpu_ctx;
        threadDataArray[j].m_nr = &cpu_nr;
        threadDataArray[j].m_length = elementsPerThread;
        int rc = pthread_create(&threads[j], NULL, KernelSHA256d, (void*)&threadDataArray[j]);
        if (rc)
        {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }
    
    //Stop GPU timers
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
     
    // GPU  Copy Nonce result back to host
    CUDA_SAFE_CALL(cudaMemcpy((void *) &gpu_nr, d_nr, sizeof(Nonce_result), cudaMemcpyDeviceToHost));
   
    // GPU Syncrhonize
    cudaDeviceSynchronize();

    // Pthread Synchronization
    for (i = 0; i < threadNum; i++)
    {
        int rc = pthread_join(threads[i], NULL);
        if (rc)
        {
            printf("joining error %d\n", rc);
            exit(-1);
        }
    }

    // Stop CPU Pthread Timer
    if(clock_gettime(CLOCK_REALTIME, &stop) == -1)
    {
        perror("clock gettime");
    }

    //Free memory on device
    CUDA_SAFE_CALL(cudaFree(d_ctx));
    CUDA_SAFE_CALL(cudaFree(d_nr));
    CUDA_SAFE_CALL(cudaFree(d_debug));
    CUDA_SAFE_CALL(cudaFree(d_hashes));

    // CPU Results
    cpu_time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;

    if(clock_gettime(CLOCK_REALTIME, &e2eStop) == -1)
    {
        perror("clock gettime");
    }

    e2eTime = (e2eStop.tv_sec - e2eStart.tv_sec)+ (double)(e2eStop.tv_nsec - e2eStart.tv_nsec)/1e9;

    printf("\n[Pthread] Tested %ld hashes\n", hashes);
    printf("[Pthread] Execution time = %f s\n", cpu_time);
    printf("[Pthread] Hashrate = %f H/s\n", hashes / cpu_time);
    
    // GPU Results
    long long int num_hashes = blockX;
    num_hashes *= gridX * gridY;
    printf("\n[CUDA] Tested %ld hashes\n", num_hashes);
    printf("[CUDA] GPU execution time = %f s\n", gpu_time * 1e-3);
    printf("[CUDA] Hashrate = %.2f H/s\n", num_hashes/(gpu_time * 1e-3));
   
    if (cpu_nr.nonce_found)
    {
        printf("\nNonce found in CPU! %.8x\n", cpu_nr.nonce);
    }
    else
    {
        printf("\nNonce not found :(\n");
    }
    
    if (gpu_nr.nonce_found)
    {
        printf("\nNonce found in GPU! %.8x\n", gpu_nr.nonce);
    }
    else
    {
        printf("\nNonce not found :(\n");
    }

    printf("\nE2E total execution time: %lf s\n", e2eTime);

    return 0;
}

//Declare SHA-256 constants
__constant__ uint32_t k_[64] =
{
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};


#define NONCE_VAL (gridDim.x*blockDim.x*blockIdx.y + blockDim.x*blockIdx.x + threadIdx.x)

__global__ void kernel_sha256d(SHA256_CTX* ctx, Nonce_result* nr, void* debug, unsigned int* hash_limit)
{
    unsigned int m[64];
    unsigned int hash[8];
    unsigned int a, b, c, d, e, f, g, h, t1, t2;
    int i, j;
    unsigned int nonce = NONCE_VAL;

    if (nonce < *hash_limit)
    {
        return;
    }

    //Compute SHA-256 Message Schedule
    unsigned int* le_data = (unsigned int*) ctx->data;
    for(i = 0; i < 16; i++)
    {
        m[i] = le_data[i];
    }

    //Replace the nonce
    m[3] = nonce;
    for ( ; i < 64; i++)
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
    SHA256_COMPRESS_8X_GPU

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
    for (i = 16 ; i < 64; i++)
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

    SHA256_COMPRESS_1X_GPU

    hash[0] = ENDIAN_SWAP_32(a + 0x6a09e667);
    hash[1] = ENDIAN_SWAP_32(b + 0xbb67ae85);
    hash[2] = ENDIAN_SWAP_32(c + 0x3c6ef372);
    hash[3] = ENDIAN_SWAP_32(d + 0xa54ff53a);
    hash[4] = ENDIAN_SWAP_32(e + 0x510e527f);
    hash[5] = ENDIAN_SWAP_32(f + 0x9b05688c);
    hash[6] = ENDIAN_SWAP_32(g + 0x1f83d9ab);
    hash[7] = ENDIAN_SWAP_32(h + 0x5be0cd19);

    unsigned char* hhh = (unsigned char *) hash;
    i = 0;
    while(hhh[i] == ctx->difficulty[i])
    {
        i++;
    }

    if(hhh[i] < ctx->difficulty[i])
    {
        //Synchronization Issue
        //Kind of a hack but it really doesn't matter which nonce
        //is written to the output, they're all winners :)
        //Further it's unlikely to even find a nonce let alone 2

        nr->nonce_found = true;
        //The nonce here has the correct endianess,
        //but it must be stored in the block in little endian order
        nr->nonce = nonce;
    }
}
