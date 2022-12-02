#include "../inc/parallel_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <stdbool.h>
#include <stdint.h>

int main(int argc, char* argv[])
{
    int i, j;
    unsigned char* data = test_block;
    
    if (argc < 4)
    {
        printf("Wrong argument. Sample correct format: ./gpu_miner grid.x(8192) grid.y(8192) block.x(64)\n");
        return -1;
    }

    const unsigned int gridX = atoi(argv[1]);
    const unsigned int gridY = atoi(argv[2]);
    const unsigned int blockX = atoi(argv[3]);
    /*
        Host Side Preprocessing
        The goal here is to prepare and compute everything that will be shared by all threads.
    */
    
    //Initialize Cuda stuff
    dim3 DimGrid(gridX, gridY);
    #ifndef ITERATE_BLOCKS
    dim3 DimBlock(blockX, 1);
    #endif

    //Used to store a nonce if a block is mined
    Nonce_result h_nr;
    initialize_nonce_result(&h_nr);

    //Compute the shared portion of the SHA-256d calculation
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, (unsigned char*) data, 80);    //ctx.state contains a-h
    sha256_pad(&ctx);
    //Rearrange endianess of data to optimize device reads
    unsigned int* le_data = (unsigned int*)ctx.data;
    unsigned int le;
    for(i = 0, j = 0; i < 16; i++, j += 4)
    {
        //Get the data out as big endian
        //Store it as little endian via x86
        //On the device side cast the pointer as int* and dereference it correctly
        le = (ctx.data[j] << 24) | (ctx.data[j + 1] << 16) | (ctx.data[j + 2] << 8) | (ctx.data[j + 3]);
        le_data[i] = le;
    }

    //Decodes and stores the difficulty in a 32-byte array for convenience
    customize_difficulty(ctx.difficulty, 2);

    //Data buffer for sending debug information to/from the GPU
    unsigned char debug[32];
    unsigned char* d_debug;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_debug, 32 * sizeof(unsigned char)));
    CUDA_SAFE_CALL(cudaMemcpy(d_debug, (void *) &debug, 32 * sizeof(unsigned char), cudaMemcpyHostToDevice));

    //Allocate space on Global Memory
    SHA256_CTX* d_ctx;
    Nonce_result* d_nr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_ctx, sizeof(SHA256_CTX)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_nr, sizeof(Nonce_result)));

    /*
        Kernel Execution
        Measure and launch the kernel and start mining
    */
    //Copy data to device
    CUDA_SAFE_CALL(cudaMemcpy(d_ctx, (void *) &ctx, sizeof(SHA256_CTX), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_nr, (void *) &h_nr, sizeof(Nonce_result), cudaMemcpyHostToDevice));

    float elapsed_gpu;
    long long int num_hashes;
    //Start timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //Launch Kernel
    kernel_sha256d<<<DimGrid, DimBlock>>>(d_ctx, d_nr, (void *) d_debug);

    //Stop timers
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //Copy nonce result back to host
    CUDA_SAFE_CALL(cudaMemcpy((void *) &h_nr, d_nr, sizeof(Nonce_result), cudaMemcpyDeviceToHost));

    /*    
        Post Processing
        Check the results of mining and print out debug information
    */

    //Cuda Printf output
    cudaDeviceSynchronize();

    //Free memory on device
    CUDA_SAFE_CALL(cudaFree(d_ctx));
    CUDA_SAFE_CALL(cudaFree(d_nr));
    CUDA_SAFE_CALL(cudaFree(d_debug));

    //Output the results
    if(h_nr.nonce_found)
    {
        printf("Nonce found! %.8x\n", h_nr.nonce);
        compute_and_print_hash(data, h_nr.nonce);
    }
    else
    {
        printf("Nonce not found :(\n");
    }
    
    num_hashes = blockX;
    num_hashes *= gridX * gridY;
    printf("Tested %lld hashes\n", num_hashes);
    printf("GPU execution time: %f ms\n", elapsed_gpu);
    printf("Hashrate: %.2f H/s\n", num_hashes/(elapsed_gpu * 1e-3));

    return 0;
}
