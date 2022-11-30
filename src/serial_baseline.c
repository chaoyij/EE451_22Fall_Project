#include <stdio.h>
#include <stdlib.h>

#include "../inc/sha256.h"
#include "../inc/utils.h"
#include "../inc/test.h"


int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Wrong argument. Sample correct format: ./parallel_pthread maxNumBits maxDifficultyBits\n");
        return -1;
    }

    const unsigned int maxNumBits = atoi(argv[1]);
    const unsigned int maxDifficultyBits = atoi(argv[2]);

    int i, j, k;
    unsigned char *data = test_block;
    unsigned char hash[32], difficulty[32];
    SHA256_CTX ctx;
    Nonce_result nr;

    initialize_nonce_result(&nr);

    set_difficulty(difficulty, maxDifficultyBits);

    int hashes = (1 << maxNumBits);
    tick();
    for(j = 0; j < hashes; j++)
    {
        //Hash the block header
        sha256_init(&ctx);
        sha256_update(&ctx, data, 80);
        sha256_final(&ctx, hash);
        //Hash
        sha256_init(&ctx);
        sha256_update(&ctx, hash, 32);
        sha256_final(&ctx, hash);

        //Check the difficulty
        k = 0;
        while(hash[k] == difficulty[k])
        {
            k++;
        }
        if(hash[k] < difficulty[k]) 
        {
            nr.nonce_found = true;
            nr.nonce = j;
            #ifdef MINING_MODE
            break;
            #endif
        }
    }
    tock();

    long int time = get_execution_time();
    printf("Total number of testing hashes = %ld\n", hashes);
    printf("Execution time = %ld seconds\n", time * 1e-9);
    printf("Hashrate = %f hashes/second\n", hashes / (time * 1e-9));

    if(nr.nonce_found) 
    {
        printf("Nonce found! %.8x\n", nr.nonce);
    }
    else 
    {
        printf("Nonce not found :(\n");
    }
}
