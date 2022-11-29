OBJECTS=gpu_miner cpu_miner pthread_cpu_miner hetero_miner

all: $(OBJECTS)

clean:
	 rm $(OBJECTS) sha256.o utils.o

hetero_miner: parallel_hetero.cu utils.o sha256.o
	 nvcc -O1 -v -lrt -lm -lpthread -arch=sm_35 -o $@ $^

gpu_miner: parallel_cuda.cu utils.o sha256.o
	 nvcc -O1 -v -lrt -lm -arch=sm_35 -o $@ $^

verify_gpu: parallel_cuda.cu utils.o sha256.o
	 nvcc -O1 -v -lrt -lm -D VERIFY_HASH -arch=sm_35 -o $@ $^

cpu_miner: serial_baseline.c sha256.o utils.o
	 gcc -O1 -v -o $@ $^ -lrt

pthread_cpu_miner: parallel_pthread.c sha256.o utils.o
	 gcc -O1 -v -lrt -lm -lpthread -o $@ $^ 

sha256.o: sha256.c
	 gcc -O1 -v -c -o $@ $^

utils.o: utils.c
	 gcc -O1 -v -c -o $@ $^ -lrt
