#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
	    timer.start();
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            printf("labwork 3 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            break;
        case 4:
	    timer.start();
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            printf("labwork 4 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            break;
        case 5:
	    timer.start();
            labwork.labwork5_CPU();
            labwork.saveOutputImage("labwork5-cpu-out.jpg");
            printf("labwork 5 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
	    labwork.labwork5_GPU();
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            printf("labwork 5 GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {		// let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    #pragma omp parallel for
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        #pragma omp parallel for
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of GPU %d\n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, i);
	printf("\nDevice number #%d\n", i);
	printf("Device name: %s\n", prop.name);
	printf("Clock rate: %d\n", prop.clockRate);
	printf("Processor count: %d\n", getSPcores(prop));
	printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
	printf("Warp size: %d\n", prop.warpSize); 
    }    
}

__global__ void grayscale(uchar3* input, uchar3* output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork3_GPU() {
    // inputImage->width, inputImage->height    
	int pixelCount = inputImage->width * inputImage->height;
	outputImage = static_cast<char *>(malloc(pixelCount * 3));
	int blockSize = 1024;
	int numBlock = pixelCount / blockSize;

    // cuda malloc: devInput, devOutput
	uchar3 *devInput;
	uchar3 *devOutput;
	cudaMalloc(&devInput, inputImage->width * inputImage->height * 3);	
	cudaMalloc(&devOutput, inputImage->width * inputImage->height * 3);

    // cudaMemcpy: inputImage (hostInput) -> devInput
	cudaMemcpy(devInput, inputImage->buffer, inputImage->width * inputImage->height * 3, cudaMemcpyHostToDevice);

    // launch the kernel
	grayscale<<<numBlock, blockSize>>>(devInput, devOutput);

    // cudaMemcpy: devOutput -> inputImage (host)
	cudaMemcpy(outputImage, devOutput, inputImage->width * inputImage->height * 3, cudaMemcpyDeviceToHost);

    // cudaFree
	cudaFree(&devInput);
	cudaFree(&devOutput);

}

__global__ void grayscale2d(uchar3* input, uchar3* output, int width, int height) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx > width) return;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy > height) return;
    int tid = tidx + tidy * width;
    output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork4_GPU() {
    // inputImage->width, inputImage->height    
	int pixelCount = inputImage->width * inputImage->height;
	outputImage = static_cast<char *>(malloc(pixelCount * 3));
	//int blockSize = 1024;
	//int numBlock = pixelCount / blockSize;
	dim3 blockSize = dim3(32, 32);
	dim3 gridSize = dim3((inputImage->width + 31) / blockSize.x, (inputImage->height + 31) / blockSize.y);

    // cuda malloc: devInput, devOutput
	uchar3 *devInput;
	uchar3 *devOutput;
	cudaMalloc(&devInput, inputImage->width * inputImage->height * 3);	
	cudaMalloc(&devOutput, inputImage->width * inputImage->height * 3);

    // cudaMemcpy: inputImage (hostInput) -> devInput
	cudaMemcpy(devInput, inputImage->buffer, inputImage->width * inputImage->height * 3, cudaMemcpyHostToDevice);

    // launch the kernel
	grayscale2d<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height);

    // cudaMemcpy: devOutput -> inputImage (host)
	cudaMemcpy(outputImage, devOutput, inputImage->width * inputImage->height * 3, cudaMemcpyDeviceToHost);

    // cudaFree
	cudaFree(&devInput);
	cudaFree(&devOutput);
   
}



// CPU implementation of Gaussian Blur
void Labwork::labwork5_CPU() {
    int kernel[] = { 0, 0, 1, 2, 1, 0, 0,  
                     0, 3, 13, 22, 13, 3, 0,  
                     1, 13, 59, 97, 59, 13, 1,  
                     2, 22, 97, 159, 97, 22, 2,  
                     1, 13, 59, 97, 59, 13, 1,  
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0 };
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = (char*) malloc(pixelCount * sizeof(char) * 3);
    for (int row = 0; row < inputImage->height; row++) {
        for (int col = 0; col < inputImage->width; col++) {
            int sum = 0;
            int c = 0;
            for (int y = -3; y <= 3; y++) {
                for (int x = -3; x <= 3; x++) {
                    int i = col + x;
                    int j = row + y;
                    if (i < 0) continue;
                    if (i >= inputImage->width) continue;
                    if (j < 0) continue;
                    if (j >= inputImage->height) continue;
                    int tid = j * inputImage->width + i;
                    unsigned char gray = (inputImage->buffer[tid * 3] + inputImage->buffer[tid * 3 + 1] + inputImage->buffer[tid * 3 + 2])/3;
                    int coefficient = kernel[(y+3) * 7 + x + 3];
                    sum = sum + gray * coefficient;
                    c += coefficient;
                }
            }
            sum /= c;
            int posOut = row * inputImage->width + col;
            outputImage[posOut * 3] = outputImage[posOut * 3 + 1] = outputImage[posOut * 3 + 2] = sum;
        }
    }
}

__global__ void blur(uchar3* input, uchar3* output, int width, int height) {
    int kernel[] = { 0, 0, 1, 2, 1, 0, 0,  
                     0, 3, 13, 22, 13, 3, 0,  
                     1, 13, 59, 97, 59, 13, 1,  
                     2, 22, 97, 159, 97, 22, 2,  
                     1, 13, 59, 97, 59, 13, 1,  
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0 };

    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx > width) return;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy > height) return;
    int tid = tidx + tidy * width;

    int s = 0;
    int weightsSum = 0;

    for (int row = -3; row <= 3; row++) {
	for (int col = -3; col <= 3; col++) {
	    int tempTid = tid + row * width + col;
	    if (tempTid < 0) continue;
	    if (tempTid >= width * height) continue;

	    int gray = (input[tempTid].x + input[tempTid].y + input[tempTid].z) / 3;
	    s += gray * kernel[(row + 3) * 7 + col + 3];
	    weightsSum += kernel[(row + 3) * 7 + col + 3];
	}
    }

    s /= weightsSum;
    output[tid].x = output[tid].y = output[tid].z = s;
}

__global__ void blurShared(uchar3* input, uchar3* output, int* coefficients, int width, int height) {

    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx > width) return;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy > height) return;
    int tid = tidx + tidy * width;

    int s = 0;
    int weightsSum = 0;

    int localTid = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ int scoefficients[49];
    if (localTid < 49) {
	scoefficients[localTid] = coefficients[localTid];
    }
    __syncthreads();    

    for (int row = -3; row <= 3; row++) {
	for (int col = -3; col <= 3; col++) {
	    int tempTid = tid + row * width + col;
	    //int tempTid = (row + 3) * 7 + col + 30;
	    if (tempTid < 0) continue;
	    if (tempTid >= width * height) continue;

	    int gray = (input[tempTid].x + input[tempTid].y + input[tempTid].z) / 3;
	    s += gray * scoefficients[(row + 3) * 7 + col + 3];
	    weightsSum += scoefficients[(row + 3) * 7 + col + 3];
	}
    }

    s /= weightsSum;
    output[tid].x = output[tid].y = output[tid].z = s;
}

void Labwork::labwork5_GPU() {
    int coefficients[] = { 0, 0, 1, 2, 1, 0, 0,  
                     0, 3, 13, 22, 13, 3, 0,  
                     1, 13, 59, 97, 59, 13, 1,  
                     2, 22, 97, 159, 97, 22, 2,  
                     1, 13, 59, 97, 59, 13, 1,  
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0 };


    // inputImage->width, inputImage->height    
	int pixelCount = inputImage->width * inputImage->height;
	outputImage = static_cast<char *>(malloc(pixelCount * 3));
	//int blockSize = 1024;
	//int numBlock = pixelCount / blockSize;
	dim3 blockSize = dim3(32, 32);
	dim3 gridSize = dim3((inputImage->width + 31) / blockSize.x, (inputImage->height + 31) / blockSize.y);

    // cuda malloc: devInput, devOutput
	uchar3 *devInput;
	uchar3 *devOutput;
	cudaMalloc(&devInput, inputImage->width * inputImage->height * 3);	
	cudaMalloc(&devOutput, inputImage->width * inputImage->height * 3);

    // cudaMemcpy: inputImage (hostInput) -> devInput
	cudaMemcpy(devInput, inputImage->buffer, inputImage->width * inputImage->height * 3, cudaMemcpyHostToDevice);

    // launch the kernel
	blurShared<<<gridSize, blockSize>>>(devInput, devOutput, coefficients, inputImage->width, inputImage->height);

    // cudaMemcpy: devOutput -> inputImage (host)
	cudaMemcpy(outputImage, devOutput, inputImage->width * inputImage->height * 3, cudaMemcpyDeviceToHost);

    // cudaFree
	cudaFree(&devInput);
	cudaFree(&devOutput);
    
}

void Labwork::labwork6_GPU() {

}

void Labwork::labwork7_GPU() {

}

void Labwork::labwork8_GPU() {

}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU() {

}
