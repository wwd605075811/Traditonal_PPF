#include "F1.cuh"

__global__ void kernelAq(int * globalArray){
    int globalThreadId = blockIdx.x * blockDim.y + threadIdx.x;
    printf("blockIdx.x:%d * blockDim.y:%d + threadIdx.x:%d = globalThreadId:%d\n", blockIdx.x, blockDim.y, threadIdx.x, globalThreadId);
    globalArray[globalThreadId] = globalThreadId;
}

void m()
{
    int elementCount = 32;
    int dataSize = elementCount * sizeof(int);

    cudaSetDevice(0);

    int * managedArray;
    int *m_h=(int*)malloc (dataSize);
    cudaMalloc((int**)&managedArray, dataSize);
    kernelAq <<<4,8>>>(managedArray);
    cudaDeviceSynchronize();
    cudaMemcpy(m_h,managedArray,dataSize,cudaMemcpyDeviceToHost);

    cout<<"同步完成！"<<endl;

    // Printing a portion of results can be another good debugging approach
    for(int i = 0; i < elementCount; i++){
        printf("%d%s", m_h[i], (i < elementCount - 1) ? ", " : "\n");
    }

    cudaFree(managedArray);
    free(m_h);
    cudaDeviceReset();
}


bool Iqq()
{

    // Set which device should be used
    // The code will default to 0 if not called though
    cudaSetDevice(0);
    // Call a device function from the host: a kernel launch
    // Which will print from the device
    kernelAq <<<2,33>>>();

    // This call waits for all of the submitted GPU work to complete
    cudaDeviceSynchronize();

    // Destroys and cleans up all resources associated with the current device.
    // It will reset the device immediately. It is the caller's responsibility
    //    to ensure that the device work has completed
    cudaDeviceReset();

    return true;
}