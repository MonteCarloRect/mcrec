#include <stdio.h>


//get device count and device properties
int get_device_prop(int &deviceCount, cudaDeviceProp* &deviceProp){
    //varaibles
    cudaError_t currentError;	//current error
    cudaDeviceProp temppd;	//temp varaible
    
    currentError=cudaGetDeviceCount(&deviceCount);
    printf("dev count in function %d \n", deviceCount);
    if (currentError!=cudaSuccess){
        fprintf(stderr,"Cannot get CUDA device count: %s\n", cudaGetErrorString(currentError));
        return 1;
    }
    if (!deviceCount){
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }
    deviceProp=(cudaDeviceProp*) malloc(deviceCount*sizeof(cudaDeviceProp));
    for (int i=0;i<deviceCount;i++){
        cudaGetDeviceProperties(&deviceProp[i],i);
        cudaGetDeviceProperties(&temppd,i);
        deviceProp[i]=temppd;
        printf("Device name %s \n", deviceProp[i].name);
        printf("Max Threads Dim: %d %d %d \n", deviceProp[i].maxThreadsDim[0],deviceProp[i].maxThreadsDim[1],deviceProp[i].maxThreadsDim[2]);
        printf("Max Grid Size: %d %d %d \n", deviceProp[i].maxGridSize[0], deviceProp[i].maxGridSize[1], deviceProp[i].maxGridSize[2]);
    }
    return 0;
}






