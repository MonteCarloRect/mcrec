#include <stdio.h>
#include <stdlib.h>
#include "global.h"
#include "mcrec.h"
#include "initial.h"
//#include <curand_kernel.h>
//#include <curand.h>


int main (int argc, char * argv[]){

//openlog file
    logFile=fopen("calculation.log","w");
    //
    get_device_prop(deviceCount, deviceProp);
    if(deviceCount<1){
        printf("No CUDA device is detected\n");
        return 1;
    }
    write_prop_log(deviceCount,deviceProp,logFile);
    //read initial data
    read_options(config);
    write_config_log(config,logFile);
    //read gro data for each molecules
    initMol = (molecules *) malloc(config.subNum * sizeof(molecules));
    read_init_gro(config, initMol);
    //read topology of molecules
    read_top(allParams,paramsLines);
    for(int i=0;i<paramsLines;i++){
        printf("  %s \n",allParams[i].aName);
    }
    printf("number of top lines %d\n",paramsLines);
    //create initial simulation
    initial_flows(config, initFlows, initMol, gpuSingleBox, paramsLines, allParams, gpuParams, hostParams, gpuMixParams, hostMixParams, deviceProp);
    //copy data to GPU device
    data_to_device(gBox, initFlows,gConf, config, gTop, hostParams, initMol);
    cuErr = cudaGetLastError();
    printf("Cuda data2device last error: %s\n", cudaGetErrorString(cuErr));
    //calculate initial flows
    dim3 singleThread(config.singleXDim);
    printf(" grid %d  - %d \n", singleThread.x, singleThread.y);
    single_calc<<<config.flowNum, singleThread>>>(config.singleYDim, gConf, gTop, gBox);
    cudaDeviceSynchronize();
    cuErr = cudaGetLastError();
    printf("Cuda singlecalc last error: %s\n", cudaGetErrorString(cuErr));
    printf("ololo\n");
//close log file
    freeAll(gpuSingleBox,initFlows,config);
    fclose(logFile);


}
