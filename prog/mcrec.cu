#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "global.h"
#include "mcrec.h"
#include "initial.h"
//#include <curand_kernel.h>
//#include <curand.h>


int main (int argc, char * argv[]){
    //begin time
    time(&beginTime);
    printf("begin time %s", asctime(localtime(&beginTime))); 
//openlog file
    logFile=fopen("calculation.log","w");
    //
    get_device_prop(deviceCount, deviceProp);
    if(deviceCount< 1 ){
        printf("No CUDA device is detected\n");
        return 1;
    }
    write_prop_log(deviceCount, deviceProp, logFile);
    //
//    cuErr = cudaMalloc(&gDBox, deviceCount * sizeof(gDoublebox));
//    if(cuErr != cudaSuccess){
//        printf("Cannot allocate gDBox %s line %d, err: %s, device %d\n", __FILE__, __LINE__, cudaGetErrorString(cuErr), deviceCount);
//    }

//    for(int i = 0; i < deviceCount; i++){
//        cuErr = cudaMalloc((void**)&gConf, deviceCount * sizeof(gOptions));
//        if(cuErr != cudaSuccess){
//            printf("Cannot allocate gConf %s line %d, err: %s, device %d\n", __FILE__, __LINE__, cudaGetErrorString(cuErr), deviceCount);
//        }
//    }
//    printf("test 16.02\n");
//    cuErr = cudaMalloc(&gTop, deviceCount * sizeof(gMolecula));
//    if(cuErr != cudaSuccess){
//        printf("Cannot allocate gConf %s line %d, err: %s, device %d\n", __FILE__, __LINE__, cudaGetErrorString(cuErr), deviceCount);
//    }
    
    gConf = (gOptions*) malloc(deviceCount * sizeof(gOptions));
    gTop = (gMolecula*) malloc(deviceCount * sizeof(gMolecula));
    gDBox = (gDoublebox*) malloc(deviceCount * sizeof(gDoublebox)); //allocate GPU data
    
    
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
    data_to_device(gBox, initFlows, gConf, config, gTop, hostParams, initMol, hostData, hostTop, hostConf, deviceCount, gDBox);
    cuErr = cudaGetLastError();
    printf("Cuda data2device last error: %s\n", cudaGetErrorString(cuErr));
    //calculate initial flows
    dim3 singleThread(config.singleXDim);
    printf(" grid %d  - %d \n", singleThread.x, singleThread.y);
    
    cuErr = cudaSetDevice(0);  //set to current device
    if(cuErr != cudaSuccess){
        printf("Cannot swtich to device %s line %d, err: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));
    }
    single_calc<<<config.flowNum, singleThread,0, gDBox[0].stream>>>(config.singleYDim, gConf[0], gTop[0], gBox);
    
    cudaDeviceSynchronize();
    cuErr = cudaGetLastError();
    printf("Cuda singlecalc last error: %s\n", cudaGetErrorString(cuErr));
    //get single box data from gpu
    data_from_device(gBox, hostData, config);
    //calculate pressure enegry correction
    rcut(hostData, config, hostTop, initMol, hostConf);
    //print out results
    write_singlebox_log(logFile, hostData);
    
    //
    for(int i = 0; i < config.flowNum; i++){
        printf("flow %d avergae energy %f\n",i, hostData.avEnergy[i]);
    }
    //Double box
    //initial inserting of molecules
    
    //push data to devices
    
    //start cycle
    
    
//close log file
    freeAll(gpuSingleBox,initFlows,config);
    fclose(logFile);
    //end time
    time(&endTime);
    printf("end time %s\n", asctime(localtime(&endTime))); 
    printf("elapsed time %f sec", difftime(endTime, beginTime));
    
    //generate initial stata on plates
    plates_initial_state(config, doubleBox, hostData, initMol, deviceCount);
    //int double_box_init_allocate(options &config, hDoubleBox &doubleBox, int deviceCount);
    double_box_init_allocate(config, doubleBox, deviceCount);
    printf("double allocate done\n");
    
//    int double_box_host_to_device(options &config, hDoubleBox &doubleBox, gDoublebox gDBox, gDoublebox hDBox, gSingleBox &hostData, molecules* initMol, int deviceCount);
    double_box_host_to_device(config, doubleBox, gDBox, hDBox, hostData, initMol, deviceCount);
    printf("host to device done\n");
    
    double_equilibration(gDBox, doubleBox, gConf, gTop);
    printf("equlibration done");
    
    //printf("%f - %f\n", hDBox.xa[0], hDBox.xm[0]);
    printf("Done\n");
}
