#include <stdio.h>
#include <stdlib.h>
#include "global.h"
#include "mcrec.h"
#include "initial.h"


int main (int argc, char * argv[]){

//openlog file
    logFile=fopen("calculation.log","w");
    //
    get_device_prop(deviceCount, deviceProp);
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
    initial_flows(config, initFlows,initMol,gpuSingleBox,paramsLines,allParams,gpuParams,hostParams, gpuMixParams, hostMixParams,deviceProp);
    printf("\n test %d \n", initFlows[0].molNum);
    //put data to device
//    data_to_device(initFlows,gpuSingleBox,config);
    
    printf("\n test2 %d %d \n", initFlows[0].molNum, config.flowNum);
    ///calculate initial flows
    dim3 singleThread(config.singleXDim);
    printf(" grid %d  - %d \n", singleThread.x, singleThread.y);
    single_calc<<<config.flowNum,singleThread>>>(gpuSingleBox,gpuParams,config.singleYDim);
//close log file
    freeAll(gpuSingleBox,initFlows,config);
    fclose(logFile);


}
