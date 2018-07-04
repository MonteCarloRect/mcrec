//
#include <stdio.h>
#include <math.h>
#include "../mcrec.h"

int initial_flows(options config, flows* &initFlows){
    float* l_x; //latice coords
    float* l_y;
    float* l_z;
    int latice;
    float laticeDelta;
    int id;
    //create arrays
    initFlows=(flows*) malloc(config.flowNum * sizeof(flows));
    for(int i=0; i<config.flowNum;i++){
        initFlows[i].molNum=2000;
        //get latice size 
        latice=ceil(pow(initFlows[i].molNum,1.0/3.0));
        l_x = (float*) malloc(latice*latice*latice*sizeof(float));
        l_y = (float*) malloc(latice*latice*latice*sizeof(float));
        l_z = (float*) malloc(latice*latice*latice*sizeof(float));
        //define system size
        if(config.flowEns[i]==NVT){
            laticeDelta=pow(initFlows[i].molNum/(config.flowN[i]*NA/1.0e24),1.0/3.0)/latice;
            printf("latice delta %f\n", laticeDelta);
        }
        //get coordinats
        id=0;
        for(int l_i=0; l_i<latice; l_i++){
            for(int l_j=0; l_j<latice; l_j++){
                for(int l_k=0; l_k<latice; l_k++){
                    l_x[id]=l_i*laticeDelta;
                    l_y[id]=l_j*laticeDelta;
                    l_z[id]=l_k*laticeDelta;
                    id++;
                }
            }
        }
        initFlows[i].xm=(float*) malloc(initFlows[i].molNum*sizeof(float));
        initFlows[i].ym=(float*) malloc(initFlows[i].molNum*sizeof(float));
        initFlows[i].zm=(float*) malloc(initFlows[i].molNum*sizeof(float));
    }
    return 1;
}
