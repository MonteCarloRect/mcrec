//
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "../mcrec.h"

int initial_flows(options config, singleBox* &initFlows,molecules* initMol, singleBox* &gpuSingleBox){
    float* l_x; //latice coords
    float* l_y;
    float* l_z;
    int* l_t; //0 --- empty
    int latice;
    int latice3;
    float laticeDelta;
    int id;
    int moleculePerBox;
    int sum;
    int randMol;
    int l_type;
    
    moleculePerBox=2000;
    srand(time(0));
    
    //create arrays
    initFlows=(singleBox*) malloc(config.flowNum * sizeof(singleBox));
    for(int i=0; i<config.flowNum;i++){
        initFlows[i].molNum=moleculePerBox;
        //get latice size 
        latice=ceil(pow(initFlows[i].molNum,1.0/3.0));
        latice3=latice*latice*latice;
        l_x = (float*) malloc(latice3*sizeof(float));
        l_y = (float*) malloc(latice3*sizeof(float));
        l_z = (float*) malloc(latice3*sizeof(float));
        l_t = (int*) malloc(latice3*sizeof(int));
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
                    l_t[id]=0;
                    id++;
                }
            }
        }
        initFlows[i].xm=(float*) malloc(initFlows[i].molNum*sizeof(float));
        initFlows[i].ym=(float*) malloc(initFlows[i].molNum*sizeof(float));
        initFlows[i].zm=(float*) malloc(initFlows[i].molNum*sizeof(float));
        initFlows[i].typeMolNum=(int*) malloc(config.subNum*sizeof(int));
        initFlows[i].type=(int*) malloc(initFlows[i].molNum*sizeof(int));
        //get numbers of molecules of each types
        sum=0;
        for(int j=0;j<config.subNum-1;j++){
            initFlows[i].typeMolNum[j]=initFlows[i].molNum*config.flowX[i][j];
            sum+=initFlows[i].typeMolNum[j];
        }
        initFlows[i].typeMolNum[config.subNum-1]=initFlows[i].molNum-sum;
        //set molecules to places
        id=0;
        for(int j=0;j<config.subNum;j++){
            sum=0;
            while(sum<initFlows[i].typeMolNum[j]){
                randMol=rand() % latice3;
                
                if(l_t[randMol]==0){
//                    printf("random n %d \n", randMol);
                    initFlows[i].xm[id]=l_x[randMol];   //set ccordinates
                    initFlows[i].ym[id]=l_y[randMol];
                    initFlows[i].zm[id]=l_z[randMol];
                    l_t[randMol]=1; //now plase not empty
                    initFlows[i].type[id]=i;
                    id++;
                    sum++;
                }
            }
        }
        initFlows[i].xa=(float**) malloc(initFlows[i].molNum*sizeof(float*));
        initFlows[i].ya=(float**) malloc(initFlows[i].molNum*sizeof(float*));
        initFlows[i].za=(float**) malloc(initFlows[i].molNum*sizeof(float*));
        //set atoms to places
        for(int j=0; j<initFlows[i].molNum;j++){    //check all molecules
            l_type=initFlows[i].type[j];  //get molecules type
            initFlows[i].xa[j]=(float*) malloc(initMol[l_type].atomNum*sizeof(float));
            initFlows[i].ya[j]=(float*) malloc(initMol[l_type].atomNum*sizeof(float));
            initFlows[i].za[j]=(float*) malloc(initMol[l_type].atomNum*sizeof(float));
            for(int k=0;k<initMol[l_type].atomNum;k++){
                initFlows[i].xa[j][k]=initMol[l_type].x[k];
                initFlows[i].ya[j][k]=initMol[l_type].y[k];
                initFlows[i].za[j][k]=initMol[l_type].z[k];
            }
        }
    }
    //data to GPU
    cudaMalloc(&gpuSingleBox, config.flowNum*sizeof(singleBox));
    cudaMemcpy(gpuSingleBox,initFlows,config.flowNum*sizeof(singleBox),cudaMemcpyHostToDevice);
    
    //check used atoms
    for(int i=0;i<config.subNum;i++){
        for(int j=0;j<initMol[i].atomNum;j++){
            
        }
    
    }
        
        
    return 1;
}
