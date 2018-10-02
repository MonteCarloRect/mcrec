//
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "../mcrec.h"
#include <time.h>
//#include "../initial.h"

int initial_flows(options &config, singleBox* &initFlows,molecules* initMol, singleBox* &gpuSingleBox, int lines, potentialParam* allParams, potentialParam* gpuParams, potentialParam* hostParams, mixParam** gpuMixParams, mixParam** hostMixParams,cudaDeviceProp* deviceProp){
    float* l_x; //latice coords
    float* l_y;
    float* l_z;
    int* l_t; //0 --- empty
    int latice;
    int latice3;
    float laticeDelta;
    int id;
    int sum;
    int moleculePerBox;
    int randMol;
    int l_type;
    int* linesList;
    int linesEnd;
    int checked;
    //int sizey;
    
    config.singleXDim=ceil(deviceProp[0].maxThreadsPerBlock);
    config.singleYDim=ceil(2000/config.singleXDim)+1;
    moleculePerBox=config.singleYDim*config.singleYDim;
    srand(time(0));
//    printf(" ydim  %d \n", config.singleYDim);
//    getchar();
    //create arrays
    initFlows=(singleBox*) malloc(config.flowNum * sizeof(singleBox));
    srand(time(0));
    //create arrays
//    moleculesNumber=(int*) malloc(config.subNum * sizeof(int));
    for(int i=0; i<config.flowNum;i++){
        initFlows[i].molNum=moleculePerBox;
        //get latice size 
        latice=ceil(pow(initFlows[i].molNum,1.0/3.0));
//        latice=ceil(pow(initFlows[i].molNum,1.0/3.0))+1;
        latice3=latice*latice*latice;
        l_x = (float*) malloc(latice3*sizeof(float));
        l_y = (float*) malloc(latice3*sizeof(float));
        l_z = (float*) malloc(latice3*sizeof(float));
        l_t = (int*) malloc(latice3*sizeof(int));  //check place busy
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
                    l_t[id]=0;    //set place not busy
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
        initFlows[i].aType=(int**) malloc(initFlows[i].molNum*sizeof(int*));
        initFlows[i].aNum=(int*)malloc(initFlows[i].molNum*sizeof(int));
        printf("num of mol %d\n", initFlows[i].molNum);
        //set atoms to places
        for(int j=0; j<initFlows[i].molNum;j++){    //check all molecules
            //
            //
            
            l_type=initFlows[i].type[j];  //get molecules type
            initFlows[i].aNum[j]=initMol[l_type].atomNum;
            initFlows[i].xa[j]=(float*) malloc(initMol[l_type].atomNum*sizeof(float));
            initFlows[i].ya[j]=(float*) malloc(initMol[l_type].atomNum*sizeof(float));
            initFlows[i].za[j]=(float*) malloc(initMol[l_type].atomNum*sizeof(float));
            initFlows[i].aType[j]=(int*) malloc(initMol[l_type].atomNum*sizeof(int));
            //printf("type %d numbers of atom in molecule %d\n",l_type,initMol[l_type].atomNum);
            for(int k=0;k<initMol[l_type].atomNum;k++){
                initFlows[i].xa[j][k]=initMol[l_type].x[k];
                initFlows[i].ya[j][k]=initMol[l_type].y[k];
                initFlows[i].za[j][k]=initMol[l_type].z[k];
            }
        }
    }
    //getchar();
    //data to GPU
    cudaMalloc(&gpuSingleBox, config.flowNum*sizeof(singleBox));
    cudaMemcpy(gpuSingleBox,initFlows,config.flowNum*sizeof(singleBox),cudaMemcpyHostToDevice);
    
//    //check used atoms
//    for(int i=0;i<config.subNum;i++){
//        for(int j=0;j<initMol[i].atomNum;j++){
//            
//        }
//    }
    linesList=(int*) malloc (lines*sizeof(int));
    //check used atoms
    id=0;
    for(int i=0;i<lines;i++){
        checked=0;
        for(int sub=0;sub<config.subNum;sub++){
            for(int at=0;at<initMol[sub].atomNum;at++){
//                if(allParams[i].aName==initMol[sub].atomNum)
//                printf("|%s|%s|%d|%d\n",allParams[i].aName,initMol[sub].aName[at], strcmp(initMol[sub].aName[at],allParams[i].aName),strcmp(allParams[i].aName,initMol[sub].aName[at]));
                if(strcmp(initMol[sub].aName[at],allParams[i].aName)==0 || abs(strcmp(initMol[sub].aName[at],allParams[i].aName))==127){
                    printf("i %d sub %d a %d %s  %s \n",i,sub, at, allParams[i].aName,initMol[sub].aName[at]);
                    //add line from top file to gpu
                    checked=1;
                }
            }
        }
        if(checked==1){
            linesList[id]=i;
            id++;
        }
    }
    linesEnd=id;
    //host parameters
    hostParams=(potentialParam*)malloc(linesEnd*sizeof(potentialParam));
    for(int i=0;i<linesEnd;i++){
        hostParams[i]=allParams[linesList[i]];
        printf("host aname %d %s\n",i, hostParams[i].aName);
    }
    //add mixrule
    //mixParam hostMixParams[linesEnd][linesEnd];
    hostMixParams = (mixParam**) malloc(linesEnd*sizeof(mixParam*));
    for(int i=0;i<linesEnd;i++){
        hostMixParams[i]=(mixParam*) malloc(linesEnd*sizeof(mixParam));
        for(int j=0;j<linesEnd;j++){
            if(config.mixRule==LB){
                hostMixParams[i][j].sigma=0.5*(hostParams[i].sigma+hostParams[j].sigma);
                hostMixParams[i][j].epsilon=sqrt(hostParams[i].epsilon*hostParams[j].epsilon);
                hostMixParams[i][j].alpha=0.0;
                hostMixParams[i][j].charge=hostParams[i].charge*hostParams[j].charge;
            }
        }
    }
    //add atom types
    for(int i=0;i<config.flowNum;i++){  //flow
        for(int j=0;j<initFlows[i].molNum;j++){ //molecule
            
            for(int k=0;k<initFlows[i].aNum[j];k++){    //atom
                //get curent atom name
                //initMol[initFlows[i].type[j]].aName[k]
                for(int chk=0;chk<linesEnd;chk++){  //check from list
                    if(strcmp(initMol[initFlows[i].type[j]].aName[k],hostParams[chk].aName)==0 || abs(strcmp(initMol[initFlows[i].type[j]].aName[k],hostParams[chk].aName))==127 ){
                        
                    }
                }
                
            }
            
        }
        
    }
    
    //malloc gpu mix params
    cudaMalloc(&gpuMixParams,linesEnd*linesEnd*sizeof(mixParam));
    cudaMemcpy(gpuMixParams,hostMixParams,linesEnd*linesEnd*sizeof(mixParam),cudaMemcpyHostToDevice);
//    for(int i=0;i<linesEnd;i++){
//        cudaMalloc(&gpuMixParams[i],linesEnd*sizeof(mixParam));
//    }
    
    //malloc parameters
//    cudaMalloc(&gpuParams,linesEnd*sizeof(potentialParam));
//    cudaMemcpy(gpuParams,hostParams,linesEnd*sizeof(potentialParam),cudaMemcpyHostToDevice);
    
//    cudaMemcpy(gpuParams,)
    
//        //get numbers of molecules of each substance
//        sum=0;
//        for(int s_i=0; s_i<config.subNum-1;s_i++){
//            moleculesNumber[s_i]=ceil(config.flowX[i][s_i]*initFlows[i].molNum);
//            sum+=moleculesNumber[s_i];
//        }
//        moleculesNumber[config.subNum-1]=initFlows[i].molNum - sum;
//        //set molecules to placees
//        id=0;
//        for(int cur_sub=0;cur_sub<config.subNum;cur_sub++){
//            set=0;
//            while(set<moleculesNumber[cur_sub]){
//                //get random position
//                curRand = rand() % latice3;
//                if(check[curRand]/=-1){
//                    check[curRand]=cur_sub;
//                    initFlows[i].xm[id]=l_x[curRand];
//                    initFlows[i].ym[id]=l_y[curRand];
//                    initFlows[i].zm[id]=l_z[curRand];
//                    id++;
//                    set++;
////                    printf("set %d %d\n",set,curRand);
//                }
//            }
//        }
//    }
        
        
    return 1;
}


