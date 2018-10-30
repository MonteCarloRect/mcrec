#include "../mcrec.h"

int data_to_device(gSingleBox &gBox, singleBox* &inputData, gOptions gConf, options &config, gMolecula gTop, potentialParam* Param, molecules* initMol){
    //allocate and copy data to GPU
    gSingleBox hostData;
    int sum;
    int id;
    //===========================MOLECULES
    //numbers of molecules
    hostData.molNum=(unsigned int*)malloc(config.flowNum * sizeof(int));
    for(int i=0;i<config.flowNum;i++){
        hostData.molNum[i]=inputData[i].molNum;
    }
    cudaMalloc(&gBox.molNum, config.flowNum*sizeof(int));
    cudaMemcpy(gBox.molNum, hostData.molNum, config.flowNum*sizeof(singleBox), cudaMemcpyHostToDevice);
    
    //coordinats of molecules
    sum=0;  //calculate total number oof molecules
    for(int i=0;i<config.flowNum;i++){
        sum+=inputData[i].molNum;
    }
    hostData.xm=(float*)malloc(sum * sizeof(float));
    hostData.ym=(float*)malloc(sum * sizeof(float));
    hostData.zm=(float*)malloc(sum * sizeof(float));
    hostData.mType=(unsigned int*)malloc(sum * sizeof(int));
    for(int i=0; i<config.flowNum; i++){
        for(int j=0; j<inputData[i].molNum; j++){
            id=i*config.flowNum+j;
            hostData.xm[id]=inputData[i].xm[j];
            hostData.ym[id]=inputData[i].ym[j];
            hostData.zm[id]=inputData[i].zm[j];
            hostData.mType[id]=inputData[i].type[j];
        }
    }
    cudaMalloc(&gBox.xm,  sum*sizeof(float));
    cudaMalloc(&gBox.ym,  sum*sizeof(float));
    cudaMalloc(&gBox.zm,  sum*sizeof(float));
    cudaMalloc(&gBox.mType,  sum*sizeof(int));
    
    cudaMemcpy(gBox.xm, hostData.xm, sum*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gBox.ym, hostData.ym, sum*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gBox.zm, hostData.zm, sum*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gBox.mType, hostData.mType, sum*sizeof(int), cudaMemcpyHostToDevice);
    //numbers of molecules of each type
    sum=0;
    for(int i=0; i<config.flowNum; i++){
        sum+=config.subNum;
    }
    hostData.typeMolNum=(unsigned int*) malloc (sum * sizeof(int));
    for(int i=0; i<config.flowNum; i++){
        for(int j=0; j<config.subNum; j++){
            id=config.flowNum*i+j;
            hostData.typeMolNum[id]=inputData[i].typeMolNum[j];
        }
    }
    cudaMalloc(&gBox.typeMolNum,  sum*sizeof(int));
    cudaMemcpy(gBox.typeMolNum, hostData.typeMolNum, sum*sizeof(int), cudaMemcpyHostToDevice);
    
    //type of atoms
    
    
    
    //atoms
    sum=0;
    for(int i=0; i<config.flowNum; i++){
        for(int j=0; j<config.subNum; j++){
            sum+=inputData[i].typeMolNum[j];
        }
    }
    hostData.xa=(float*) malloc(sum*sizeof(float));
    hostData.ya=(float*) malloc(sum*sizeof(float));
    hostData.za=(float*) malloc(sum*sizeof(float));
    hostData.aType=(unsigned int*) malloc(sum*sizeof(int));
    for(int i=0; i<config.flowNum; i++){
        for(int j=0; j<config.subNum; j++){
            for(int k=0; k<inputData[i].typeMolNum[j]; k++){
                id=i*(config.flowNum*config.subNum)+j*config.subNum+k;
                hostData.xa[id]=inputData[i].xa[j][k];
                hostData.ya[id]=inputData[i].ya[j][k];
                hostData.za[id]=inputData[i].za[j][k];
                hostData.aType[id]=inputData[i].aType[j][k];
            }
        }
    }
    cudaMalloc(&gBox.xa, sum*sizeof(float));
    cudaMalloc(&gBox.ya, sum*sizeof(float));
    cudaMalloc(&gBox.za, sum*sizeof(float));
    cudaMalloc(&gBox.aType, sum*sizeof(int));
    cudaMemcpy(gBox.xa, hostData.xa, sum*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gBox.ya, hostData.ya, sum*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gBox.za, hostData.za, sum*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gBox.aType, hostData.aType, sum*sizeof(int), cudaMemcpyHostToDevice);
    
    //============================TOPOLOGY
    gMolecula hostTop;
    //matrix
    hostTop.sigma=(float*) malloc (config.potNum * config.potNum * sizeof(float));
    hostTop.epsi=(float*) malloc (config.potNum * config.potNum * sizeof(float));
    hostTop.charge=(float*) malloc (config.potNum * config.potNum * sizeof(float));
    //single atom
    hostTop.aNum=(unsigned int*) malloc (config.subNum * sizeof(int));
    for(int i=0; i<config.subNum; i++){
        hostTop.aNum[i]=initMol[i].atomNum;
    }
    
//    printf("potnum %d\n", config.potNum);
    for(int i=0; i<config.potNum; i++){
        for(int j=0; j<config.potNum; j++){
            id=i*config.potNum+j;
//            printf("id %d i %d j %f\n", id, i, Param[i].sigma);
            //add mixture rule
            hostTop.sigma[id]=0.5*(Param[i].sigma+Param[j].sigma);
//            printf("sigma %f\n", hostTop.sigma[id]);
            hostTop.epsi[id]=sqrt(Param[i].epsilon*Param[j].epsilon);
//            printf("epsi %f\n", hostTop.epsi[id]);
            hostTop.charge[id]=Param[i].charge*Param[i].charge;
//            printf("charge %f\n", hostTop.charge[id]);
            //add mixture rule
        }
    }
//    printf("OLOLO5\n");
    //gpu data
    cudaMalloc(&gTop.sigma, config.potNum * config.potNum * sizeof(float));
    cudaMalloc(&gTop.epsi, config.potNum * config.potNum * sizeof(float));
    cudaMalloc(&gTop.charge, config.potNum * config.potNum * sizeof(float));
    cudaMalloc(&gTop.aNum, config.subNum * sizeof(int));
    
    cudaMemcpy(gTop.sigma, hostTop.sigma, config.potNum * config.potNum * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gTop.epsi, hostTop.epsi, config.potNum * config.potNum * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gTop.charge, hostTop.charge, config.potNum * config.potNum * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gTop.aNum, hostTop.aNum, config.subNum * sizeof(int), cudaMemcpyHostToDevice);
    
    //============================CONFIG
    gOptions hostConf;
    hostConf.Temp=(float*) malloc(config.flowNum*sizeof(float));
    for(int i=0; i<config.flowNum; i++){
        hostConf.Temp[i]=config.flowT[i];
    }
    cudaMalloc(&gConf.Temp, config.flowNum*sizeof(float));
    cudaMemcpy(gConf.Temp, hostConf.Temp, config.flowNum*sizeof(float), cudaMemcpyHostToDevice);
    
    return 0;
}



