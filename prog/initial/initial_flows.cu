//
#include <stdio.h>
#include <math.h>
#include "../mcrec.h"
#include <time.h>

int initial_flows(options config, flows* &initFlows){
    float* l_x; //latice coords
    float* l_y;
    float* l_z;
    int latice;
    float laticeDelta;
    int id;
    int* check;
    int* moleculesNumber;
    int sum;    //temporally sum
    int set;    //seted current
    int curRand;
    int latice3;
    
    srand(time(0));
    //create arrays
    initFlows=(flows*) malloc(config.flowNum * sizeof(flows));  
    moleculesNumber=(int*) malloc(config.subNum * sizeof(int));
    for(int i=0; i<config.flowNum;i++){
        initFlows[i].molNum=2000;
        //get latice size 
        latice=ceil(pow(initFlows[i].molNum,1.0/3.0))+1;
        latice3=latice*latice*latice;
        l_x = (float*) malloc(latice3*sizeof(float));
        l_y = (float*) malloc(latice3*sizeof(float));
        l_z = (float*) malloc(latice3*sizeof(float));
        //define system size
        if(config.flowEns[i]==NVT){
            laticeDelta=pow(initFlows[i].molNum/(config.flowN[i]*NA/1.0e24),1.0/3.0)/latice;
            printf("latice delta %f\n", laticeDelta);
        }
        //get coordinats
        check=(int*) malloc(latice3*sizeof(int));  //check place busy
        id=0;
        for(int l_i=0; l_i<latice; l_i++){
            for(int l_j=0; l_j<latice; l_j++){
                for(int l_k=0; l_k<latice; l_k++){
                    l_x[id]=l_i*laticeDelta;
                    l_y[id]=l_j*laticeDelta;
                    l_z[id]=l_k*laticeDelta;
                    check[id]=-1;    //set place not busy
                    id++;
                }
            }
        }
        initFlows[i].xm=(float*) malloc(initFlows[i].molNum*sizeof(float));
        initFlows[i].ym=(float*) malloc(initFlows[i].molNum*sizeof(float));
        initFlows[i].zm=(float*) malloc(initFlows[i].molNum*sizeof(float));
        //get numbers of molecules of each substance
        sum=0;
        for(int s_i=0; s_i<config.subNum-1;s_i++){
            moleculesNumber[s_i]=ceil(config.flowX[i][s_i]*initFlows[i].molNum);
            sum+=moleculesNumber[s_i];
        }
        moleculesNumber[config.subNum-1]=initFlows[i].molNum - sum;
        //set molecules to placees
        id=0;
        for(int cur_sub=0;cur_sub<config.subNum;cur_sub++){
            set=0;
            while(set<moleculesNumber[cur_sub]){
                //get random position
                curRand = rand() % latice3;
                if(check[curRand]/=-1){
                    check[curRand]=cur_sub;
                    initFlows[i].xm[id]=l_x[curRand];
                    initFlows[i].ym[id]=l_y[curRand];
                    initFlows[i].zm[id]=l_z[curRand];
                    id++;
                    set++;
//                    printf("set %d %d\n",set,curRand);
                }
            }
        }
    }
    return 1;
}
