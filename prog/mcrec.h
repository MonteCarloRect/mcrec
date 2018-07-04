#ifndef MC_H
#define MC_H

#include "global.h"

extern struct options{
    int subNum;
    char subFile[SUBNUMMAX][BUFFER];
    int flowNum;
    int* flowEns;    //flow ensample
    float* flowT;   //flow temperature
    float* flowN;   //flow density [mol/l]
    float* flowP;   //flow pressure [bar]
    int* flowIns;
    float** flowX;
} config;

extern struct molecules{
    int atomNum;
    float* x;
    float* y;
    float* z;
    float* vx;
    float* vy;
    float* vz;
} initial;

extern struct flows{
    int molNum;
    float* xm;  //molecule coords
    float* ym;
    float* zm;
    float** xa;  //atom coords
    float** ya;
    float** za;
    int vaporNum;   //molecules in vapor phase
    int liquidNum;  //molecules in liquid phase
    int* vaporList; //list of molecules in vapor phase
    int* liquidList; //list of molecules in liquid phase
} initialFlows;

extern int deviceCount;
extern cudaDeviceProp* deviceProp;	//array of device properties

extern FILE* logFile;

//functions
//initial
int get_device_prop(int &deviceCount, cudaDeviceProp* &deviceProp);
int read_options(options &config);
int read_init_gro(options config, molecules* &initial);
//write
int write_prop_log(int deviceCount, cudaDeviceProp* deviceProp,FILE* logFile);
int write_config_log(options con,FILE* logFile);

//flows
int initial_flows(options config, flows* &initFlows);

#endif
