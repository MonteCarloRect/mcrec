#ifndef MC_H
#define MC_H

#include "global.h"

extern struct options{
    int subNum;
    char subFile[SUBNUMMAX][BUFFER];
    int flowNum;
    float* flowT;
    float* flowN;
    float* flowP;
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

#endif
